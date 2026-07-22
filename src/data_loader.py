import logging

import duckdb
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def build_multihot_features(con, table, id_col, val_col, valid_ids, top_k, trim=None):
    """Generic function to build Top-K multi-hot encoding using DuckDB"""
    query_trim = (
        f"SUBSTRING(CAST({val_col} AS VARCHAR), 1, {trim})"
        if trim
        else f"CAST({val_col} AS VARCHAR)"
    )

    vocab_df = con.query(f"""
        SELECT {query_trim} as code, count(*) as cnt
        FROM read_csv_auto('/home/hanwen/data/mimic/iii/{table}.csv', sample_size=-1)
        WHERE {id_col} IN {valid_ids} AND {val_col} IS NOT NULL
        GROUP BY code ORDER BY cnt DESC LIMIT {top_k}
    """).df()
    vocab = vocab_df["code"].tolist()

    raw_df = con.query(f"""
        SELECT {id_col} as target_id, {query_trim} as code
        FROM read_csv_auto('/home/hanwen/data/mimic/iii/{table}.csv', sample_size=-1)
        WHERE {id_col} IN {valid_ids} AND {query_trim} IN {tuple(vocab) if len(vocab)>1 else f"('{vocab[0]}')"}
    """).df()

    feature_dict = {}
    for tid, group in raw_df.groupby("target_id"):
        vec = np.zeros(len(vocab), dtype=np.float32)
        for code in group["code"]:
            if code in vocab:
                vec[vocab.index(code)] = 1.0
        feature_dict[tid] = vec

    return feature_dict, len(vocab)


def enrich_stays_with_features(stays_df, con):
    logging.info("Computing additional features for stays...")

    # Register temporary table for DuckDB
    con.register("stays_df_tmp", stays_df[["HADM_ID", "stay_id", "INTIME"]])

    transfers_df = con.query("""
        SELECT t.HADM_ID, count(*) as pre_icu_transfers
        FROM read_csv_auto('/home/hanwen/data/mimic/iii/TRANSFERS.csv', sample_size=-1) t
        JOIN stays_df_tmp s ON t.HADM_ID = s.HADM_ID
        WHERE CAST(t.OUTTIME AS TIMESTAMP) <= CAST(s.INTIME AS TIMESTAMP)
        GROUP BY t.HADM_ID
    """).df()

    surg_df = con.query("""
        SELECT 
            s.HADM_ID, 
            count(p.ITEMID) as surg_count,
            max(CAST(p.ENDTIME AS TIMESTAMP)) as last_surg_endtime
        FROM stays_df_tmp s
        JOIN read_csv_auto('/home/hanwen/data/mimic/iii/PROCEDUREEVENTS_MV.csv', sample_size=-1) p 
            ON s.HADM_ID = p.HADM_ID AND CAST(p.ENDTIME AS TIMESTAMP) <= CAST(s.INTIME AS TIMESTAMP)
        GROUP BY s.HADM_ID
    """).df()

    hw_df = con.query("""
        SELECT 
            s.stay_id,
            max(CASE WHEN c.ITEMID IN (920, 226730) THEN c.VALUENUM END) as height,
            max(CASE WHEN c.ITEMID IN (762, 226512) THEN c.VALUENUM END) as weight
        FROM stays_df_tmp s
        JOIN read_csv_auto('/home/hanwen/data/mimic/iii/CHARTEVENTS.csv', sample_size=-1) c
            ON s.stay_id = c.ICUSTAY_ID
        WHERE c.ITEMID IN (920, 226730, 762, 226512) AND c.VALUENUM IS NOT NULL
        GROUP BY s.stay_id
    """).df()

    df = stays_df.copy()

    df = df.merge(transfers_df, on="HADM_ID", how="left")
    df["pre_icu_transfers"] = df["pre_icu_transfers"].fillna(0)

    df = df.merge(surg_df, on="HADM_ID", how="left")
    df["surg_count"] = df["surg_count"].fillna(0)
    df["surg_flag"] = (df["surg_count"] > 0).astype(int)

    df["last_surg_endtime"] = pd.to_datetime(df["last_surg_endtime"])
    df["INTIME"] = pd.to_datetime(df["INTIME"])
    gap_hours = (df["INTIME"] - df["last_surg_endtime"]).dt.total_seconds() / 3600.0
    df["surg_gap"] = np.where(gap_hours > 0, gap_hours, 0)
    df["log_surg_gap"] = np.log1p(df["surg_gap"])

    df["EDREGTIME"] = pd.to_datetime(df["EDREGTIME"])
    df["EDOUTTIME"] = pd.to_datetime(df["EDOUTTIME"])
    ed_wait = (df["EDOUTTIME"] - df["EDREGTIME"]).dt.total_seconds() / 3600.0
    df["log_ed_wait"] = np.log1p(np.where((ed_wait > 0) & (ed_wait < 240), ed_wait, 0))

    df = df.merge(hw_df, on="stay_id", how="left")
    df["height"] = df["height"].fillna(0)
    df["weight"] = df["weight"].fillna(0)

    con.unregister("stays_df_tmp")

    return df


def compute_icu_load_features(stays_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ICU stay-duration features using the pre-existing LOS column
    (Length of Stay in days) from ICUSTAYS.

    Note on MIMIC-III time-shifting:
      MIMIC-III shifts each patient's timestamps by an independent random offset,
      so comparing absolute timestamps *across* patients is meaningless.  We
      therefore do NOT compute any cross-patient concurrency metric (load_index).
      Instead, we derive two within-patient / population-mean features that are
      robust to the per-patient time shift:

      - log_icu_los     : log1p(LOS in hours) — captures stay duration on a
                          log scale, robust to outliers.
      - icu_speedup_los : unit_mean_LOS - actual_LOS — deviation from the
                          *same ICU unit*’s average LOS (grouped by FIRST_CAREUNIT).
                          Positive = shorter than peers in the same unit type,
                          suggesting potential capacity-driven early discharge.
                          Using unit-level mean avoids confounding from the large
                          baseline LOS differences across ICU types (e.g. MICU vs
                          CSRU vs TSICU).
    """
    logging.info(
        "Computing ICU LOS features (unit-grouped speedup via FIRST_CAREUNIT)..."
    )
    df = stays_df.copy()

    # LOS in ICUSTAYS is in days; convert to hours
    los_hours = df["LOS"].fillna(0.0).clip(lower=0.0) * 24.0
    df["_los_h"] = los_hours

    # Per-unit mean LOS (group by FIRST_CAREUNIT) — vectorised, no loop needed
    unit_mean_los = df.groupby("FIRST_CAREUNIT")["_los_h"].transform("mean")

    df["log_icu_los"] = np.log1p(los_hours).astype(np.float32)
    df["icu_speedup_los"] = (unit_mean_los - los_hours).astype(
        np.float32
    )  # positive = shorter than unit peers
    df.drop(columns=["_los_h"], inplace=True)

    # Log per-unit stats for sanity check
    unit_stats = stays_df.copy()
    unit_stats["_los_h"] = los_hours
    per_unit = unit_stats.groupby("FIRST_CAREUNIT")["_los_h"].agg(["mean", "count"])
    stats_str = ", ".join(
        f"{u}: {row['mean']:.1f}h(n={int(row['count'])})"
        for u, row in per_unit.iterrows()
    )
    logging.info(f"ICU LOS by unit — {stats_str}")
    logging.info(
        f"icu_speedup_los (unit-adjusted): range=[{df['icu_speedup_los'].min():.1f}, "
        f"{df['icu_speedup_los'].max():.1f}]h, mean={df['icu_speedup_los'].mean():.2f}h"
    )
    return df


def fetch_mimic3_data(embeddings_dict, note_emb_dim=768, task="readmission"):
    logging.info(
        f"Connecting to DuckDB and loading MIMIC-III features for task: {task}..."
    )
    con = duckdb.connect()

    stay_ids = list(embeddings_dict.keys())
    if not stay_ids:
        return None, None, None, None, None, None

    # 1. Stays and Demographics
    logging.info("Loading Demographics...")
    stays_df = con.query(f"""
        SELECT 
            s.SUBJECT_ID, s.HADM_ID, s.ICUSTAY_ID as stay_id, s.INTIME, s.OUTTIME, s.FIRST_CAREUNIT, s.LOS,
            a.ETHNICITY, a.MARITAL_STATUS, a.RELIGION, a.INSURANCE, a.ADMISSION_TYPE, a.ADMISSION_LOCATION,
            a.EDREGTIME, a.EDOUTTIME, a.HOSPITAL_EXPIRE_FLAG,
            p.GENDER, p.DOB
        FROM read_csv_auto('/home/hanwen/data/mimic/iii/ICUSTAYS.csv', sample_size=-1) s
        JOIN read_csv_auto('/home/hanwen/data/mimic/iii/ADMISSIONS.csv', sample_size=-1) a ON s.HADM_ID = a.HADM_ID
        JOIN read_csv_auto('/home/hanwen/data/mimic/iii/PATIENTS.csv', sample_size=-1) p ON s.SUBJECT_ID = p.SUBJECT_ID
        WHERE s.ICUSTAY_ID IN {tuple(stay_ids)}
    """).df()

    # Two conditions (OR logic) define a positive label:
    #   A. Cross-admission: The patient (same SUBJECT_ID) has another ICU admission
    #      within 30 days of the current discharge (different HADM_ID).
    #   B. Within-admission: The patient has another ICU admission during the same
    #      hospital stay (same HADM_ID) where INTIME > current OUTTIME.
    stays_df["INTIME"] = pd.to_datetime(stays_df["INTIME"])
    stays_df["OUTTIME"] = pd.to_datetime(stays_df["OUTTIME"])
    stays_df = stays_df.sort_values(["SUBJECT_ID", "INTIME"]).reset_index(drop=True)

    if task == "readmission":
        logging.info(
            "Computing ICU readmission labels (30-day cross-admission OR within-admission)..."
        )
        readmitted_flags = []
        source_a_count = 0
        source_b_count = 0
        for i, row in stays_df.iterrows():
            # Condition A: Cross-admission to ICU within 30 days
            cond_a = stays_df[
                (stays_df["SUBJECT_ID"] == row["SUBJECT_ID"])
                & (stays_df["HADM_ID"] != row["HADM_ID"])
                & (stays_df["INTIME"] > row["OUTTIME"])
                & (stays_df["INTIME"] <= row["OUTTIME"] + pd.Timedelta(days=30))
            ]
            # Condition B: Readmission to ICU within the same hospital stay (same HADM_ID)
            cond_b = stays_df[
                (stays_df["HADM_ID"] == row["HADM_ID"])
                & (stays_df["INTIME"] > row["OUTTIME"])
            ]
            flag = 1 if (len(cond_a) > 0 or len(cond_b) > 0) else 0
            readmitted_flags.append(flag)
            if flag:
                if len(cond_a) > 0:
                    source_a_count += 1
                if len(cond_b) > 0:
                    source_b_count += 1
        stays_df["label"] = readmitted_flags
        logging.info(
            f"ICU readmission rate: {stays_df['label'].mean():.1%} "
            f"({stays_df['label'].sum()} / {len(stays_df)} stays) | "
            f"Cross-admission(A): {source_a_count}, Within-admission(B): {source_b_count}"
        )
    elif task == "mortality":
        logging.info("Computing In-Hospital Mortality labels...")
        stays_df["label"] = (
            stays_df["HOSPITAL_EXPIRE_FLAG"].fillna(0).astype(int).tolist()
        )
        logging.info(
            f"In-Hospital Mortality rate: {stays_df['label'].mean():.1%} "
            f"({stays_df['label'].sum()} / {len(stays_df)} stays)"
        )
    else:
        raise ValueError(f"Unknown task: {task}")

    stays_df["DOB"] = pd.to_datetime(stays_df["DOB"], errors="coerce")
    stays_df["age"] = (stays_df["INTIME"] - stays_df["DOB"]).dt.days / 365.25
    stays_df["age"] = stays_df["age"].clip(0, 100)

    # Extract original features instead of fitting OLS
    stays_df = enrich_stays_with_features(stays_df, con)

    # Add ICU LOS pressure features (uses ICUSTAYS.LOS column, no cross-patient timestamp comparison)
    stays_df = compute_icu_load_features(stays_df)

    cont_cols = [
        "age",
        "pre_icu_transfers",
        "surg_count",
        "log_surg_gap",
        "log_ed_wait",
        "icu_speedup_los",
        "log_icu_los",
        "height",
        "weight",
    ]
    cat_cols = [
        "GENDER",
        "MARITAL_STATUS",
        "RELIGION",
        "ETHNICITY",
        "INSURANCE",
        "ADMISSION_TYPE",
        "ADMISSION_LOCATION",
        "FIRST_CAREUNIT",
        "surg_flag",
    ]

    static_encoders, static_dims = {}, {}
    for col in cont_cols:
        static_dims[col] = 0
    for col in cat_cols:
        stays_df[col] = stays_df[col].fillna("UNKNOWN").astype(str)
        le = LabelEncoder()
        stays_df[col] = le.fit_transform(stays_df[col])
        static_encoders[col] = le
        static_dims[col] = len(le.classes_)

    hadm_ids_tuple = tuple(stays_df["HADM_ID"].unique().tolist())

    # 2. Extract High-Dim Sparse Features
    logging.info("Loading ICD Diagnoses (Top 64)...")
    icd_dict, icd_dim = build_multihot_features(
        con, "DIAGNOSES_ICD", "HADM_ID", "ICD9_CODE", hadm_ids_tuple, top_k=64, trim=3
    )

    logging.info("Loading DRG Codes (Top 64)...")
    drg_dict, drg_dim = build_multihot_features(
        con, "DRGCODES", "HADM_ID", "DRG_CODE", hadm_ids_tuple, top_k=64
    )

    logging.info("Loading Procedures ICD (Top 64)...")
    proc_dict, proc_dim = build_multihot_features(
        con, "PROCEDURES_ICD", "HADM_ID", "ICD9_CODE", hadm_ids_tuple, top_k=64, trim=3
    )

    logging.info("Loading Pharmacy / Prescriptions (Top 64)...")
    rx_dict, rx_dim = build_multihot_features(
        con, "PRESCRIPTIONS", "HADM_ID", "DRUG", hadm_ids_tuple, top_k=64
    )

    multihot_dims = {"icd": icd_dim, "drg": drg_dim, "proc": proc_dim, "rx": rx_dim}

    # 3. Dynamic Sequence Features
    item_map = {
        211: 0,
        220045: 0,
        618: 1,
        220210: 1,
        646: 2,
        220277: 2,
        51: 3,
        220050: 3,
        8368: 4,
        220051: 4,  # Diastolic BP
        52: 5,
        220052: 5,
        225312: 5,  # MAP
        678: 6,
        223761: 6,
        676: 6,
        223762: 6,  # Temperature
        807: 7,
        811: 7,
        1529: 7,
        225664: 7,
        220621: 7,  # Glucose
        184: 8,
        220739: 8,  # GCS Eye Opening
        454: 9,
        223901: 9,  # GCS Motor Response
        723: 10,
        223900: 10,  # GCS Verbal Response
        198: 11,  # GCS Total
        780: 12,
        1126: 12,
        220274: 12,  # pH
        3420: 13,
        190: 13,
        223835: 13,  # FiO2
        3348: 14,
        115: 14,
        224308: 14,
        8377: 14,  # Capillary Refill Rate
    }
    logging.info(f"Querying CHARTEVENTS for sequences...")
    events_df = con.query(f"""
        SELECT ICUSTAY_ID as stay_id, CHARTTIME, ITEMID, VALUENUM
        FROM read_csv_auto('/home/hanwen/data/mimic/iii/CHARTEVENTS.csv', sample_size=-1)
        WHERE ICUSTAY_ID IN {tuple(stay_ids)} AND ITEMID IN {tuple(item_map.keys())} AND VALUENUM IS NOT NULL
    """).df()

    logging.info("Formatting dataset...")
    X_seq, X_static, X_mh, X_note, Y = [], [], [], [], []
    dropped_count = 0

    for _, stay in stays_df.iterrows():
        sid = stay["stay_id"]
        hadm = stay["HADM_ID"]

        # Sequence
        evs = events_df[events_df["stay_id"] == sid].copy()
        if evs.empty:
            dropped_count += 1
            continue

        evs["CHARTTIME"] = pd.to_datetime(evs["CHARTTIME"])
        evs["hour"] = (
            (evs["CHARTTIME"] - stay["INTIME"]).dt.total_seconds() / 3600
        ).astype(int)
        evs = evs[(evs["hour"] >= 0) & (evs["hour"] < 48)]

        observed_channels = evs["ITEMID"].map(item_map).nunique()
        if observed_channels < 8:
            dropped_count += 1
            continue

        # Bug3 fix: initialize with NaN so that un-observed slots are truly missing,
        # and real zero-valued measurements are NOT incorrectly treated as absent.
        seq = np.full((48, 15), np.nan, dtype=np.float32)
        for _, e in evs.iterrows():
            seq[int(e["hour"]), item_map[e["ITEMID"]]] = e["VALUENUM"]

        # ffill: carry last observed value forward; fill remaining leading NaNs with 0
        df_seq = pd.DataFrame(seq).ffill().fillna(0.0)
        X_seq.append(df_seq.values)

        # Static
        X_static.append([stay[col] for col in cont_cols + cat_cols])

        # Multihot
        mh_vecs = []
        mh_vecs.append(icd_dict.get(hadm, np.zeros(icd_dim, dtype=np.float32)))
        mh_vecs.append(drg_dict.get(hadm, np.zeros(drg_dim, dtype=np.float32)))
        mh_vecs.append(proc_dict.get(hadm, np.zeros(proc_dim, dtype=np.float32)))
        mh_vecs.append(rx_dict.get(hadm, np.zeros(rx_dim, dtype=np.float32)))
        X_mh.append(np.concatenate(mh_vecs))

        val = embeddings_dict.get(sid)
        if val is None:
            val = np.zeros(note_emb_dim, dtype=np.float32)
        elif isinstance(val, dict):
            val = val.get("embedding", np.zeros(note_emb_dim, dtype=np.float32))
        X_note.append(np.array(val, dtype=np.float32))

        Y.append(stay["label"])

    logging.info(
        f"Dropped {dropped_count} stays due to extreme missingness (< 3 observed signals)"
    )
    return (
        np.array(X_seq),
        np.array(X_static, dtype=np.float32),
        np.array(X_mh, dtype=np.float32),
        np.array(X_note),
        np.array(Y),
        static_dims,
        multihot_dims,
    )


def flatten_features(X_seq, X_static, X_mh):
    return np.concatenate(
        [np.mean(X_seq, axis=1), X_seq[:, -1, :], X_static, X_mh], axis=1
    )
