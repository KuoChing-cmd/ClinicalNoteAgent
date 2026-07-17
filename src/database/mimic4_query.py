"""MIMIC4 data extraction queries for asymmetric information game theory research."""

import logging
import re
import time as pytime
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import and_, bindparam, func, or_, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from .mimic4_models import (
    Admission,
    DiagnosisICD,
    DICDDiagnosis,
    DICDProcedure,
    EligiblePatient,
    ICUStay,
    Patient,
    Prescription,
    ProcedureICD,
    Transfer,
)

DEFAULT_CHARTEVENT_LABEL_WHITELIST_TOP100: tuple[str, ...] = (
    "Heart Rate",
    "Respiratory Rate",
    "O2 saturation pulseoxymetry",
    "Non Invasive Blood Pressure systolic",
    "Non Invasive Blood Pressure diastolic",
    "Non Invasive Blood Pressure mean",
    "Orientation",
    "GCS - Eye Opening",
    "GCS - Verbal Response",
    "GCS - Motor Response",
    "Alarms On",
    "Temperature Fahrenheit",
    "Arterial Blood Pressure mean",
    "Arterial Blood Pressure diastolic",
    "Arterial Blood Pressure systolic",
    "Parameters Checked",
    "Activity / Mobility (JH-HLM)",
    "Richmond-RAS Scale",
    "Strength R Arm",
    "Strength L Arm",
    "Strength R Leg",
    "Strength L Leg",
    "ST Segment Monitoring On",
    "Goal Richmond-RAS Scale",
    "Pain Level",
    "Braden Activity",
    "Braden Mobility",
    "Braden Moisture",
    "Braden Nutrition",
    "Braden Sensory Perception",
    "Braden Friction/Shear",
    "Glucose finger stick (range 70-100)",
    "O2 Flow",
    "Central Venous Pressure",
    "Heart Rate Alarm - Low",
    "Heart rate Alarm - High",
    "O2 Saturation Pulseoxymetry Alarm - Low",
    "Resp Alarm - High",
    "Resp Alarm - Low",
    "O2 Saturation Pulseoxymetry Alarm - High",
    "History of falling (within 3 mnths)",
    "Secondary diagnosis",
    "Gait/Transferring",
    "Mental status",
    "IV/Saline lock",
    "Ambulatory aid",
    "20 Gauge Dressing Occlusive",
    "20 Gauge placed in outside facility",
    "SpO2 Desat Limit",
    "Non-Invasive Blood Pressure Alarm - Low",
    "Non-Invasive Blood Pressure Alarm - High",
    "20 Gauge placed in the field",
    "Pain Level Response",
    "18 Gauge Dressing Occlusive",
    "18 Gauge placed in outside facility",
    "Hematocrit (serum)",
    "Potassium (serum)",
    "Sodium (serum)",
    "Chloride (serum)",
    "Cough/Deep Breath",
    "Hemoglobin",
    "Creatinine (serum)",
    "CAM-ICU MS Change",
    "BUN",
    "HCO3 (serum)",
    "Anion gap",
    "Glucose (serum)",
    "Magnesium",
    "Platelet Count",
    "WBC",
    "18 Gauge placed in the field",
    "Inspired O2 Fraction",
    "Orientation to Person",
    "Orientation to Place",
    "Orientation to Time",
    "Phosphorous",
    "Calcium non-ionized",
    "Riker-SAS Scale",
    "Current Dyspnea Assessment",
    "Acuity Workload Question 1",
    "Acuity Workload Question 2",
    "Multi Lumen placed in outside facility",
    "Arterial Line placed in outside facility",
    "Incentive Spirometry",
    "PTT",
    "Prothrombin time",
    "INR",
    "Pulmonary Artery Pressure diastolic",
    "Pulmonary Artery Pressure systolic",
    "Pulmonary Artery Pressure mean",
    "Temperature Celsius",
    "High risk (>51) interventions",
    "Back Care",
    "PEEP set",
    "Skin Care",
    "Arterial Line Dressing Occlusive",
    "Minute Volume",
    "Tidal Volume (observed)",
    "Mean Airway Pressure",
    "Peak Insp. Pressure",
)

logger = logging.getLogger(__name__)


class MIMIC4DataExtractor:
    """Extract patient data from MIMIC4 database for game theory research."""

    def __init__(self, session: Session):
        """Initialize extractor with database session.

        Args:
            session: SQLAlchemy database session
        """
        self.session = session

    @staticmethod
    def _try_parse_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        s = str(value).strip()
        if not s:
            return None
        m = re.search(r"[-+]?\d*\.?\d+", s)
        if not m:
            return None
        try:
            return float(m.group(0))
        except Exception:
            return None

    @staticmethod
    def _parse_bp_pair(value: Any) -> tuple[Optional[float], Optional[float]]:
        if value is None:
            return None, None
        s = str(value).strip()
        if "/" not in s:
            return None, None
        left, right = s.split("/", 1)
        return MIMIC4DataExtractor._try_parse_float(
            left
        ), MIMIC4DataExtractor._try_parse_float(right)

    @staticmethod
    def _coerce_charttime(value: Any) -> datetime | None:
        if isinstance(value, datetime):
            return value
        if isinstance(value, date):
            return datetime.combine(value, time.min)
        if isinstance(value, str) and value.strip():
            try:
                parsed = datetime.fromisoformat(value.strip())
                return parsed
            except Exception:
                return None
        return None

    @staticmethod
    def _is_retryable_mysql_disconnect(exc: Exception) -> bool:
        message = str(getattr(exc, "orig", exc)).lower()
        return (
            "(2013" in message
            or "(2006" in message
            or "(2003" in message
            or "lost connection to mysql server during query" in message
            or "mysql server has gone away" in message
            or "can't connect to mysql server" in message
            or "connection refused" in message
        )

    def _reset_session_after_disconnect(self) -> None:
        try:
            self.session.rollback()
        except Exception:
            pass
        try:
            self.session.close()
        except Exception:
            pass

    def _execute_mappings_with_retry(
        self,
        statement: Any,
        params: Dict[str, Any],
        *,
        stream_results: bool = False,
        max_retries: int = 8,
    ) -> List[Dict[str, Any]]:
        attempt = 0
        while True:
            try:
                stmt = (
                    statement.execution_options(stream_results=True)
                    if stream_results
                    else statement
                )
                return [
                    dict(r) for r in self.session.execute(stmt, params).mappings().all()
                ]
            except OperationalError as exc:
                attempt += 1
                self._reset_session_after_disconnect()
                if attempt >= int(
                    max_retries
                ) or not self._is_retryable_mysql_disconnect(exc):
                    raise
                wait_s = min(10.0, 0.8 * float(attempt))
                logger.warning(
                    "Retrying SQL execution after MySQL disconnect (attempt=%s/%s, wait=%.1fs)",
                    attempt,
                    max_retries,
                    wait_s,
                )
                pytime.sleep(wait_s)

    def _fetch_labevents_chunked(
        self,
        *,
        hadm_id: int,
        stay_id: int,
        intime: datetime,
        outtime: datetime,
        max_rows: int,
        lab_itemids: Optional[Sequence[int]] = None,
        batch_size: int = 5000,
        max_retries: int = 8,
    ) -> List[Dict[str, Any]]:
        """Fetch labevents in keyset-pagination chunks to reduce disconnect risk."""
        rows: List[Dict[str, Any]] = []
        cursor_charttime: Optional[datetime] = None
        cursor_event_id: Optional[int] = None
        safe_batch_size = max(200, min(int(batch_size), int(max_rows)))

        while len(rows) < int(max_rows):
            remaining = int(max_rows) - len(rows)
            current_limit = min(safe_batch_size, remaining)

            if lab_itemids:
                chunk_sql = text("""
                    SELECT
                        :stay_id AS stay_id,
                        le.charttime,
                        le.itemid,
                        le.labevent_id,
                        COALESCE(dl.label, CONCAT('LAB:', CAST(le.itemid AS CHAR))) AS feature,
                        le.valuenum AS value,
                        le.valueuom
                    FROM labevents le
                    LEFT JOIN d_labitems dl ON dl.itemid = le.itemid
                    WHERE le.hadm_id = :hadm_id
                      AND le.charttime >= :intime
                      AND le.charttime <= :outtime
                      AND le.valuenum IS NOT NULL
                      AND le.itemid IN :itemids
                      AND (
                        :cursor_charttime IS NULL
                        OR le.charttime > :cursor_charttime
                        OR (le.charttime = :cursor_charttime AND le.labevent_id > :cursor_event_id)
                      )
                    ORDER BY le.charttime ASC, le.labevent_id ASC
                    LIMIT :chunk_limit
                    """).bindparams(bindparam("itemids", expanding=True))
                chunk_params: Dict[str, Any] = {
                    "stay_id": int(stay_id),
                    "hadm_id": int(hadm_id),
                    "intime": intime,
                    "outtime": outtime,
                    "itemids": [int(x) for x in lab_itemids],
                    "cursor_charttime": cursor_charttime,
                    "cursor_event_id": cursor_event_id,
                    "chunk_limit": int(current_limit),
                }
            else:
                chunk_sql = text("""
                    SELECT
                        :stay_id AS stay_id,
                        le.charttime,
                        le.itemid,
                        le.labevent_id,
                        COALESCE(dl.label, CONCAT('LAB:', CAST(le.itemid AS CHAR))) AS feature,
                        le.valuenum AS value,
                        le.valueuom
                    FROM labevents le
                    LEFT JOIN d_labitems dl ON dl.itemid = le.itemid
                    WHERE le.hadm_id = :hadm_id
                      AND le.charttime >= :intime
                      AND le.charttime <= :outtime
                      AND le.valuenum IS NOT NULL
                      AND (
                        :cursor_charttime IS NULL
                        OR le.charttime > :cursor_charttime
                        OR (le.charttime = :cursor_charttime AND le.labevent_id > :cursor_event_id)
                      )
                    ORDER BY le.charttime ASC, le.labevent_id ASC
                    LIMIT :chunk_limit
                    """)
                chunk_params = {
                    "stay_id": int(stay_id),
                    "hadm_id": int(hadm_id),
                    "intime": intime,
                    "outtime": outtime,
                    "cursor_charttime": cursor_charttime,
                    "cursor_event_id": cursor_event_id,
                    "chunk_limit": int(current_limit),
                }

            attempt = 0
            chunk_rows: List[Dict[str, Any]] = []
            while True:
                try:
                    chunk_rows = [
                        dict(r)
                        for r in (
                            self.session.execute(
                                chunk_sql.execution_options(stream_results=True),
                                chunk_params,
                            )
                            .mappings()
                            .all()
                        )
                    ]
                    break
                except OperationalError as exc:
                    attempt += 1
                    self._reset_session_after_disconnect()
                    if attempt >= int(
                        max_retries
                    ) or not self._is_retryable_mysql_disconnect(exc):
                        raise
                    safe_batch_size = max(200, safe_batch_size // 2)
                    pytime.sleep(min(8.0, 0.8 * float(attempt)))
                    logger.warning(
                        "Retrying labevents chunk after MySQL disconnect (attempt=%s/%s, batch_size=%s)",
                        attempt,
                        max_retries,
                        safe_batch_size,
                    )

            if not chunk_rows:
                break

            rows.extend(chunk_rows)
            last_row = chunk_rows[-1]
            cursor_charttime = last_row.get("charttime")
            cursor_event_id = int(last_row.get("labevent_id") or 0)

            if len(chunk_rows) < current_limit:
                break

        return rows

    def _fetch_inputevents_chunked(
        self,
        *,
        hadm_id: int,
        stay_id: int,
        intime: datetime,
        outtime: datetime,
        max_rows: int,
        batch_size: int = 5000,
        max_retries: int = 8,
    ) -> List[Dict[str, Any]]:
        """Fetch inputevents in chunks to avoid single large query disconnects."""
        rows: List[Dict[str, Any]] = []
        cursor_charttime: Optional[datetime] = None
        cursor_orderid: int = -1
        cursor_itemid: int = -1
        safe_batch_size = max(200, min(int(batch_size), int(max_rows)))

        while len(rows) < int(max_rows):
            remaining = int(max_rows) - len(rows)
            current_limit = min(safe_batch_size, remaining)

            chunk_sql = text("""
                SELECT
                    ie.stay_id,
                    COALESCE(ie.starttime, ie.endtime) AS charttime,
                    ie.itemid,
                    COALESCE(ie.orderid, 0) AS orderid,
                    di.label AS feature,
                    CASE
                        WHEN ie.amount IS NOT NULL THEN ie.amount
                        WHEN ie.rate IS NOT NULL THEN ie.rate
                        ELSE NULL
                    END AS value,
                    CASE
                        WHEN ie.amount IS NOT NULL THEN ie.amountuom
                        ELSE ie.rateuom
                    END AS valueuom
                FROM inputevents ie
                INNER JOIN d_items di ON di.itemid = ie.itemid
                WHERE ie.hadm_id = :hadm_id
                  AND ie.stay_id = :stay_id
                  AND COALESCE(ie.starttime, ie.endtime) >= :intime
                  AND COALESCE(ie.starttime, ie.endtime) <= :outtime
                  AND (ie.amount IS NOT NULL OR ie.rate IS NOT NULL)
                  AND (
                    :cursor_charttime IS NULL
                    OR COALESCE(ie.starttime, ie.endtime) > :cursor_charttime
                    OR (
                        COALESCE(ie.starttime, ie.endtime) = :cursor_charttime
                        AND COALESCE(ie.orderid, 0) > :cursor_orderid
                    )
                    OR (
                        COALESCE(ie.starttime, ie.endtime) = :cursor_charttime
                        AND COALESCE(ie.orderid, 0) = :cursor_orderid
                        AND ie.itemid > :cursor_itemid
                    )
                  )
                ORDER BY
                    COALESCE(ie.starttime, ie.endtime) ASC,
                    COALESCE(ie.orderid, 0) ASC,
                    ie.itemid ASC
                LIMIT :chunk_limit
                """)
            chunk_params: Dict[str, Any] = {
                "hadm_id": int(hadm_id),
                "stay_id": int(stay_id),
                "intime": intime,
                "outtime": outtime,
                "cursor_charttime": cursor_charttime,
                "cursor_orderid": int(cursor_orderid),
                "cursor_itemid": int(cursor_itemid),
                "chunk_limit": int(current_limit),
            }

            attempt = 0
            chunk_rows: List[Dict[str, Any]] = []
            while True:
                try:
                    chunk_rows = [
                        dict(r)
                        for r in (
                            self.session.execute(
                                chunk_sql.execution_options(stream_results=True),
                                chunk_params,
                            )
                            .mappings()
                            .all()
                        )
                    ]
                    break
                except OperationalError as exc:
                    attempt += 1
                    self._reset_session_after_disconnect()
                    if attempt >= int(
                        max_retries
                    ) or not self._is_retryable_mysql_disconnect(exc):
                        raise
                    safe_batch_size = max(200, safe_batch_size // 2)
                    pytime.sleep(min(8.0, 0.8 * float(attempt)))
                    logger.warning(
                        "Retrying inputevents chunk after MySQL disconnect (attempt=%s/%s, batch_size=%s)",
                        attempt,
                        max_retries,
                        safe_batch_size,
                    )

            if not chunk_rows:
                break

            rows.extend(chunk_rows)
            last_row = chunk_rows[-1]
            cursor_charttime = last_row.get("charttime")
            cursor_orderid = int(last_row.get("orderid") or 0)
            cursor_itemid = int(last_row.get("itemid") or 0)

            if len(chunk_rows) < current_limit:
                break

        return rows

    def extract_eligible_patients(
        self,
        min_los_days: int = 3,
        min_transfers: int = 2,
        min_diagnoses: int = 1,
        min_age: int = 18,
        max_age: int = 89,
        limit: int = 1000,
    ) -> int:
        """Create temporary table of eligible patients for game theory research.

        筛选标准:
        1. 有完整的住院记录
        2. 有转科记录（体现信息传递和决策复杂性）
        3. 住院时长 >= min_los_days（有足够的医疗决策过程）
        4. 有诊断记录（体现医疗信息）
        5. 18-89岁成年患者

        Args:
            min_los_days: 最短住院天数
            min_transfers: 最少转科次数
            min_diagnoses: 最少诊断数量
            min_age: 最小年龄
            max_age: 最大年龄
            limit: 样本数量限制

        Returns:
            创建的符合条件患者数量
        """
        logger.info("Creating eligible patients table...")

        # Clear existing data from eligible_patients table BEFORE querying
        # Use raw SQL DELETE for maximum reliability
        try:
            result = self.session.execute(text("DELETE FROM eligible_patients"))
            self.session.commit()
            deleted_count = result.rowcount if hasattr(result, "rowcount") else 0
            if deleted_count > 0:
                logger.info(
                    f"Cleared {deleted_count} existing records from eligible_patients"
                )
            else:
                logger.info("eligible_patients table is empty, starting fresh")
        except Exception as e:
            logger.warning(
                f"Could not clear existing data (table may not exist yet): {e}"
            )
            self.session.rollback()

        # Build the query using SQLAlchemy
        # Subquery for transfer counts
        transfer_subq = (
            self.session.query(
                Transfer.hadm_id,
                func.count(func.distinct(Transfer.transfer_id)).label("transfer_count"),
            )
            .group_by(Transfer.hadm_id)
            .having(func.count(func.distinct(Transfer.transfer_id)) >= min_transfers)
            .subquery()
        )

        # Subquery for diagnosis counts
        diagnosis_subq = (
            self.session.query(
                DiagnosisICD.hadm_id,
                func.count(func.distinct(DiagnosisICD.icd_code)).label(
                    "diagnosis_count"
                ),
            )
            .group_by(DiagnosisICD.hadm_id)
            .having(func.count(func.distinct(DiagnosisICD.icd_code)) >= min_diagnoses)
            .subquery()
        )

        # Main query
        query = (
            self.session.query(
                Patient.subject_id,
                Admission.hadm_id,
                Patient.gender,
                (
                    func.year(Admission.admittime)
                    - Patient.anchor_year
                    + Patient.anchor_age
                ).label("age_at_admission"),
                Admission.admittime,
                Admission.dischtime,
                func.datediff(Admission.dischtime, Admission.admittime).label(
                    "los_days"
                ),
                Admission.admission_type,
                Admission.hospital_expire_flag,
                transfer_subq.c.transfer_count,
                diagnosis_subq.c.diagnosis_count,
            )
            .join(Admission, Patient.subject_id == Admission.subject_id)
            .join(transfer_subq, Admission.hadm_id == transfer_subq.c.hadm_id)
            .join(diagnosis_subq, Admission.hadm_id == diagnosis_subq.c.hadm_id)
            .filter(
                and_(
                    func.datediff(Admission.dischtime, Admission.admittime)
                    >= min_los_days,
                    (
                        func.year(Admission.admittime)
                        - Patient.anchor_year
                        + Patient.anchor_age
                    ).between(min_age, max_age),
                    Admission.dischtime.isnot(None),
                )
            )
            .limit(limit)
        )

        # Insert into eligible_patients table
        eligible_data = query.all()
        count = 0

        # Batch insert for better performance
        eligible_patients = []
        for row in eligible_data:
            eligible_patient = EligiblePatient(
                subject_id=row.subject_id,
                hadm_id=row.hadm_id,
                gender=row.gender,
                age_at_admission=row.age_at_admission,
                admittime=row.admittime,
                dischtime=row.dischtime,
                los_days=row.los_days,
                admission_type=row.admission_type,
                hospital_expire_flag=row.hospital_expire_flag,
                transfer_count=row.transfer_count,
                diagnosis_count=row.diagnosis_count,
            )
            eligible_patients.append(eligible_patient)
            count += 1

        # Bulk insert all records at once
        if eligible_patients:
            try:
                self.session.bulk_save_objects(eligible_patients)
                self.session.commit()
                logger.info(f"Created {count} eligible patients")
            except Exception as e:
                logger.error(f"Failed to insert eligible patients: {e}")
                self.session.rollback()
                raise
        else:
            logger.warning("No eligible patients found matching the criteria")

        return count

    def get_eligible_patients_summary(self) -> Dict[str, Any]:
        """查看筛选出的患者样本概况.

        Returns:
            统计摘要字典
        """
        summary = self.session.query(
            func.count(func.distinct(EligiblePatient.subject_id)).label(
                "total_patients"
            ),
            func.count(func.distinct(EligiblePatient.hadm_id)).label(
                "total_admissions"
            ),
            func.avg(EligiblePatient.age_at_admission).label("avg_age"),
            func.avg(EligiblePatient.los_days).label("avg_los"),
            func.avg(EligiblePatient.transfer_count).label("avg_transfers"),
            func.avg(EligiblePatient.diagnosis_count).label("avg_diagnoses"),
            func.sum(EligiblePatient.hospital_expire_flag).label("deaths"),
        ).first()

        return {
            "total_patients": summary.total_patients or 0,
            "total_admissions": summary.total_admissions or 0,
            "avg_age": round(float(summary.avg_age or 0), 2),
            "avg_los": round(float(summary.avg_los or 0), 2),
            "avg_transfers": round(float(summary.avg_transfers or 0), 2),
            "avg_diagnoses": round(float(summary.avg_diagnoses or 0), 2),
            "deaths": summary.deaths or 0,
        }

    def get_eligible_patients_sample(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取符合条件的患者样本.

        Args:
            limit: 返回记录数

        Returns:
            患者样本列表
        """
        patients = self.session.query(EligiblePatient).limit(limit).all()

        result = []
        for p in patients:
            result.append(
                {
                    "subject_id": p.subject_id,
                    "hadm_id": p.hadm_id,
                    "gender": p.gender,
                    "age_at_admission": p.age_at_admission,
                    "admittime": p.admittime,
                    "dischtime": p.dischtime,
                    "los_days": p.los_days,
                    "admission_type": p.admission_type,
                    "hospital_expire_flag": p.hospital_expire_flag,
                    "transfer_count": p.transfer_count,
                    "diagnosis_count": p.diagnosis_count,
                }
            )

        return result

    def get_patient_basic_info(
        self, hadm_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """查询1: 获取筛选患者的基本信息.

        Args:
            hadm_id: 可选，指定住院ID

        Returns:
            患者基本信息列表
        """
        query = (
            self.session.query(
                EligiblePatient.subject_id,
                EligiblePatient.hadm_id,
                EligiblePatient.gender,
                EligiblePatient.age_at_admission,
                Patient.anchor_year,
                Patient.dod.label("date_of_death"),
                EligiblePatient.admittime,
                EligiblePatient.dischtime,
                EligiblePatient.los_days,
                EligiblePatient.admission_type,
                EligiblePatient.hospital_expire_flag,
                EligiblePatient.transfer_count,
                EligiblePatient.diagnosis_count,
            )
            .join(Patient, EligiblePatient.subject_id == Patient.subject_id)
            .order_by(EligiblePatient.admittime.desc())
        )

        if hadm_id:
            query = query.filter(EligiblePatient.hadm_id == hadm_id)

        results = query.all()

        return [
            {
                "subject_id": r.subject_id,
                "hadm_id": r.hadm_id,
                "gender": r.gender,
                "age_at_admission": r.age_at_admission,
                "anchor_year": r.anchor_year,
                "date_of_death": r.date_of_death,
                "admittime": r.admittime,
                "dischtime": r.dischtime,
                "los_days": r.los_days,
                "admission_type": r.admission_type,
                "hospital_expire_flag": r.hospital_expire_flag,
                "transfer_count": r.transfer_count,
                "diagnosis_count": r.diagnosis_count,
            }
            for r in results
        ]

    def get_admission_details(
        self, hadm_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """查询2: 获取筛选患者的住院详情.

        Args:
            hadm_id: 可选，指定住院ID

        Returns:
            住院详情列表
        """
        query = self.session.query(
            Admission.hadm_id,
            Admission.subject_id,
            Admission.admittime.label("入院时间"),
            Admission.dischtime.label("出院时间"),
            Admission.deathtime.label("死亡时间"),
            func.datediff(Admission.dischtime, Admission.admittime).label("住院天数"),
            Admission.admission_type.label("入院类型"),
            Admission.admission_location.label("入院来源"),
            Admission.discharge_location.label("出院去向"),
            Admission.insurance,
            Admission.language,
            Admission.marital_status,
            Admission.race,
            Admission.hospital_expire_flag,
        ).join(EligiblePatient, Admission.hadm_id == EligiblePatient.hadm_id)

        if hadm_id:
            query = query.filter(Admission.hadm_id == hadm_id)

        results = query.all()

        return [
            {
                "hadm_id": r.hadm_id,
                "subject_id": r.subject_id,
                "入院时间": r.入院时间,
                "出院时间": r.出院时间,
                "死亡时间": r.死亡时间,
                "住院天数": r.住院天数,
                "入院类型": r.入院类型,
                "入院来源": r.入院来源,
                "出院去向": r.出院去向,
                "insurance": r.insurance,
                "language": r.language,
                "marital_status": r.marital_status,
                "race": r.race,
                "hospital_expire_flag": r.hospital_expire_flag,
            }
            for r in results
        ]

    def get_icu_stays(self, hadm_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """查询3: 获取筛选患者的ICU入住记录.

        Args:
            hadm_id: 可选，指定住院ID

        Returns:
            ICU入住记录列表
        """
        query = (
            self.session.query(
                ICUStay.subject_id,
                ICUStay.hadm_id,
                ICUStay.stay_id,
                ICUStay.intime.label("icu_入院时间"),
                ICUStay.outtime.label("icu_出院时间"),
                func.timestampdiff(text("HOUR"), ICUStay.intime, ICUStay.outtime).label(
                    "icu_停留小时数"
                ),
                ICUStay.los.label("重症监护_长度"),
                ICUStay.first_careunit.label("首次入住科室"),
                ICUStay.last_careunit.label("最后逗留科室"),
            )
            .join(EligiblePatient, ICUStay.hadm_id == EligiblePatient.hadm_id)
            .order_by(ICUStay.subject_id, ICUStay.intime)
        )

        if hadm_id:
            query = query.filter(ICUStay.hadm_id == hadm_id)

        results = query.all()

        return [
            {
                "subject_id": r.subject_id,
                "hadm_id": r.hadm_id,
                "stay_id": r.stay_id,
                "icu_入院时间": r.icu_入院时间,
                "icu_出院时间": r.icu_出院时间,
                "icu_停留小时数": r.icu_停留小时数,
                "重症监护_长度": r.重症监护_长度,
                "首次入住科室": r.首次入住科室,
                "最后逗留科室": r.最后逗留科室,
            }
            for r in results
        ]

    def get_transfers(self, hadm_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """查询3B: 获取筛选患者的转科记录（包含无ICU记录的患者）.

        Args:
            hadm_id: 可选，指定住院ID

        Returns:
            转科记录列表
        """
        query = (
            self.session.query(
                Transfer.subject_id,
                Transfer.hadm_id,
                Transfer.transfer_id,
                Transfer.eventtype.label("事件类型"),
                Transfer.careunit.label("护理单元"),
                Transfer.intime.label("入科时间"),
                Transfer.outtime.label("出科时间"),
                func.timestampdiff(
                    text("HOUR"), Transfer.intime, Transfer.outtime
                ).label("停留小时数"),
            )
            .join(EligiblePatient, Transfer.hadm_id == EligiblePatient.hadm_id)
            .order_by(Transfer.subject_id, Transfer.intime)
        )

        if hadm_id:
            query = query.filter(Transfer.hadm_id == hadm_id)

        results = query.all()

        return [
            {
                "subject_id": r.subject_id,
                "hadm_id": r.hadm_id,
                "transfer_id": r.transfer_id,
                "事件类型": r.事件类型,
                "护理单元": r.护理单元,
                "入科时间": r.入科时间,
                "出科时间": r.出科时间,
                "停留小时数": r.停留小时数,
            }
            for r in results
        ]

    def get_diagnoses(self, hadm_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """查询4: 获取筛选患者的诊断记录.

        Args:
            hadm_id: 可选，指定住院ID

        Returns:
            诊断记录列表
        """
        query = (
            self.session.query(
                DiagnosisICD.hadm_id,
                DiagnosisICD.subject_id,
                DiagnosisICD.icd_code,
                DiagnosisICD.icd_version,
                DiagnosisICD.seq_num.label("诊断序号"),
                DICDDiagnosis.long_title.label("诊断描述"),
            )
            .join(EligiblePatient, DiagnosisICD.hadm_id == EligiblePatient.hadm_id)
            .outerjoin(
                DICDDiagnosis,
                and_(
                    DiagnosisICD.icd_code == DICDDiagnosis.icd_code,
                    DiagnosisICD.icd_version == DICDDiagnosis.icd_version,
                ),
            )
            .order_by(
                DiagnosisICD.subject_id, DiagnosisICD.hadm_id, DiagnosisICD.seq_num
            )
        )

        if hadm_id:
            query = query.filter(DiagnosisICD.hadm_id == hadm_id)

        results = query.all()

        return [
            {
                "hadm_id": r.hadm_id,
                "subject_id": r.subject_id,
                "icd_code": r.icd_code,
                "icd_version": r.icd_version,
                "诊断序号": r.诊断序号,
                "诊断描述": r.诊断描述,
            }
            for r in results
        ]

    def get_procedures(self, hadm_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """查询5: 获取筛选患者的处置/治疗记录.

        Args:
            hadm_id: 可选，指定住院ID

        Returns:
            处置记录列表
        """
        query = (
            self.session.query(
                ProcedureICD.hadm_id,
                ProcedureICD.subject_id,
                ProcedureICD.icd_code,
                ProcedureICD.icd_version,
                ProcedureICD.seq_num.label("处置序号"),
                DICDProcedure.long_title.label("处置描述"),
            )
            .join(EligiblePatient, ProcedureICD.hadm_id == EligiblePatient.hadm_id)
            .outerjoin(
                DICDProcedure,
                and_(
                    ProcedureICD.icd_code == DICDProcedure.icd_code,
                    ProcedureICD.icd_version == DICDProcedure.icd_version,
                ),
            )
            .order_by(
                ProcedureICD.subject_id, ProcedureICD.hadm_id, ProcedureICD.seq_num
            )
        )

        if hadm_id:
            query = query.filter(ProcedureICD.hadm_id == hadm_id)

        results = query.all()

        return [
            {
                "hadm_id": r.hadm_id,
                "subject_id": r.subject_id,
                "icd_code": r.icd_code,
                "icd_version": r.icd_version,
                "处置序号": r.处置序号,
                "处置描述": r.处置描述,
            }
            for r in results
        ]

    def get_prescriptions(
        self, hadm_id: Optional[int] = None, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """查询6: 获取筛选患者的药物记录（采样示例）.

        由于数据量可能很大，建议使用limit限制返回数量

        Args:
            hadm_id: 可选，指定住院ID
            limit: 可选，限制返回数量

        Returns:
            药物记录列表
        """
        query = (
            self.session.query(
                Prescription.hadm_id,
                Prescription.subject_id,
                Prescription.starttime.label("用药开始时间"),
                Prescription.stoptime.label("用药停止时间"),
                Prescription.drug.label("药物名称"),
                Prescription.dose_val_rx.label("给定剂量"),
                Prescription.dose_unit_rx.label("剂量单位"),
                Prescription.route.label("给药路线"),
            )
            .join(EligiblePatient, Prescription.hadm_id == EligiblePatient.hadm_id)
            .order_by(Prescription.subject_id, Prescription.starttime)
        )

        if hadm_id:
            query = query.filter(Prescription.hadm_id == hadm_id)

        if limit:
            query = query.limit(limit)

        results = query.all()

        return [
            {
                "hadm_id": r.hadm_id,
                "subject_id": r.subject_id,
                "用药开始时间": r.用药开始时间,
                "用药停止时间": r.用药停止时间,
                "药物名称": r.药物名称,
                "给定剂量": r.给定剂量,
                "剂量单位": r.剂量单位,
                "给药路线": r.给药路线,
            }
            for r in results
        ]

    def build_icu_xt_series(
        self,
        *,
        hadm_id: int,
        stay_id: Optional[int] = None,
        vital_itemids: Optional[Sequence[int]] = None,
        include_all_chartevents: bool = False,
        chartevent_label_whitelist: Optional[Sequence[str]] = None,
        include_outputevents: bool = True,
        include_datetimeevents: bool = False,
        include_labevents: bool = False,
        include_inputevents: bool = False,
        include_omr: bool = False,
        pre_discharge_hours: Optional[int] = None,
        lab_itemids: Optional[Sequence[int]] = None,
        max_rows: int = 50000,
    ) -> Dict[str, Any]:
        """构建 ICU 监护状态时序 X_t。

        主路径：admissions + icustays + chartevents + d_items。
        可选补充：outputevents / datetimeevents / labevents / inputevents / omr。

        Args:
            hadm_id: 住院 ID（必填）
            stay_id: ICU stay_id（可选，缺省为该住院全部 ICU 段）
            vital_itemids: 生命体征 itemid 白名单；为空时使用常见 vital labels 过滤
            include_all_chartevents: 是否加载 ICU 窗口内全部数值型 chartevents（不再按 vital 标签过滤）
            chartevent_label_whitelist: chartevents label 白名单；为空时使用前100类默认白名单
            include_outputevents: 是否加入出量事件
            include_datetimeevents: 是否加入 datetimeevents 结构化时间事件
            include_labevents: 是否加入实验室序列
            include_inputevents: 是否加入 inputevents 干预事件（amount/rate）
            include_omr: 是否加入 omr 基础体征（按 chartdate 映射到时间轴）
            pre_discharge_hours: 若为正数，仅查询 ICU 出科前 N 小时数据（SQL 侧裁剪）
            lab_itemids: labevents itemid 白名单（include_labevents=True 时可选）
            max_rows: 每类事件查询上限，防止单次查询过大

        Returns:
            dict: {
                "hadm_id": int,
                "windows": [...],
                "x_t": [
                    {"stay_id", "charttime", "source", "itemid", "feature", "value", "valueuom"},
                    ...
                ]
            }
        """
        if hadm_id <= 0:
            raise ValueError("hadm_id must be positive")

        windows_query = text("""
            SELECT
                i.stay_id,
                i.subject_id,
                i.intime,
                i.outtime,
                a.admittime,
                a.dischtime
            FROM icustays i
            INNER JOIN admissions a ON a.hadm_id = i.hadm_id
            WHERE i.hadm_id = :hadm_id
              AND (:stay_id IS NULL OR i.stay_id = :stay_id)
              AND i.intime IS NOT NULL
              AND i.outtime IS NOT NULL
              AND i.outtime > i.intime
            ORDER BY i.intime ASC
            """)
        windows_rows = self._execute_mappings_with_retry(
            windows_query,
            {"hadm_id": int(hadm_id), "stay_id": stay_id},
        )
        if not windows_rows:
            return {"hadm_id": int(hadm_id), "windows": [], "x_t": []}

        rows_out: List[Dict[str, Any]] = []
        default_chartevent_labels_top100 = tuple(
            str(x).strip()
            for x in (
                chartevent_label_whitelist or DEFAULT_CHARTEVENT_LABEL_WHITELIST_TOP100
            )
            if str(x).strip()
        )

        for w in windows_rows:
            cur_stay_id = int(w["stay_id"])
            cur_subject_id = int(w["subject_id"])
            intime = w["intime"]
            outtime = w["outtime"]
            query_intime = intime
            if pre_discharge_hours is not None and int(pre_discharge_hours) > 0:
                clipped_start = outtime - timedelta(hours=int(pre_discharge_hours))
                if clipped_start > query_intime:
                    query_intime = clipped_start

            if include_all_chartevents:
                chart_sql = text("""
                    SELECT
                        ce.stay_id,
                        ce.charttime,
                        ce.itemid,
                        di.label AS feature,
                        ce.valuenum AS value,
                        ce.valueuom
                    FROM chartevents ce
                    INNER JOIN d_items di ON di.itemid = ce.itemid
                    WHERE ce.hadm_id = :hadm_id
                      AND ce.stay_id = :stay_id
                      AND ce.charttime >= :intime
                      AND ce.charttime <= :outtime
                      AND ce.valuenum IS NOT NULL
                    ORDER BY ce.charttime ASC
                    LIMIT :max_rows
                    """)
                chart_params = {
                    "hadm_id": int(hadm_id),
                    "stay_id": cur_stay_id,
                    "intime": query_intime,
                    "outtime": outtime,
                    "max_rows": int(max_rows),
                }
            elif vital_itemids:
                chart_sql = text("""
                    SELECT
                        ce.stay_id,
                        ce.charttime,
                        ce.itemid,
                        di.label AS feature,
                        ce.valuenum AS value,
                        ce.valueuom
                    FROM chartevents ce
                    INNER JOIN d_items di ON di.itemid = ce.itemid
                    WHERE ce.hadm_id = :hadm_id
                      AND ce.stay_id = :stay_id
                      AND ce.charttime >= :intime
                      AND ce.charttime <= :outtime
                      AND ce.valuenum IS NOT NULL
                      AND ce.itemid IN :itemids
                    ORDER BY ce.charttime ASC
                    LIMIT :max_rows
                    """).bindparams(bindparam("itemids", expanding=True))
                chart_params = {
                    "hadm_id": int(hadm_id),
                    "stay_id": cur_stay_id,
                    "intime": query_intime,
                    "outtime": outtime,
                    "itemids": [int(x) for x in vital_itemids],
                    "max_rows": int(max_rows),
                }
            else:
                chart_sql = text("""
                    SELECT
                        ce.stay_id,
                        ce.charttime,
                        ce.itemid,
                        di.label AS feature,
                        ce.valuenum AS value,
                        ce.valueuom
                    FROM chartevents ce
                    INNER JOIN d_items di ON di.itemid = ce.itemid
                    WHERE ce.hadm_id = :hadm_id
                      AND ce.stay_id = :stay_id
                      AND ce.charttime >= :intime
                      AND ce.charttime <= :outtime
                      AND ce.valuenum IS NOT NULL
                                            AND di.label IN :labels
                    ORDER BY ce.charttime ASC
                    LIMIT :max_rows
                    """).bindparams(bindparam("labels", expanding=True))
                chart_params = {
                    "hadm_id": int(hadm_id),
                    "stay_id": cur_stay_id,
                    "intime": query_intime,
                    "outtime": outtime,
                    "labels": list(default_chartevent_labels_top100),
                    "max_rows": int(max_rows),
                }

            chart_rows = self._execute_mappings_with_retry(chart_sql, chart_params)
            rows_out.extend(
                {
                    "stay_id": int(r["stay_id"]),
                    "charttime": r["charttime"],
                    "source": "chartevents",
                    "itemid": int(r["itemid"]),
                    "feature": str(r.get("feature") or ""),
                    "value": float(r["value"]),
                    "valueuom": str(r.get("valueuom") or ""),
                }
                for r in chart_rows
            )

            if include_outputevents:
                output_sql = text("""
                    SELECT
                        oe.stay_id,
                        oe.charttime,
                        oe.itemid,
                        di.label AS feature,
                        oe.value,
                        oe.valueuom
                    FROM outputevents oe
                    INNER JOIN d_items di ON di.itemid = oe.itemid
                    WHERE oe.hadm_id = :hadm_id
                      AND oe.stay_id = :stay_id
                      AND oe.charttime >= :intime
                      AND oe.charttime <= :outtime
                      AND oe.value IS NOT NULL
                    ORDER BY oe.charttime ASC
                    LIMIT :max_rows
                    """)
                output_rows = self._execute_mappings_with_retry(
                    output_sql,
                    {
                        "hadm_id": int(hadm_id),
                        "stay_id": cur_stay_id,
                        "intime": query_intime,
                        "outtime": outtime,
                        "max_rows": int(max_rows),
                    },
                )
                rows_out.extend(
                    {
                        "stay_id": int(r["stay_id"]),
                        "charttime": r["charttime"],
                        "source": "outputevents",
                        "itemid": int(r["itemid"]),
                        "feature": str(r.get("feature") or ""),
                        "value": float(r["value"]),
                        "valueuom": str(r.get("valueuom") or ""),
                    }
                    for r in output_rows
                )

            if include_datetimeevents:
                datetime_sql = text("""
                    SELECT
                        de.stay_id,
                        de.charttime,
                        de.itemid,
                        di.label AS feature,
                        1.0 AS value,
                        '' AS valueuom
                    FROM datetimeevents de
                    INNER JOIN d_items di ON di.itemid = de.itemid
                    WHERE de.hadm_id = :hadm_id
                      AND de.stay_id = :stay_id
                      AND de.charttime >= :intime
                      AND de.charttime <= :outtime
                    ORDER BY de.charttime ASC
                    LIMIT :max_rows
                    """)
                dt_rows = self._execute_mappings_with_retry(
                    datetime_sql,
                    {
                        "hadm_id": int(hadm_id),
                        "stay_id": cur_stay_id,
                        "intime": query_intime,
                        "outtime": outtime,
                        "max_rows": int(max_rows),
                    },
                )
                rows_out.extend(
                    {
                        "stay_id": int(r["stay_id"]),
                        "charttime": r["charttime"],
                        "source": "datetimeevents",
                        "itemid": int(r["itemid"]),
                        "feature": str(r.get("feature") or ""),
                        "value": float(r["value"]),
                        "valueuom": "",
                    }
                    for r in dt_rows
                )

            if include_labevents:
                lab_rows = self._fetch_labevents_chunked(
                    hadm_id=int(hadm_id),
                    stay_id=cur_stay_id,
                    intime=query_intime,
                    outtime=outtime,
                    max_rows=int(max_rows),
                    lab_itemids=lab_itemids,
                )
                rows_out.extend(
                    {
                        "stay_id": int(r["stay_id"]),
                        "charttime": r["charttime"],
                        "source": "labevents",
                        "itemid": int(r["itemid"]),
                        "feature": str(r.get("feature") or ""),
                        "value": float(r["value"]),
                        "valueuom": str(r.get("valueuom") or ""),
                    }
                    for r in lab_rows
                )

            if include_inputevents:
                input_rows = self._fetch_inputevents_chunked(
                    hadm_id=int(hadm_id),
                    stay_id=cur_stay_id,
                    intime=query_intime,
                    outtime=outtime,
                    max_rows=int(max_rows),
                )
                rows_out.extend(
                    {
                        "stay_id": int(r["stay_id"]),
                        "charttime": r["charttime"],
                        "source": "inputevents",
                        "itemid": int(r["itemid"]),
                        "feature": f"INPUT {str(r.get('feature') or '')}",
                        "value": float(r["value"]),
                        "valueuom": str(r.get("valueuom") or ""),
                    }
                    for r in input_rows
                    if r.get("charttime") is not None and r.get("value") is not None
                )

            if include_omr:
                omr_sql = text("""
                    SELECT
                        o.chartdate,
                        o.result_name,
                        o.result_value
                    FROM omr o
                    WHERE o.subject_id = :subject_id
                      AND o.chartdate >= DATE(:admittime)
                      AND o.chartdate <= DATE(:outtime)
                    ORDER BY o.chartdate ASC
                    LIMIT :max_rows
                    """)
                omr_rows = self._execute_mappings_with_retry(
                    omr_sql,
                    {
                        "subject_id": cur_subject_id,
                        "admittime": w["admittime"],
                        "outtime": outtime,
                        "max_rows": int(max_rows),
                    },
                )

                for r in omr_rows:
                    charttime = self._coerce_charttime(r.get("chartdate"))
                    if charttime is None:
                        continue
                    result_name = str(r.get("result_name") or "").strip()
                    result_value = r.get("result_value")
                    lower_name = result_name.lower()

                    if (
                        "blood pressure" in lower_name
                        and isinstance(result_value, str)
                        and "/" in result_value
                    ):
                        sbp, dbp = self._parse_bp_pair(result_value)
                        if sbp is not None:
                            rows_out.append(
                                {
                                    "stay_id": cur_stay_id,
                                    "charttime": charttime,
                                    "source": "omr",
                                    "itemid": 0,
                                    "feature": "Blood Pressure Systolic",
                                    "value": float(sbp),
                                    "valueuom": "mmHg",
                                }
                            )
                        if dbp is not None:
                            rows_out.append(
                                {
                                    "stay_id": cur_stay_id,
                                    "charttime": charttime,
                                    "source": "omr",
                                    "itemid": 0,
                                    "feature": "Blood Pressure Diastolic",
                                    "value": float(dbp),
                                    "valueuom": "mmHg",
                                }
                            )
                        continue

                    fv = self._try_parse_float(result_value)
                    if fv is None:
                        continue
                    rows_out.append(
                        {
                            "stay_id": cur_stay_id,
                            "charttime": charttime,
                            "source": "omr",
                            "itemid": 0,
                            "feature": f"OMR {result_name}" if result_name else "OMR",
                            "value": float(fv),
                            "valueuom": "",
                        }
                    )

        rows_out.sort(
            key=lambda x: (
                self._coerce_charttime(x.get("charttime")) or datetime.min,
                x["stay_id"],
                x["source"],
                x["itemid"],
            )
        )
        windows = [
            {
                "stay_id": int(w["stay_id"]),
                "subject_id": int(w["subject_id"]),
                "intime": w["intime"],
                "outtime": w["outtime"],
                "admittime": w["admittime"],
                "dischtime": w["dischtime"],
            }
            for w in windows_rows
        ]
        return {
            "hadm_id": int(hadm_id),
            "vital_label_hints": list(default_chartevent_labels_top100),
            "windows": windows,
            "x_t": rows_out,
        }

    def build_icu_xt_series_batch(
        self,
        *,
        stay_rows: List[Any],
        vital_itemids: Optional[Sequence[int]] = None,
        include_all_chartevents: bool = False,
        chartevent_label_whitelist: Optional[Sequence[str]] = None,
        include_outputevents: bool = True,
        include_datetimeevents: bool = False,
        include_labevents: bool = False,
        include_inputevents: bool = False,
        include_omr: bool = False,
        pre_discharge_hours: Optional[int] = None,
        lab_itemids: Optional[Sequence[int]] = None,
        batch_size: int = 100,
    ) -> Dict[int, Dict[str, Any]]:
        """Batch version of build_icu_xt_series for resolving N+1 queries.
        Returns a dict mapping stay_id -> payload.
        """
        out_payloads: Dict[int, Dict[str, Any]] = {}
        if not stay_rows:
            return out_payloads

        default_chartevent_labels_top100 = tuple(
            str(x).strip()
            for x in (
                chartevent_label_whitelist or DEFAULT_CHARTEVENT_LABEL_WHITELIST_TOP100
            )
            if str(x).strip()
        )

        # We will process stay_rows in chunks to avoid overly large IN (...) clauses
        for i in range(0, len(stay_rows), batch_size):
            chunk_rows = stay_rows[i : i + batch_size]
            chunk_stay_ids = []
            chunk_hadm_ids = set()
            chunk_subject_ids = set()

            # Extract attributes dynamically (supports dict or dataclass)
            stay_map = {}
            for s in chunk_rows:
                sid = int(
                    getattr(
                        s, "stay_id", s.get("stay_id") if isinstance(s, dict) else 0
                    )
                )
                hid = int(
                    getattr(
                        s, "hadm_id", s.get("hadm_id") if isinstance(s, dict) else 0
                    )
                )
                sub_id = int(
                    getattr(
                        s,
                        "subject_id",
                        s.get("subject_id") if isinstance(s, dict) else 0,
                    )
                )
                chunk_stay_ids.append(sid)
                chunk_hadm_ids.add(hid)
                chunk_subject_ids.add(sub_id)

                # compute window query limits per stay
                intime = getattr(
                    s, "intime", s.get("intime") if isinstance(s, dict) else None
                )
                outtime = getattr(
                    s, "outtime", s.get("outtime") if isinstance(s, dict) else None
                )

                query_intime = intime
                if (
                    pre_discharge_hours is not None
                    and int(pre_discharge_hours) > 0
                    and outtime is not None
                ):
                    clipped_start = outtime - timedelta(hours=int(pre_discharge_hours))
                    if clipped_start > query_intime:
                        query_intime = clipped_start

                stay_map[sid] = {
                    "stay_id": sid,
                    "hadm_id": hid,
                    "subject_id": sub_id,
                    "intime": intime,
                    "outtime": outtime,
                    "query_intime": query_intime,
                }
                out_payloads[sid] = {
                    "hadm_id": hid,
                    "vital_label_hints": list(default_chartevent_labels_top100),
                    "windows": [
                        {
                            "stay_id": sid,
                            "subject_id": sub_id,
                            "intime": intime,
                            "outtime": outtime,
                            "admittime": getattr(
                                s, "admittime", intime
                            ),  # Fallback to intime if admittime not provided
                            "dischtime": getattr(s, "dischtime", outtime),
                        }
                    ],
                    "x_t": [],
                }

            chunk_hadm_ids = list(chunk_hadm_ids)
            chunk_subject_ids = list(chunk_subject_ids)

            # Fetch chartevents
            if include_all_chartevents:
                chart_sql = text("""
                    SELECT ce.stay_id, ce.charttime, ce.itemid, di.label AS feature, ce.valuenum AS value, ce.valueuom
                    FROM chartevents ce INNER JOIN d_items di ON di.itemid = ce.itemid
                    WHERE ce.stay_id IN :stay_ids AND ce.valuenum IS NOT NULL
                    """).bindparams(bindparam("stay_ids", expanding=True))
                chart_params = {"stay_ids": chunk_stay_ids}
            elif vital_itemids:
                chart_sql = text("""
                    SELECT ce.stay_id, ce.charttime, ce.itemid, di.label AS feature, ce.valuenum AS value, ce.valueuom
                    FROM chartevents ce INNER JOIN d_items di ON di.itemid = ce.itemid
                    WHERE ce.stay_id IN :stay_ids AND ce.valuenum IS NOT NULL AND ce.itemid IN :itemids
                    """).bindparams(
                    bindparam("stay_ids", expanding=True),
                    bindparam("itemids", expanding=True),
                )
                chart_params = {
                    "stay_ids": chunk_stay_ids,
                    "itemids": [int(x) for x in vital_itemids],
                }
            else:
                chart_sql = text("""
                    SELECT ce.stay_id, ce.charttime, ce.itemid, di.label AS feature, ce.valuenum AS value, ce.valueuom
                    FROM chartevents ce INNER JOIN d_items di ON di.itemid = ce.itemid
                    WHERE ce.stay_id IN :stay_ids AND ce.valuenum IS NOT NULL AND di.label IN :labels
                    """).bindparams(
                    bindparam("stay_ids", expanding=True),
                    bindparam("labels", expanding=True),
                )
                chart_params = {
                    "stay_ids": chunk_stay_ids,
                    "labels": list(default_chartevent_labels_top100),
                }

            chart_rows = self._execute_mappings_with_retry(chart_sql, chart_params)
            for r in chart_rows:
                sid = int(r["stay_id"])
                # Python side filtering for intime/outtime
                if (
                    r["charttime"] is None
                    or r["charttime"] < stay_map[sid]["query_intime"]
                    or r["charttime"] > stay_map[sid]["outtime"]
                ):
                    continue
                out_payloads[sid]["x_t"].append(
                    {
                        "stay_id": sid,
                        "charttime": r["charttime"],
                        "source": "chartevents",
                        "itemid": int(r["itemid"]),
                        "feature": str(r.get("feature") or ""),
                        "value": float(r["value"]),
                        "valueuom": str(r.get("valueuom") or ""),
                    }
                )

            # Fetch outputevents
            if include_outputevents:
                output_sql = text("""
                    SELECT oe.stay_id, oe.charttime, oe.itemid, di.label AS feature, oe.value, oe.valueuom
                    FROM outputevents oe INNER JOIN d_items di ON di.itemid = oe.itemid
                    WHERE oe.stay_id IN :stay_ids AND oe.value IS NOT NULL
                    """).bindparams(bindparam("stay_ids", expanding=True))
                output_rows = self._execute_mappings_with_retry(
                    output_sql, {"stay_ids": chunk_stay_ids}
                )
                for r in output_rows:
                    sid = int(r["stay_id"])
                    if (
                        r["charttime"] is None
                        or r["charttime"] < stay_map[sid]["query_intime"]
                        or r["charttime"] > stay_map[sid]["outtime"]
                    ):
                        continue
                    out_payloads[sid]["x_t"].append(
                        {
                            "stay_id": sid,
                            "charttime": r["charttime"],
                            "source": "outputevents",
                            "itemid": int(r["itemid"]),
                            "feature": str(r.get("feature") or ""),
                            "value": float(r["value"]),
                            "valueuom": str(r.get("valueuom") or ""),
                        }
                    )

            # Fetch datetimeevents
            if include_datetimeevents:
                datetime_sql = text("""
                    SELECT de.stay_id, de.charttime, de.itemid, di.label AS feature, 1.0 AS value, '' AS valueuom
                    FROM datetimeevents de INNER JOIN d_items di ON di.itemid = de.itemid
                    WHERE de.stay_id IN :stay_ids
                    """).bindparams(bindparam("stay_ids", expanding=True))
                dt_rows = self._execute_mappings_with_retry(
                    datetime_sql, {"stay_ids": chunk_stay_ids}
                )
                for r in dt_rows:
                    sid = int(r["stay_id"])
                    if (
                        r["charttime"] is None
                        or r["charttime"] < stay_map[sid]["query_intime"]
                        or r["charttime"] > stay_map[sid]["outtime"]
                    ):
                        continue
                    out_payloads[sid]["x_t"].append(
                        {
                            "stay_id": sid,
                            "charttime": r["charttime"],
                            "source": "datetimeevents",
                            "itemid": int(r["itemid"]),
                            "feature": str(r.get("feature") or ""),
                            "value": float(r["value"]),
                            "valueuom": "",
                        }
                    )

            # Fetch labevents
            if include_labevents and chunk_hadm_ids:
                if lab_itemids:
                    lab_sql = text("""
                        SELECT le.hadm_id, le.charttime, le.itemid, le.labevent_id,
                               COALESCE(dl.label, CONCAT('LAB:', CAST(le.itemid AS CHAR))) AS feature,
                               le.valuenum AS value, le.valueuom
                        FROM labevents le LEFT JOIN d_labitems dl ON dl.itemid = le.itemid
                        WHERE le.hadm_id IN :hadm_ids AND le.valuenum IS NOT NULL AND le.itemid IN :itemids
                        """).bindparams(
                        bindparam("hadm_ids", expanding=True),
                        bindparam("itemids", expanding=True),
                    )
                    lab_params = {
                        "hadm_ids": chunk_hadm_ids,
                        "itemids": [int(x) for x in lab_itemids],
                    }
                else:
                    lab_sql = text("""
                        SELECT le.hadm_id, le.charttime, le.itemid, le.labevent_id,
                               COALESCE(dl.label, CONCAT('LAB:', CAST(le.itemid AS CHAR))) AS feature,
                               le.valuenum AS value, le.valueuom
                        FROM labevents le LEFT JOIN d_labitems dl ON dl.itemid = le.itemid
                        WHERE le.hadm_id IN :hadm_ids AND le.valuenum IS NOT NULL
                        """).bindparams(bindparam("hadm_ids", expanding=True))
                    lab_params = {"hadm_ids": chunk_hadm_ids}

                lab_rows = self._execute_mappings_with_retry(lab_sql, lab_params)
                for r in lab_rows:
                    hid = int(r["hadm_id"])
                    # Find matching stay_id based on hadm_id and time bounds
                    for sid, info in stay_map.items():
                        if (
                            info["hadm_id"] == hid
                            and r["charttime"] is not None
                            and info["query_intime"]
                            <= r["charttime"]
                            <= info["outtime"]
                        ):
                            out_payloads[sid]["x_t"].append(
                                {
                                    "stay_id": sid,
                                    "charttime": r["charttime"],
                                    "source": "labevents",
                                    "itemid": int(r["itemid"]),
                                    "feature": str(r.get("feature") or ""),
                                    "value": float(r["value"]),
                                    "valueuom": str(r.get("valueuom") or ""),
                                }
                            )

            # Fetch inputevents
            if include_inputevents:
                input_sql = text("""
                    SELECT ie.stay_id, COALESCE(ie.starttime, ie.endtime) AS charttime, ie.itemid,
                           di.label AS feature,
                           CASE WHEN ie.amount IS NOT NULL THEN ie.amount WHEN ie.rate IS NOT NULL THEN ie.rate ELSE NULL END AS value,
                           CASE WHEN ie.amount IS NOT NULL THEN ie.amountuom ELSE ie.rateuom END AS valueuom
                    FROM inputevents ie INNER JOIN d_items di ON di.itemid = ie.itemid
                    WHERE ie.stay_id IN :stay_ids AND (ie.amount IS NOT NULL OR ie.rate IS NOT NULL)
                    """).bindparams(bindparam("stay_ids", expanding=True))
                input_rows = self._execute_mappings_with_retry(
                    input_sql, {"stay_ids": chunk_stay_ids}
                )
                for r in input_rows:
                    sid = int(r["stay_id"])
                    if (
                        r["charttime"] is None
                        or r["charttime"] < stay_map[sid]["query_intime"]
                        or r["charttime"] > stay_map[sid]["outtime"]
                    ):
                        continue
                    out_payloads[sid]["x_t"].append(
                        {
                            "stay_id": sid,
                            "charttime": r["charttime"],
                            "source": "inputevents",
                            "itemid": int(r["itemid"]),
                            "feature": f"INPUT {str(r.get('feature') or '')}",
                            "value": float(r["value"]),
                            "valueuom": str(r.get("valueuom") or ""),
                        }
                    )

            # Fetch omr
            if include_omr and chunk_subject_ids:
                omr_sql = text("""
                    SELECT o.subject_id, o.chartdate, o.result_name, o.result_value
                    FROM omr o
                    WHERE o.subject_id IN :subject_ids
                    """).bindparams(bindparam("subject_ids", expanding=True))
                omr_rows = self._execute_mappings_with_retry(
                    omr_sql, {"subject_ids": chunk_subject_ids}
                )
                for r in omr_rows:
                    sub_id = int(r["subject_id"])
                    charttime = self._coerce_charttime(r.get("chartdate"))
                    if charttime is None:
                        continue

                    for sid, info in stay_map.items():
                        if (
                            info["subject_id"] == sub_id
                            and info["query_intime"].date()
                            <= charttime.date()
                            <= info["outtime"].date()
                        ):
                            result_name = str(r.get("result_name") or "").strip()
                            result_value = r.get("result_value")
                            lower_name = result_name.lower()

                            if (
                                "blood pressure" in lower_name
                                and isinstance(result_value, str)
                                and "/" in result_value
                            ):
                                sbp, dbp = self._parse_bp_pair(result_value)
                                if sbp is not None:
                                    out_payloads[sid]["x_t"].append(
                                        {
                                            "stay_id": sid,
                                            "charttime": charttime,
                                            "source": "omr",
                                            "itemid": 0,
                                            "feature": "Blood Pressure Systolic",
                                            "value": float(sbp),
                                            "valueuom": "mmHg",
                                        }
                                    )
                                if dbp is not None:
                                    out_payloads[sid]["x_t"].append(
                                        {
                                            "stay_id": sid,
                                            "charttime": charttime,
                                            "source": "omr",
                                            "itemid": 0,
                                            "feature": "Blood Pressure Diastolic",
                                            "value": float(dbp),
                                            "valueuom": "mmHg",
                                        }
                                    )
                                continue

                            fv = self._try_parse_float(result_value)
                            if fv is None:
                                continue
                            out_payloads[sid]["x_t"].append(
                                {
                                    "stay_id": sid,
                                    "charttime": charttime,
                                    "source": "omr",
                                    "itemid": 0,
                                    "feature": (
                                        f"OMR {result_name}" if result_name else "OMR"
                                    ),
                                    "value": float(fv),
                                    "valueuom": "",
                                }
                            )

        # Sort all x_t arrays
        for sid, payload in out_payloads.items():
            payload["x_t"].sort(
                key=lambda x: (
                    self._coerce_charttime(x.get("charttime")) or datetime.min,
                    x["stay_id"],
                    x["source"],
                    x["itemid"],
                )
            )

        return out_payloads

    def get_complete_patient_data(self, hadm_id: int) -> Dict[str, Any]:
        """获取单个患者的完整数据（适用于博弈分析）.

        Args:
            hadm_id: 住院ID

        Returns:
            完整患者数据字典
        """
        return {
            "basic_info": self.get_patient_basic_info(hadm_id),
            "admission_details": self.get_admission_details(hadm_id),
            "icu_stays": self.get_icu_stays(hadm_id),
            "transfers": self.get_transfers(hadm_id),
            "diagnoses": self.get_diagnoses(hadm_id),
            "procedures": self.get_procedures(hadm_id),
            "prescriptions": self.get_prescriptions(hadm_id, limit=100),
        }


def extract_and_export_sample_data(
    session: Session, output_format: str = "dict"
) -> Any:
    """提取并导出样本数据用于分析.

    Args:
        session: 数据库会话
        output_format: 输出格式 ('dict', 'json', 'dataframe')

    Returns:
        根据output_format返回不同格式的数据
    """
    extractor = MIMIC4DataExtractor(session)

    # Step 0: Create eligible patients table
    count = extractor.extract_eligible_patients(
        min_los_days=3,
        min_transfers=2,
        min_diagnoses=1,
        min_age=18,
        max_age=89,
        limit=1000,
    )

    logger.info(f"Created {count} eligible patients")

    # Get summary
    summary = extractor.get_eligible_patients_summary()
    logger.info(f"Summary: {summary}")

    # Get sample patients
    sample_patients = extractor.get_eligible_patients_sample(limit=10)

    result = {
        "summary": summary,
        "sample_patients": sample_patients,
    }

    if output_format == "dict":
        return result
    elif output_format == "json":
        import json

        return json.dumps(result, ensure_ascii=False, indent=2, default=str)
    elif output_format == "dataframe":
        try:
            import pandas as pd

            return {
                "summary": pd.DataFrame([summary]),
                "sample_patients": pd.DataFrame(sample_patients),
            }
        except ImportError:
            logger.error("pandas not installed, returning dict format")
            return result
    else:
        return result
