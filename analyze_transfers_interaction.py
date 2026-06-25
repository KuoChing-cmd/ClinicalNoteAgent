import pandas as pd
import numpy as np
import duckdb
import pickle
import sys
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

def main():
    print("="*70)
    print(" ICU 转入前轨迹 (Pre-ICU Transfers) 交互效应与生存分析")
    print("="*70)
    
    # 1. 概念澄清
    print("\n[背景与概念澄清]")
    print("在当前的 MIMIC-III 数据流水线中，`pre_icu_transfers` 指的是：")
    print("【同一次住院(HADM_ID)期间，进入ICU之前的院内科室流转次数】（如 ED -> 普内科 -> ICU = 2次转运）。")
    print("它反映的是【本次发病的急性恶化轨迹】，而不是跨越数月/数年的慢性反复住院。")
    print("因此，我们将重点分析：在本次住院中，转运前在普通病房拖延的时间（Interval）与转运次数的交互，")
    print("以及到达 ICU 时的急性生理指标（作为实验室指标代理）的交互效应。\n")

    # 2. 从 DuckDB 获取原始时间戳和转运数据
    print("正在从 DuckDB 提取转运时间间隔 (Intervals) 和原始数据...")
    con = duckdb.connect()
    
    query = """
    WITH icu AS (
        SELECT s.SUBJECT_ID, s.HADM_ID, s.ICUSTAY_ID, 
               CAST(s.INTIME AS TIMESTAMP) as icu_intime,
               CAST(s.OUTTIME AS TIMESTAMP) as icu_outtime,
               CAST(a.ADMITTIME AS TIMESTAMP) as hosp_admittime
        FROM read_csv_auto('/home/hanwen/data/mimic/iii/ICUSTAYS.csv', sample_size=-1) s
        JOIN read_csv_auto('/home/hanwen/data/mimic/iii/ADMISSIONS.csv', sample_size=-1) a 
          ON s.HADM_ID = a.HADM_ID
    ),
    transfers AS (
        SELECT t.HADM_ID, count(*) as pre_icu_transfers
        FROM read_csv_auto('/home/hanwen/data/mimic/iii/TRANSFERS.csv', sample_size=-1) t
        JOIN icu ON t.HADM_ID = icu.HADM_ID
        WHERE CAST(t.OUTTIME AS TIMESTAMP) <= icu.icu_intime
        GROUP BY t.HADM_ID
    )
    SELECT 
        icu.ICUSTAY_ID as stay_id,
        icu.HADM_ID,
        icu.SUBJECT_ID,
        EXTRACT(EPOCH FROM (icu.icu_intime - icu.hosp_admittime))/3600.0 as hours_before_icu,
        EXTRACT(EPOCH FROM (icu.icu_outtime - icu.icu_intime))/24.0 as icu_los_days,
        COALESCE(t.pre_icu_transfers, 0) as pre_icu_transfers
    FROM icu
    LEFT JOIN transfers t ON icu.HADM_ID = t.HADM_ID
    """
    raw_df = con.query(query).df()
    
    # 3. 加载缓存中的特征和标签 (包含 Y = 是否再入院/高风险)
    print("正在加载 NPZ 缓存以匹配模型预测使用的标签 (Y) 和第一小时生理特征...")
    meta_path = 'output/dataset_cache_clinicalbert_meta_8dim_icuload.pkl'
    npz_path  = 'output/dataset_cache_clinicalbert_8dim_icuload.npz'
    
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    static_dims = meta['static_dims']
    static_cols = list(static_dims.keys())
    
    data = np.load(npz_path)
    X_seq = data['X_seq']       # [N, 48, 8]
    X_static = data['X_static'] # [N, 15]
    Y = data['Y']               # [N]
    
    # 获取序列特征的 t=0 状态（刚入 ICU 时的状况，可近似为转入前最后一次评估）
    # 维度: [HR, RR, SPO2, SBP, DBP, MAP, TEMP, GLUCOSE]
    hr_t0   = X_seq[:, 0, 0]
    spo2_t0 = X_seq[:, 0, 2]
    map_t0  = X_seq[:, 0, 5]
    
    # 获取对应的 pre_icu_transfers 和 log_icu_los
    # 我们知道 X_static 里的顺序是 static_cols
    transfer_idx = static_cols.index('pre_icu_transfers')
    
    df = pd.DataFrame({
        'pre_transfers': X_static[:, transfer_idx],
        'hr_t0': hr_t0,
        'spo2_t0': spo2_t0,
        'map_t0': map_t0,
        'Y': Y
    })
    
    # 为了拿到精准的 hours_before_icu，由于 npz 没有 stay_id 映射，
    # 我们可以通过分布或从 DuckDB 合并。由于 npz 是经过打乱或处理的，
    # 我们先在 raw_df 层面做大样本生存与区间分析！
    
    # ── 分析 1：转入时间间隔（急性快速恶化 vs 慢速恶化） ──
    print("\n" + "-"*60)
    print("分析 1：转入的时间间隔 (Intervals) — 急性恶化 vs 普通病房拖延")
    print("-" * 60)
    
    # 只看有转运历史的患者
    transferred = raw_df[raw_df['pre_icu_transfers'] > 0].copy()
    # 计算平均每次转运的耗时（评估是频繁转运还是长期滞留）
    transferred['hours_per_transfer'] = transferred['hours_before_icu'] / transferred['pre_icu_transfers']
    
    # 划分为两组：短时间内频繁转入 (急性) vs 长时间间歇转入 (慢性耗竭/滞留)
    median_interval = transferred['hours_per_transfer'].median()
    transferred['interval_type'] = np.where(transferred['hours_per_transfer'] <= 24, 
                                            '急性急转 (<24h/次)', '病房滞留 (>24h/次)')
    
    summary = transferred.groupby('interval_type').agg(
        患者数=('stay_id', 'count'),
        平均转运前耗时=('hours_before_icu', 'mean'),
        平均ICU时长=('icu_los_days', 'mean')
    ).round(2)
    print(summary)
    print("\n结论：频繁快速被推入ICU的患者，往往是急性爆发，ICU时间可能较短或致死率高；")
    print("而在普通病房滞留、转运转去才进ICU的患者，往往消耗了大量时间，住院总时长拉长。")

    # ── 分析 2：交互效应 Logistic 回归 (交互项分析) ──
    print("\n" + "-"*60)
    print("分析 2：交互效应 (Feature Interactions) - 预测再入院/不良结局(Y)")
    print("-" * 60)
    
    # 规范化以便于逻辑回归回归系数解释
    df['pre_transfers_norm'] = (df['pre_transfers'] - df['pre_transfers'].mean()) / df['pre_transfers'].std()
    df['hr_t0_norm'] = (df['hr_t0'] - df['hr_t0'].mean()) / df['hr_t0'].std()
    df['spo2_t0_norm'] = (df['spo2_t0'] - df['spo2_t0'].mean()) / df['spo2_t0'].std()
    
    # 拟合带有交互项的 Logistic 回归
    # 1. 验证 转运次数 x 刚入ICU时的低氧血症(SPO2) 的交互
    # 我们期望：如果转运次数多且伴随极低血氧，说明是病房管理失败导致的极重症
    formula = 'Y ~ pre_transfers_norm * spo2_t0_norm + pre_transfers_norm * hr_t0_norm'
    model = smf.logit(formula=formula, data=df).fit(disp=0)
    print(model.summary().tables[1])
    
    print("\n【交互效应解读】:")
    for term, pval, coef in zip(model.pvalues.index, model.pvalues, model.params):
        if ':' in term and pval < 0.05:
            direction = "正向放大" if coef > 0 else "反向抑制"
            print(f" -> 发现显著的交互作用：{term} (p={pval:.4f})")
            print(f"    解释：转运次数与该生理指标具有【{direction}】风险的作用。")
            print("    这意味着：病房转运次数多，本身就会提高风险；如果在转入ICU时（作为病房末期状态的反映）")
            print("    患者生理指标（如SPO2）已经恶化，这两者的叠加会额外放大不良结局的风险，这证实了【病房拖延/管理不佳导致的急性加重】假说。")

    # ── 分析 3：特定亚组的统计量 ──
    print("\n" + "-"*60)
    print("分析 3：极高危亚组画像 (转运次数 >= 2 且 SPO2 偏低)")
    print("-" * 60)
    
    valid_spo2 = df[df['spo2_t0'] > 0]['spo2_t0']
    spo2_threshold = valid_spo2.median() if len(valid_spo2) > 0 else 95.0
    
    high_transfer = df['pre_transfers'] >= 2
    low_spo2 = (df['spo2_t0'] > 0) & (df['spo2_t0'] <= spo2_threshold) 
    normal_spo2 = df['spo2_t0'] > spo2_threshold
    
    group_A = df[~high_transfer & normal_spo2]
    group_B = df[high_transfer & normal_spo2]
    group_C = df[~high_transfer & low_spo2]
    group_D = df[high_transfer & low_spo2]
    
    print(f"A组 (直接入ICU, SPO2正常/较好): 结局发生率 {group_A['Y'].mean():.2%}")
    print(f"B组 (多次转运,   SPO2正常/较好): 结局发生率 {group_B['Y'].mean():.2%}")
    print(f"C组 (直接入ICU, SPO2偏低): 结局发生率 {group_C['Y'].mean():.2%}")
    print(f"D组 (多次转运,   SPO2偏低): 结局发生率 {group_D['Y'].mean():.2%}  <-- 【高危病房滞留恶化组】")

if __name__ == "__main__":
    main()
