import duckdb
import pandas as pd
import requests
import pickle
import os
import re
import argparse
import concurrent.futures
from tqdm import tqdm
import json
import random
import time

OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1:latest"
MAX_NOTE_LEN_PER_CAT = 8000

TARGET_CATEGORIES = [
    'Nursing/other', 'Radiology', 'Nursing', 'ECG', 'Physician ', 'Discharge summary', 'Echo'
]

EXPERIENCE_BASE_PATH = 'output/experience_base.json'
CASE_BASE_PATH = 'output/case_base.json'

def load_json_base(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except:
                return []
    return []

def save_json_base(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def clean_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {3,}', ' ', text)
    return text.strip()

def llm_generate(prompt, temperature=0.1, format=None, max_retries=3):
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False, "options": {"temperature": temperature}}
    if format:
        payload["format"] = format
    for attempt in range(max_retries):
        try:
            resp = requests.post(OLLAMA_GENERATE_URL, json=payload, timeout=300)
            if resp.status_code == 200:
                return resp.json().get("response", "").strip()
        except Exception as e:
            wait_time = min(2 ** attempt * 5, 60)
            print(f"LLM Generate error (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)
    print(f"LLM Generate failed after {max_retries} retries.")
    return ""

def agent_summarize_category(category, notes_text):
    notes_text = notes_text[:MAX_NOTE_LEN_PER_CAT]
    prompt = f"""
You are an expert clinical AI. Analyze the following {category} notes to extract critical information that might predict 30-day hospital mortality or ICU readmission.
Focus on identifying any abnormal findings, unresolved issues, or high-risk indicators specific to this domain.

{category} NOTES:
{notes_text}

Provide a concise, structured summary highlighting key risk factors. Do not include introductory text.
"""
    return llm_generate(prompt)

def agent_synthesize_meta(category_summaries, rag_retriever):
    combined = ""
    for cat, summ in category_summaries.items():
        if summ:
            combined += f"\n--- {cat} Summary ---\n{summ}\n"

    # RAG: retrieve top-5 most relevant experience rules instead of injecting all
    relevant_rules = rag_retriever.search_experience(combined, top_k=5)
    rules_text = "\n".join([f"- {r}" for r in relevant_rules]) if relevant_rules else "None"

    # RAG: retrieve the most similar case as few-shot example
    similar_cases = rag_retriever.search_cases(combined, top_k=1)
    few_shot = ""
    if similar_cases:
        c = similar_cases[0]
        few_shot = f"""
Example Similar Case:
Domain Summaries: {json.dumps(c['category_summaries'], ensure_ascii=False)[:500]}...
Correct Output JSON: {json.dumps(c['meta_summary'], ensure_ascii=False)}
"""

    prompt = f"""
You are the lead meta-agent clinical AI. You will receive summaries from various domain agents for a single patient's ICU stay.
Your task is to synthesize these summaries and produce a final, structured JSON report identifying TWO distinct risks:
1. In-hospital mortality risk.
2. 30-day ICU readmission risk.

### DOCTOR EXPERIENCE RULES TO FOLLOW:
{rules_text}
{few_shot}

### PATIENT DOMAIN SUMMARIES:
{combined}

Output your analysis strictly in the following JSON format, and nothing else:
{{
    "mortality_risk_score": <int from 1 to 10, where 10 is highest risk>,
    "icu_readmission_risk_score": <int from 1 to 10, where 10 is highest risk>,
    "critical_factors_list": ["factor 1", "factor 2"],
    "final_summary": "<A concise 2-3 sentence overall synthesis>"
}}
"""
    raw = llm_generate(prompt, format="json")
    try:
        parsed = json.loads(raw)
        return raw, parsed
    except:
        return raw, {}

def agent_reflection(category_summaries, meta_summary_json, error_context):
    prompt = f"""
You are the Clinical Reflection Agent. The Meta-Agent just made predictions about a patient's mortality and ICU readmission risk, but failed on one or both tasks.

ERRORS MADE:
{error_context}

Meta-Agent Output: {json.dumps(meta_summary_json, ensure_ascii=False)}

Domain Summaries it based its decision on:
{json.dumps(category_summaries, ensure_ascii=False)[:2000]}...

Analyze why the Meta-Agent failed specifically on the mismatched task(s). Did it overweigh a symptom for mortality? Did it ignore a resolved condition for readmission?
Provide a single, concise RULE (1-2 sentences) starting with either "[Mortality Rule]:" or "[Readmission Rule]:" that should be added to the experience base to prevent this mistake in the future.
DO NOT provide conversational text. Output ONLY the rule.
"""
    return llm_generate(prompt, temperature=0.3)

def main(limit=None, evolve_limit=None):
    print("🚀 Using DuckDB and Pandas to load MIMIC-III cohort and calculate Ground Truth...")
    con = duckdb.connect()
    
    # Load all ICU stays and mortality info
    try:
        stays_df = con.query(f"""
            SELECT i.SUBJECT_ID, i.HADM_ID, i.ICUSTAY_ID as stay_id, i.INTIME, i.OUTTIME,
                   COALESCE(a.HOSPITAL_EXPIRE_FLAG, 0) as mort_label
            FROM read_csv_auto('/home/hanwen/data/mimic/iii/ICUSTAYS.csv', sample_size=-1) i
            LEFT JOIN read_csv_auto('/home/hanwen/data/mimic/iii/ADMISSIONS.csv', sample_size=-1) a
            ON i.HADM_ID = a.HADM_ID
            WHERE i.HADM_ID IS NOT NULL
        """).df()
    except Exception as e:
        print(f"Join failed, falling back to dummy labels: {e}")
        stays_df = con.query(f"""
            SELECT SUBJECT_ID, HADM_ID, ICUSTAY_ID as stay_id, INTIME, OUTTIME, 0 as mort_label
            FROM read_csv_auto('/home/hanwen/data/mimic/iii/ICUSTAYS.csv', sample_size=-1)
            WHERE HADM_ID IS NOT NULL
        """).df()

    # Calculate 30-day ICU readmission
    stays_df['INTIME'] = pd.to_datetime(stays_df['INTIME'])
    stays_df['OUTTIME'] = pd.to_datetime(stays_df['OUTTIME'])
    stays_df = stays_df.sort_values(['SUBJECT_ID', 'INTIME'])
    
    stays_df['next_intime'] = stays_df.groupby('SUBJECT_ID')['INTIME'].shift(-1)
    stays_df['days_to_next'] = (stays_df['next_intime'] - stays_df['OUTTIME']).dt.total_seconds() / (24*3600)
    
    stays_df['readm_label'] = ((stays_df['days_to_next'] <= 30) & (stays_df['days_to_next'] >= 0)).astype(int)
    stays_df.loc[stays_df['mort_label'] == 1, 'readm_label'] = 0 # Dead patients cannot be readmitted
    
    if limit:
        stays_df = stays_df.head(limit)
    
    hadm_ids = tuple(stays_df['HADM_ID'].dropna().unique().astype(int).tolist())
    
    print(f"📊 Loading target NOTEEVENTS for {len(hadm_ids)} admissions...")
    cat_tuple = tuple(TARGET_CATEGORIES)
    
    notes_df = con.query(f"""
        SELECT SUBJECT_ID, HADM_ID, CHARTDATE, CHARTTIME, CATEGORY, TEXT
        FROM read_csv_auto('/home/hanwen/data/mimic/iii/NOTEEVENTS.csv', sample_size=-1)
        WHERE HADM_ID IN {hadm_ids} AND ISERROR IS NULL AND CATEGORY IN {cat_tuple}
    """).df()

    stays_df['OUTTIME'] = pd.to_datetime(stays_df['OUTTIME'])
    notes_df['CHARTTIME'] = pd.to_datetime(notes_df['CHARTTIME'], errors='coerce')
    notes_df['CHARTDATE'] = pd.to_datetime(notes_df['CHARTDATE'], errors='coerce')
    notes_df['TIME'] = notes_df['CHARTTIME'].fillna(notes_df['CHARTDATE'])
    
    output_file = 'output/mimic3_note_summaries.pkl'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    summaries_dict = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, 'rb') as f:
                summaries_dict = pickle.load(f)
        except: pass

    experience_rules = load_json_base(EXPERIENCE_BASE_PATH)
    case_base = load_json_base(CASE_BASE_PATH)

    from rag_retriever import RAGRetriever
    print("🔍 Building RAG indices for experience rules and case base...")
    rag = RAGRetriever()
    rag.build_experience_index(experience_rules)
    rag.build_case_index(case_base)

    stay_tasks = []
    
    print("🤖 Preparing multi-agent data payload...")
    for _, stay in stays_df.iterrows():
        stay_id = int(stay['stay_id'])
        hadm_id = int(stay['HADM_ID'])
        outtime = stay['OUTTIME']
        mort_label = int(stay['mort_label'])
        readm_label = int(stay['readm_label'])
        
        stay_notes = notes_df[(notes_df['HADM_ID'] == hadm_id) & (notes_df['TIME'] <= outtime)].copy()
        stay_notes = stay_notes.sort_values(by='TIME', ascending=False)
        
        is_processed = False
        if stay_id in summaries_dict:
            val = summaries_dict[stay_id]
            if isinstance(val, dict):
                is_processed = bool(val.get('meta_summary', ''))

        if not is_processed:
            category_notes = {}
            for _, note in stay_notes.iterrows():
                cat = note['CATEGORY']
                text = clean_text(note['TEXT'])
                if text:
                    if cat not in category_notes:
                        category_notes[cat] = []
                    category_notes[cat].append(text)
            
            stay_tasks.append({
                'stay_id': stay_id,
                'category_notes': category_notes,
                'mort_label': mort_label,
                'readm_label': readm_label
            })
            
    # -- EVOLUTION PHASE --
    if evolve_limit and evolve_limit > 0:
        print(f"🧬 Starting Evolution Phase on first {evolve_limit} cases...")
        evolve_tasks = stay_tasks[:evolve_limit]
        stay_tasks = stay_tasks[evolve_limit:]
        
        for task in tqdm(evolve_tasks, desc="Evolution Phase"):
            stay_id = task['stay_id']
            cat_notes = task['category_notes']
            mort_label = task['mort_label']
            readm_label = task['readm_label']
            
            if not cat_notes: continue
                
            cat_summaries = {}
            for cat, texts in cat_notes.items():
                combined = "\n\n---\n\n".join(texts)
                summ = agent_summarize_category(cat, combined)
                if summ:
                    cat_summaries[cat] = summ
                    
            if not cat_summaries: continue
            
            raw_meta, parsed_meta = agent_synthesize_meta(cat_summaries, rag)
            
            mort_score = parsed_meta.get("mortality_risk_score", 5)
            readm_score = parsed_meta.get("icu_readmission_risk_score", 5)
            
            try: mort_score = int(mort_score)
            except: mort_score = 5
            try: readm_score = int(readm_score)
            except: readm_score = 5
            
            pred_mort = 1 if mort_score >= 7 else 0
            pred_readm = 1 if readm_score >= 7 else 0
            
            is_wrong = False
            error_prompts = []
            
            if pred_mort != mort_label:
                is_wrong = True
                error_prompts.append(f"Mortality Prediction Mismatch! Predicted {'High Risk (1)' if pred_mort else 'Low Risk (0)'}, but actual ground truth is {'High Risk (1)' if mort_label else 'Low Risk (0)'}.")
            if pred_readm != readm_label:
                is_wrong = True
                error_prompts.append(f"ICU Readmission Prediction Mismatch! Predicted {'High Risk (1)' if pred_readm else 'Low Risk (0)'}, but actual ground truth is {'High Risk (1)' if readm_label else 'Low Risk (0)'}.")
            
            if is_wrong:
                print(f"❌ Prediction mismatch for Stay {stay_id}. Reflecting...")
                error_context = "\n".join(error_prompts)
                new_rule = agent_reflection(cat_summaries, parsed_meta, error_context)
                if new_rule and len(new_rule) > 10:
                    print(f"   [New Rule Learned]: {new_rule}")
                    experience_rules.append(new_rule)
                    save_json_base(EXPERIENCE_BASE_PATH, experience_rules)
                    rag.add_experience(new_rule)
            else:
                print(f"✅ Prediction accurate for Stay {stay_id} on BOTH tasks. Adding to Case Base.")
                new_case = {
                    "stay_id": stay_id,
                    "category_summaries": cat_summaries,
                    "meta_summary": parsed_meta,
                    "mort_label": mort_label,
                    "readm_label": readm_label
                }
                case_base.append(new_case)
                save_json_base(CASE_BASE_PATH, case_base)
                rag.add_case(new_case)
                
            summaries_dict[stay_id] = {
                'category_summaries': cat_summaries,
                'meta_summary': parsed_meta,
                'raw_meta_response': raw_meta,
                'mort_label': mort_label,
                'readm_label': readm_label
            }
            with open(output_file, 'wb') as f:
                pickle.dump(summaries_dict, f)
                
    # -- PRODUCTION PHASE --
    if stay_tasks:
        print(f"⚡ Starting Production Phase (Parallel) on {len(stay_tasks)} cases...")
        
        def process_task_prod(task):
            stay_id = task['stay_id']
            cat_notes = task['category_notes']
            mort_label = task['mort_label']
            readm_label = task['readm_label']
            if not cat_notes: return stay_id, {}, "{}", {}, mort_label, readm_label
                
            cat_summaries = {}
            for cat, texts in cat_notes.items():
                combined = "\n\n---\n\n".join(texts)
                summ = agent_summarize_category(cat, combined)
                if summ: cat_summaries[cat] = summ
                    
            if not cat_summaries: return stay_id, {}, "{}", {}, mort_label, readm_label
                
            raw_meta, parsed_meta = agent_synthesize_meta(cat_summaries, rag)
            return stay_id, cat_summaries, raw_meta, parsed_meta, mort_label, readm_label

        MAX_WORKERS = 4
        save_counter = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_task_prod, task): task for task in stay_tasks}
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(stay_tasks)):
                stay_id, cat_sums, raw, parsed, mort_label, readm_label = future.result()
                summaries_dict[stay_id] = {
                    'category_summaries': cat_sums,
                    'meta_summary': parsed,
                    'raw_meta_response': raw,
                    'mort_label': mort_label,
                    'readm_label': readm_label
                }
                save_counter += 1
                if save_counter % 100 == 0:
                    with open(output_file, 'wb') as f:
                        pickle.dump(summaries_dict, f)

        with open(output_file, 'wb') as f:
            pickle.dump(summaries_dict, f)
        
    print(f"✅ Successfully completed. Total {len(summaries_dict)} summaries saved to {output_file}")
    print(f"🧬 Final Experience Rules count: {len(experience_rules)}")
    print(f"🏆 Final Golden Cases count: {len(case_base)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess MIMIC-III note embeddings")
    parser.add_argument("--limit", type=int, default=None, help="Total limit of cases")
    parser.add_argument("--evolve-limit", type=int, default=None, help="Number of cases to use for sequential evolution")
    args = parser.parse_args()
    main(limit=args.limit, evolve_limit=args.evolve_limit)
