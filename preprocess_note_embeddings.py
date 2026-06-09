import duckdb
import pandas as pd
import numpy as np
import requests
import pickle
import os
import re
import argparse
import concurrent.futures
from tqdm import tqdm

OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
MODEL_NAME = "llama3.1:latest"
MAX_NOTE_LEN = 12000

def clean_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {3,}', ' ', text)
    return text.strip()

def summarize_notes(notes_text):
    notes_text = notes_text[:MAX_NOTE_LEN]
    prompt = f"""
You are an expert clinical AI. Analyze the following ICU clinical notes and extract specific information that would be valuable for predicting the patient's likelihood of safe discharge vs readmission/mortality. 
Summarize the key clinical trajectory and explicitly list 'discharge signals' or 'risk factors'.

NOTES:
{notes_text}

Provide ONLY the summary and risk factors, without intro/outro text.
"""
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False, "options": {"temperature": 0.2}}
    try:
        resp = requests.post(OLLAMA_GENERATE_URL, json=payload, timeout=120)
        if resp.status_code == 200:
            return resp.json().get("response", "")
    except Exception as e:
        print(f"Generate error: {e}")
    return ""

def embed_text(text):
    if not text: return np.zeros(4096, dtype=np.float32)
    payload = {"model": MODEL_NAME, "prompt": text}
    try:
        resp = requests.post(OLLAMA_EMBED_URL, json=payload, timeout=120)
        if resp.status_code == 200:
            emb = resp.json().get("embedding", [])
            return np.array(emb, dtype=np.float32)
    except Exception as e:
        print(f"Embed error: {e}")
    return np.zeros(4096, dtype=np.float32)

def main(limit=None):
    print("🚀 Using DuckDB to load MIMIC-III cohort...")
    con = duckdb.connect()
    
    # We select a manageable cohort of 200 stays for demonstration
    limit_clause = f"LIMIT {limit}" if limit else ""
    stays_df = con.query(f"""
        SELECT SUBJECT_ID, HADM_ID, ICUSTAY_ID as stay_id, INTIME, OUTTIME
        FROM read_csv_auto('/home/hanwen/data/mimic/iii/ICUSTAYS.csv', sample_size=-1)
        WHERE HADM_ID IS NOT NULL
        {limit_clause}
    """).df()
    
    hadm_ids = tuple(stays_df['HADM_ID'].dropna().unique().astype(int).tolist())
    
    print(f"📊 Loading NOTEEVENTS for {len(hadm_ids)} admissions...")
    notes_df = con.query(f"""
        SELECT SUBJECT_ID, HADM_ID, CHARTDATE, CHARTTIME, CATEGORY, TEXT
        FROM read_csv_auto('/home/hanwen/data/mimic/iii/NOTEEVENTS.csv', sample_size=-1)
        WHERE HADM_ID IN {hadm_ids} AND ISERROR IS NULL
    """).df()

    # Convert times
    stays_df['INTIME'] = pd.to_datetime(stays_df['INTIME'])
    stays_df['OUTTIME'] = pd.to_datetime(stays_df['OUTTIME'])
    notes_df['CHARTTIME'] = pd.to_datetime(notes_df['CHARTTIME'], errors='coerce')
    notes_df['CHARTDATE'] = pd.to_datetime(notes_df['CHARTDATE'], errors='coerce')
    
    output_file = 'output/mimic3_note_embeddings.pkl'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    embeddings_dict = {}
    if os.path.exists(output_file):
        print(f"Loading existing embeddings from {output_file}...")
        try:
            with open(output_file, 'rb') as f:
                embeddings_dict = pickle.load(f)
        except Exception as e:
            print(f"Failed to load existing embeddings: {e}")

    stay_tasks = []
    
    print("🤖 Preparing data for parallel generation...")
    for _, stay in stays_df.iterrows():
        stay_id = int(stay['stay_id'])
        hadm_id = int(stay['HADM_ID'])
        outtime = stay['OUTTIME']
        
        stay_notes = notes_df[notes_df['HADM_ID'] == hadm_id]
        
        valid_notes = []
        for _, note in stay_notes.iterrows():
            t = note['CHARTTIME'] if not pd.isna(note['CHARTTIME']) else note['CHARTDATE']
            if pd.isna(t) or t <= outtime:
                valid_notes.append(note['TEXT'])
                
        is_processed = False
        if stay_id in embeddings_dict:
            val = embeddings_dict[stay_id]
            if isinstance(val, dict):
                is_processed = np.any(val.get('embedding', []))
            else:
                is_processed = np.any(val)

        if not is_processed:
            stay_tasks.append({
                'stay_id': stay_id,
                'valid_notes': valid_notes
            })
        
    def process_task(task):
        stay_id = task['stay_id']
        valid_notes = task['valid_notes']
        if not valid_notes:
            return stay_id, "", np.zeros(4096, dtype=np.float32)
            
        combined_text = "\n\n---\n\n".join([clean_text(txt) for txt in valid_notes])
        summary = summarize_notes(combined_text)
        emb = embed_text(summary)
        return stay_id, summary, emb

    print("⚡ Generating Summaries and Embeddings via Ollama (Concurrent Batching)...")
    # Using 8-16 workers is usually enough to fully saturate a 48GB GPU with Ollama's auto-batching.
    MAX_WORKERS = 8
    save_counter = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_task, task): task for task in stay_tasks}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(stay_tasks)):
            stay_id, summary, emb = future.result()
            embeddings_dict[stay_id] = {'summary': summary, 'embedding': emb}
            save_counter += 1
            if save_counter % 500 == 0:
                with open(output_file, 'wb') as f:
                    pickle.dump(embeddings_dict, f)

    with open(output_file, 'wb') as f:
        pickle.dump(embeddings_dict, f)
    
    print(f"✅ Successfully saved {len(embeddings_dict)} embeddings to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess MIMIC-III note embeddings")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of cases to process")
    args = parser.parse_args()
    main(limit=args.limit)
