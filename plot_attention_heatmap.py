import os
import torch
import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
from transformers import AutoTokenizer, AutoModel

import sys
sys.path.append('.')
from train_readmission_mimic3_with_notes import CrossModalAttnFusion

print("Connecting to DuckDB...")
con = duckdb.connect()

print("Fetching top 64 ICD9 codes...")
vocab_df = con.query("""
    SELECT SUBSTRING(CAST(ICD9_CODE AS VARCHAR), 1, 3) as code, count(*) as cnt
    FROM read_csv_auto('/home/hanwen/data/mimic/iii/DIAGNOSES_ICD.csv', sample_size=-1)
    WHERE ICD9_CODE IS NOT NULL
    GROUP BY code ORDER BY cnt DESC LIMIT 64
""").df()
icd_vocab = vocab_df['code'].tolist()

print("Finding an interesting patient stay...")
# Find a patient with a discharge summary
stays = con.query("""
    SELECT s.HADM_ID, s.ICUSTAY_ID
    FROM read_csv_auto('/home/hanwen/data/mimic/iii/ICUSTAYS.csv', sample_size=-1) s
    JOIN read_csv_auto('/home/hanwen/data/mimic/iii/NOTEEVENTS.csv', sample_size=-1) n ON s.HADM_ID = n.HADM_ID
    WHERE n.CATEGORY = 'Discharge summary'
    LIMIT 100
""").df()

hadm_id = stays['HADM_ID'].iloc[0]

print(f"Fetching note for HADM_ID {hadm_id}...")
notes = con.query(f"""
    SELECT TEXT
    FROM read_csv_auto('/home/hanwen/data/mimic/iii/NOTEEVENTS.csv', sample_size=-1)
    WHERE HADM_ID = {hadm_id} AND CATEGORY = 'Discharge summary'
    LIMIT 1
""").df()
note_text = notes['TEXT'].iloc[0]

print("Fetching ICD codes for patient...")
patient_icd = con.query(f"""
    SELECT SUBSTRING(CAST(ICD9_CODE AS VARCHAR), 1, 3) as code
    FROM read_csv_auto('/home/hanwen/data/mimic/iii/DIAGNOSES_ICD.csv', sample_size=-1)
    WHERE HADM_ID = {hadm_id}
""").df()['code'].tolist()

# Which top 64 codes does this patient have?
patient_icd_indices = [i for i, code in enumerate(icd_vocab) if code in patient_icd]
patient_icd_codes = [icd_vocab[i] for i in patient_icd_indices]

print(f"Patient has these top-64 ICD codes: {patient_icd_codes}")

print("Loading ClinicalBERT...")
tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
bert_model = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT').eval()

print("Splitting note into sentences and embedding...")
# A simple sentence split
sentences = [s.strip() for s in note_text.replace('\n', ' ').split('.') if len(s.strip()) > 10]
# take max 20 sentences for visualization
sentences = sentences[-20:] if len(sentences) > 20 else sentences

sentence_embeddings = []
with torch.no_grad():
    for s in sentences:
        inputs = tokenizer(s, return_tensors='pt', truncation=True, max_length=512)
        out = bert_model(**inputs)
        # pooled output
        emb = out.pooler_output
        sentence_embeddings.append(emb)

x_note_sentences = torch.cat(sentence_embeddings, dim=0) # [num_sentences, 768]

print("Loading trained CrossModalAttnFusion model...")
# Determine dimensions based on the training script
static_dims = {'age': 0, 'los_residual': 0, 'GENDER': 2, 'MARITAL_STATUS': 8, 'ETHNICITY': 41, 'INSURANCE': 5}
multihot_dims = {'icd': 64, 'drg': 64, 'proc': 64, 'rx': 64}

model = CrossModalAttnFusion(seq_dim=8, static_dims=static_dims, multihot_dims=multihot_dims, hidden_dim=64, note_dim=768)
model_path = 'output/20260614_225244_ep200_lr1em03_hd64_bs256_seqnorm_cosinelr_posw_clinicalbert/model_cross_attn.pt'
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

print("Computing sentence-to-ICD attention...")
with torch.no_grad():
    # project each sentence
    query = model.note_to_mh_queries['icd'](x_note_sentences) # [num_sentences, 32]
    emb_weight = model.mh_emb_dict['icd'].weight # [64, 32]
    
    scores = query @ emb_weight.T # [num_sentences, 64]
    
    # mask out codes the patient DOES NOT have
    mask = torch.zeros(64)
    mask[patient_icd_indices] = 1
    scores = scores.masked_fill(mask.unsqueeze(0) == 0, -1e9)
    
    attn = torch.nn.functional.softmax(scores, dim=1) # [num_sentences, 64]

# extract only the columns for codes the patient has
attn_subset = attn[:, patient_icd_indices].numpy()

# Add ICD descriptions for better readability (hardcoded common ones for top 64)
icd_desc = {
    '401': 'Hypertension', '428': 'Heart Failure', '427': 'Dysrhythmias', '414': 'Coronary Atherosclerosis',
    '272': 'Hyperlipidaemia', '250': 'Diabetes', '584': 'Acute Renal Failure', '518': 'Respiratory Failure',
    '599': 'UTI', '285': 'Anemia', '276': 'Fluid/Electrolyte', '410': 'Acute Myocardial Infarction',
    '287': 'Purpura/Thrombocytopenia', '496': 'COPD', '585': 'Chronic Kidney Disease', 'V29': 'Observation newborn',
    '486': 'Pneumonia', '284': 'Aplastic Anemia', '038': 'Septicemia', '278': 'Obesity'
}

x_labels = [f"{code} ({icd_desc.get(code, 'Other')})" for code in patient_icd_codes]

y_labels = [textwrap.shorten(s, width=80, placeholder="...") for s in sentences]

print("Plotting heatmap...")
plt.figure(figsize=(14, 12))
sns.heatmap(attn_subset, annot=True, fmt=".2f", cmap="YlOrRd", 
            xticklabels=x_labels, 
            yticklabels=y_labels)
plt.title("Sentence-to-ICD Note-Guided Cross-Attention Heatmap", fontsize=16)
plt.xlabel("Present ICD-9 Codes", fontsize=14)
plt.ylabel("Discharge Summary Sentences", fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()

output_path = '/home/hanwen/.gemini/antigravity-ide/brain/6af91c35-b567-4b9f-bb12-82fa49a7a741/sentence_icd_attention.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Heatmap saved to {output_path}")
