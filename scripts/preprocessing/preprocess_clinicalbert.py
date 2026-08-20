#!/usr/bin/env python3
"""
Re-embed clinical note summaries using Bio_ClinicalBERT (768d).

Reads existing summaries from the Llama-generated embeddings file and produces
new ClinicalBERT embeddings. Source data is NOT modified.

Usage:
    python preprocess_clinicalbert.py
    python preprocess_clinicalbert.py --batch-size 128
"""
import os
import pickle
import logging
import argparse
import numpy as np
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

CLINICALBERT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
DEFAULT_INPUT_FILE  = "output/mimic3_note_embeddings.pkl"
DEFAULT_OUTPUT_FILE = "output/mimic3_note_embeddings_clinicalbert.pkl"


def main(input_file, output_file, batch_size=64):
    # ── 1. Load existing summaries ────────────────────────────────────────────
    if not os.path.exists(input_file):
        logging.error(f"Source embeddings not found: {input_file}")
        logging.error("Run preprocess_notes.py first to generate Llama summaries.")
        return

    logging.info(f"Loading summaries from {input_file} ...")
    with open(input_file, 'rb') as f:
        llama_dict = pickle.load(f)

    # Extract (stay_id, summary_text) pairs
    tasks = []
    for stay_id, val in llama_dict.items():
        if isinstance(val, dict):
            summary = val.get('summary', '')
        else:
            summary = ''
        # Only process if we have meaningful text
        if summary and len(summary.strip()) > 10:
            tasks.append((int(stay_id), summary.strip()))

    logging.info(f"Found {len(tasks)} summaries to embed (out of {len(llama_dict)} total entries)")

    # ── 2. Load ClinicalBERT ─────────────────────────────────────────────────
    from transformers import AutoTokenizer, AutoModel

    logging.info(f"Loading ClinicalBERT: {CLINICALBERT_MODEL} ...")
    tokenizer = AutoTokenizer.from_pretrained(CLINICALBERT_MODEL)
    model = AutoModel.from_pretrained(CLINICALBERT_MODEL)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    logging.info(f"ClinicalBERT loaded on {device}")

    emb_dim = model.config.hidden_size  # 768
    logging.info(f"Embedding dimension: {emb_dim}")

    # ── 3. Batch embedding ───────────────────────────────────────────────────
    results = {}
    n_truncated = 0

    for batch_start in tqdm(range(0, len(tasks), batch_size), desc="Embedding batches"):
        batch = tasks[batch_start : batch_start + batch_size]
        stay_ids = [t[0] for t in batch]
        texts    = [t[1] for t in batch]

        # Tokenize with truncation (ClinicalBERT max = 512 tokens)
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt',
        )

        # Count truncated
        for i, text in enumerate(texts):
            token_len = encoded['attention_mask'][i].sum().item()
            if token_len >= 512:
                n_truncated += 1

        input_ids      = encoded['input_ids'].to(device)
        attention_mask  = encoded['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Use [CLS] token embedding (first token of last hidden state)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [B, 768]

        cls_np = cls_embeddings.cpu().numpy().astype(np.float32)

        for i, sid in enumerate(stay_ids):
            results[sid] = cls_np[i]

    # Fill entries without summary with zero vectors
    n_zero = 0
    for stay_id in llama_dict:
        sid = int(stay_id)
        if sid not in results:
            results[sid] = np.zeros(emb_dim, dtype=np.float32)
            n_zero += 1

    logging.info(f"Embedding complete: {len(results)} entries")
    logging.info(f"  - With ClinicalBERT embedding: {len(results) - n_zero}")
    logging.info(f"  - Zero-filled (no summary): {n_zero}")
    logging.info(f"  - Truncated at 512 tokens: {n_truncated}")

    # ── 4. Save to new file ──────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)

    # Verify
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    logging.info(f"✅ Saved to {output_file} ({file_size_mb:.1f} MB)")

    # Quick sanity check
    with open(output_file, 'rb') as f:
        check = pickle.load(f)
    sample_key = list(check.keys())[0]
    sample_val = check[sample_key]
    logging.info(f"Sanity check: key={sample_key}, shape={sample_val.shape}, dtype={sample_val.dtype}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-embed clinical note summaries with ClinicalBERT")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT_FILE,
                        help="Path to input embeddings/summaries pickle file")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_FILE,
                        help="Path to save new ClinicalBERT embeddings pickle file")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for ClinicalBERT inference (default: 64)")
    args = parser.parse_args()
    main(input_file=args.input, output_file=args.output, batch_size=args.batch_size)
