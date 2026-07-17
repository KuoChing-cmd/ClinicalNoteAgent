import os
import sys

def main():
    with open('train_readmission_mimic3_with_notes.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    os.makedirs('src', exist_ok=True)
    with open('src/__init__.py', 'w') as f:
        f.write("")
        
    # 1. models.py (lines 28 to 789)
    models_code = "".join(lines[28:790])
    with open('src/models.py', 'w', encoding='utf-8') as f:
        f.write("import torch\nimport torch.nn as nn\nimport math\n\n")
        f.write(models_code)
        
    # 2. engine.py (lines 790 to 1144)
    engine_code = "".join(lines[790:1145])
    with open('src/engine.py', 'w', encoding='utf-8') as f:
        f.write("import torch\nimport numpy as np\nimport logging\n")
        f.write("from torch.utils.data import DataLoader, TensorDataset\n")
        f.write("from sklearn.metrics import roc_auc_score, auc, precision_recall_curve\n")
        f.write("from tqdm import tqdm\n\n")
        f.write(engine_code)
        
    # 3. data_loader.py (lines 1145 to 1488)
    data_code = "".join(lines[1145:1488])
    with open('src/data_loader.py', 'w', encoding='utf-8') as f:
        f.write("import pandas as pd\nimport numpy as np\nimport duckdb\nimport logging\n\n")
        f.write(data_code)
        
    # 4. train.py (lines 1488 to end)
    train_code = "".join(lines[1488:])
    # Include original imports + local imports
    imports_code = "".join(lines[0:28])
    
    local_imports = """
from src.models import (
    LSTMLateFusionWithNotes, TransformerEarlyFusionWithNotes, CrossModalAttnFusion,
    GatedFusionWithNotes, TransformerSeqEncoder, TransformerPretrainer, PretrainedTransformerCrossModalFusion
)
from src.data_loader import fetch_mimic3_data, flatten_features
from src.engine import train_model, evaluate_model, _evaluate_xgb_probs, pretrain_transformer
"""
    with open('train.py', 'w', encoding='utf-8') as f:
        f.write(imports_code)
        f.write(local_imports)
        f.write(train_code)

if __name__ == '__main__':
    main()
