import os

def modify():
    with open('train_readmission_mimic3_with_notes.py', 'r') as f:
        lines = f.readlines()

    def replace_line(search, replace):
        for i, line in enumerate(lines):
            if search in line:
                lines[i] = line.replace(search, replace)

    # 1. EXP_CONFIG
    replace_line('"hidden_dim":  64,', '"hidden_dim":  128,\n    "num_layers":  4,\n    "nhead":       8,\n    "dropout":     0.2,')

    # 2. LSTMLateFusionWithNotes
    replace_line('def __init__(self, seq_dim, static_dims=None, multihot_dims=None, hidden_dim=64, note_dim=4096, use_notes=True):',
                 'def __init__(self, seq_dim, static_dims=None, multihot_dims=None, hidden_dim=64, note_dim=4096, use_notes=True, num_layers=4, dropout=0.2):')
    replace_line('self.lstm = nn.LSTM(input_size=seq_dim, hidden_size=hidden_dim, batch_first=True, dropout=0.1, num_layers=2)',
                 'self.lstm = nn.LSTM(input_size=seq_dim, hidden_size=hidden_dim, batch_first=True, dropout=dropout, num_layers=num_layers)')

    # 3. TransformerEarlyFusionWithNotes
    replace_line('class TransformerEarlyFusionWithNotes(nn.Module):', 'class TransformerEarlyFusionWithNotes(nn.Module):') # just a marker
    for i, line in enumerate(lines):
        if 'class TransformerEarlyFusionWithNotes(nn.Module):' in line:
            idx = i
            break
    lines[idx+1] = lines[idx+1].replace('def __init__(self, seq_dim, static_dims=None, multihot_dims=None, hidden_dim=64, note_dim=4096, use_notes=True):',
                                        'def __init__(self, seq_dim, static_dims=None, multihot_dims=None, hidden_dim=64, note_dim=4096, use_notes=True, num_layers=4, nhead=8, dropout=0.2):')
    for i in range(idx, idx+20):
        if 'self.pos_encoder = PositionalEncoding(hidden_dim, dropout=0.1)' in lines[i]:
            lines[i] = lines[i].replace('dropout=0.1', 'dropout=dropout')
        if 'encoder_layers = nn.TransformerEncoderLayer(' in lines[i]:
            lines[i] = lines[i].replace('nhead=4', 'nhead=nhead').replace('dropout=0.1', 'dropout=dropout')
        if 'self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=2)' in lines[i]:
            lines[i] = lines[i].replace('num_layers=2', 'num_layers=num_layers')

    # 4. CrossModalAttnFusion
    for i, line in enumerate(lines):
        if 'class CrossModalAttnFusion(nn.Module):' in line:
            idx = i
            break
    lines[idx+15] = lines[idx+15].replace('hidden_dim=64, note_dim=4096, nhead=4,', 'hidden_dim=64, note_dim=4096, nhead=8,')
    lines[idx+16] = lines[idx+16].replace('num_virtual_tokens=4, num_lstm_layers=2):', 'num_virtual_tokens=4, num_lstm_layers=4, dropout=0.2):')
    for i in range(idx, idx+35):
        if 'dropout=0.1 if num_lstm_layers > 1 else 0.0' in lines[i]:
            lines[i] = lines[i].replace('dropout=0.1', 'dropout=dropout')
        if 'embed_dim=hidden_dim, num_heads=nhead,' in lines[i]:
            pass # already nhead variable
        if 'dropout=0.1, batch_first=True' in lines[i]:
            lines[i] = lines[i].replace('dropout=0.1', 'dropout=dropout')

    # 5. GatedFusionWithNotes
    for i, line in enumerate(lines):
        if 'class GatedFusionWithNotes(nn.Module):' in line:
            idx = i
            break
    lines[idx+14] = lines[idx+14].replace('hidden_dim=64, note_dim=4096, num_lstm_layers=2):', 'hidden_dim=64, note_dim=4096, num_lstm_layers=4, dropout=0.2):')
    for i in range(idx, idx+25):
        if 'dropout=0.1 if num_lstm_layers > 1 else 0.0' in lines[i]:
            lines[i] = lines[i].replace('dropout=0.1', 'dropout=dropout')

    # 6. TransformerSeqEncoder
    for i, line in enumerate(lines):
        if 'class TransformerSeqEncoder(nn.Module):' in line:
            idx = i
            break
    lines[idx+1] = lines[idx+1].replace('def __init__(self, seq_input_dim, hidden_dim, num_layers=2):', 'def __init__(self, seq_input_dim, hidden_dim, num_layers=4, nhead=8, dropout=0.2):')
    for i in range(idx, idx+15):
        if 'self.pos_encoder = PositionalEncoding(hidden_dim, dropout=0.1)' in lines[i]:
            lines[i] = lines[i].replace('dropout=0.1', 'dropout=dropout')
        if 'nhead=4,' in lines[i]:
            lines[i] = lines[i].replace('nhead=4', 'nhead=nhead')
        if 'dropout=0.1,' in lines[i]:
            lines[i] = lines[i].replace('dropout=0.1', 'dropout=dropout')

    # 7. PretrainedTransformerCrossModalFusion
    for i, line in enumerate(lines):
        if 'class PretrainedTransformerCrossModalFusion(nn.Module):' in line:
            idx = i
            break
    lines[idx+2] = lines[idx+2].replace('hidden_dim=64, note_dim=4096):', 'hidden_dim=64, note_dim=4096, nhead=8, dropout=0.2):')
    for i in range(idx, idx+25):
        if 'embed_dim=hidden_dim, num_heads=4,' in lines[i]:
            lines[i] = lines[i].replace('num_heads=4', 'num_heads=nhead')
        if 'dropout=0.1, batch_first=True' in lines[i]:
            lines[i] = lines[i].replace('dropout=0.1', 'dropout=dropout')

    # 8. Main script instantiations
    replace_line("model_base = LSTMLateFusionWithNotes(seq_dim=8, static_dims=ordered_static_dims, multihot_dims=multihot_dims, use_notes=False)",
                 "model_base = LSTMLateFusionWithNotes(seq_dim=8, static_dims=ordered_static_dims, multihot_dims=multihot_dims, use_notes=False, num_layers=EXP_CONFIG['num_layers'], dropout=EXP_CONFIG['dropout'])")
    replace_line("model_notes = LSTMLateFusionWithNotes(seq_dim=8, static_dims=ordered_static_dims, multihot_dims=multihot_dims, note_dim=EXP_CONFIG['note_dim'], use_notes=True)",
                 "model_notes = LSTMLateFusionWithNotes(seq_dim=8, static_dims=ordered_static_dims, multihot_dims=multihot_dims, note_dim=EXP_CONFIG['note_dim'], use_notes=True, num_layers=EXP_CONFIG['num_layers'], dropout=EXP_CONFIG['dropout'])")
    replace_line("model_tf_notes = TransformerEarlyFusionWithNotes(seq_dim=8, static_dims=ordered_static_dims, multihot_dims=multihot_dims, note_dim=EXP_CONFIG['note_dim'], use_notes=True)",
                 "model_tf_notes = TransformerEarlyFusionWithNotes(seq_dim=8, static_dims=ordered_static_dims, multihot_dims=multihot_dims, note_dim=EXP_CONFIG['note_dim'], use_notes=True, num_layers=EXP_CONFIG['num_layers'], nhead=EXP_CONFIG['nhead'], dropout=EXP_CONFIG['dropout'])")
    replace_line("pretrained_encoder = TransformerSeqEncoder(seq_input_dim=8, hidden_dim=EXP_CONFIG['hidden_dim'], num_layers=2)",
                 "pretrained_encoder = TransformerSeqEncoder(seq_input_dim=8, hidden_dim=EXP_CONFIG['hidden_dim'], num_layers=EXP_CONFIG['num_layers'], nhead=EXP_CONFIG['nhead'], dropout=EXP_CONFIG['dropout'])")
    
    # For CrossModalAttnFusion inside main
    for i, line in enumerate(lines):
        if 'model_cross = CrossModalAttnFusion(' in line:
            idx = i
            break
    lines[idx+3] = lines[idx+3].replace('nhead=4, num_virtual_tokens=4', 'nhead=EXP_CONFIG["nhead"], num_virtual_tokens=4, num_lstm_layers=EXP_CONFIG["num_layers"], dropout=EXP_CONFIG["dropout"]')

    # For GatedFusionWithNotes inside main
    for i, line in enumerate(lines):
        if 'model_gated = GatedFusionWithNotes(' in line:
            idx = i
            break
    lines[idx+2] = lines[idx+2].replace('hidden_dim=EXP_CONFIG[\'hidden_dim\'], note_dim=EXP_CONFIG[\'note_dim\'],',
                                        'hidden_dim=EXP_CONFIG[\'hidden_dim\'], note_dim=EXP_CONFIG[\'note_dim\'], num_lstm_layers=EXP_CONFIG["num_layers"], dropout=EXP_CONFIG["dropout"]')

    # For PretrainedTransformerCrossModalFusion inside main
    for i, line in enumerate(lines):
        if 'model_pretrain_cross = PretrainedTransformerCrossModalFusion(' in line:
            idx = i
            break
    lines[idx+3] = lines[idx+3].replace('hidden_dim=EXP_CONFIG[\'hidden_dim\'], note_dim=EXP_CONFIG[\'note_dim\'],',
                                        'hidden_dim=EXP_CONFIG[\'hidden_dim\'], note_dim=EXP_CONFIG[\'note_dim\'], nhead=EXP_CONFIG["nhead"], dropout=EXP_CONFIG["dropout"]')


    with open('train_readmission_mimic3_with_notes.py', 'w') as f:
        f.writelines(lines)

if __name__ == '__main__':
    modify()
