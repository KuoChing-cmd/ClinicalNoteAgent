import re

with open('train_readmission_mimic3_with_notes.py', 'r') as f:
    content = f.read()

# 1. Replace _multihot_to_embedding
old_multihot = r"    def _multihot_to_embedding\(self,\s*x_group[: \w\.]*,\s*emb[: \w\.]*\)[ \w\.\-]*>?[ \w\.]*:\n\s*summed = x_group @ emb\.weight\n\s*denom[ ]*=[ ]*torch\.clamp\(x_group\.sum\(dim=1, keepdim=True\), min=1\.0\)\n\s*return summed / denom"
new_multihot = """    def _multihot_to_embedding(self, x_group, emb, query_proj=None, x_note=None):
        import torch
        import torch.nn.functional as F
        if x_note is not None and query_proj is not None:
            query = query_proj(x_note)
            scores = query @ emb.weight.T
            scores = scores.masked_fill(x_group == 0, -1e9)
            attn = F.softmax(scores, dim=1)
            has_codes = (x_group.sum(dim=1, keepdim=True) > 0).float()
            return (attn @ emb.weight) * has_codes
        else:
            summed = x_group @ emb.weight
            denom  = torch.clamp(x_group.sum(dim=1, keepdim=True), min=1.0)
            return summed / denom"""

content, count = re.subn(old_multihot, new_multihot, content)
print(f"Replaced _multihot_to_embedding {count} times")

# 2. Replace _encode_multihot
old_encode = r"    def _encode_multihot\(self, x_mh\):\n\s*mh_embs, col_offset = \[\], 0\n\s*for name, vocab_size in self\.multihot_dims\.items\(\):\n\s*group = x_mh\[:, col_offset : col_offset \+ vocab_size\]\n\s*mh_embs\.append\(self\._multihot_to_embedding\(group, self\.mh_emb_dict\[name\]\)\)\n\s*col_offset \+= vocab_size\n\s*return self\.mh_head\(torch\.cat\(mh_embs, dim=1\)\)"

new_encode = """    def _encode_multihot(self, x_mh, x_note=None):
        mh_embs, col_offset = [], 0
        import torch
        for name, vocab_size in self.multihot_dims.items():
            group = x_mh[:, col_offset : col_offset + vocab_size]
            query_proj = getattr(self, 'note_to_mh_queries', {}).get(name, None)
            mh_embs.append(self._multihot_to_embedding(group, self.mh_emb_dict[name], query_proj, x_note))
            col_offset += vocab_size
        return self.mh_head(torch.cat(mh_embs, dim=1))"""

content, count = re.subn(old_encode, new_encode, content)
print(f"Replaced _encode_multihot {count} times")

# 3. Add self.note_to_mh_queries in constructors
# Let's hook after self.mh_head definition
old_init = r"(self\.mh_head = nn\.Sequential\([\s\S]*?\n\s*\))"
new_init = r"\1\n        if getattr(self, 'use_notes', True) and getattr(self, 'note_dim', None) is not None:\n            import torch.nn as nn\n            self.note_to_mh_queries = nn.ModuleDict()\n            for name, vocab_size in self.multihot_dims.items():\n                emb_dim = max(8, min(32, vocab_size // 4))\n                self.note_to_mh_queries[name] = nn.Linear(self.note_dim, emb_dim)"

content, count = re.subn(old_init, new_init, content)
print(f"Added note_to_mh_queries {count} times")

# Wait, `TransformerEarlyFusionWithNotes` and `LSTMLateFusionWithNotes` define multihot directly in `forward`!
# Let's check them.
with open('train_readmission_mimic3_with_notes.py', 'w') as f:
    f.write(content)

