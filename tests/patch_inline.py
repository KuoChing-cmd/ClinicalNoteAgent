import re

with open('train_readmission_mimic3_with_notes.py', 'r') as f:
    content = f.read()

old_inline = r"(for name, vocab_size in self\.multihot_dims\.items\(\):\n\s*group = x_mh\[:, col_offset : col_offset \+ vocab_size\]\n\s*)mh_embs\.append\(self\._multihot_to_embedding\(group, self\.mh_emb_dict\[name\]\)\)"

new_inline = r"\1query_proj = getattr(self, 'note_to_mh_queries', {}).get(name, None)\n                mh_embs.append(self._multihot_to_embedding(group, self.mh_emb_dict[name], query_proj, x_note))"

content, count = re.subn(old_inline, new_inline, content)
print(f"Replaced inline multihot {count} times")

with open('train_readmission_mimic3_with_notes.py', 'w') as f:
    f.write(content)

