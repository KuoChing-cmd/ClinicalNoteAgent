with open('train_readmission_mimic3_with_notes.py', 'r') as f:
    code = f.read()

code = code.replace("if name in ['age', 'los_residual']:", "if self.static_dims[name] == 0:")
code = code.replace("if 'age' in self.static_dims:\n                static_repr_dim += 1\n            if 'los_residual' in self.static_dims:\n                static_repr_dim += 1", "static_repr_dim += sum(1 for v in self.static_dims.values() if v == 0)")
code = code.replace("if 'age' in self.static_dims:\n                static_repr_dim += 1\n            if 'los_residual' in self.static_dims:\n                static_repr_dim += 1", "static_repr_dim += sum(1 for v in self.static_dims.values() if v == 0)")

with open('train_readmission_mimic3_with_notes.py', 'w') as f:
    f.write(code)
print("Done")
