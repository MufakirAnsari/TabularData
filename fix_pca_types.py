import sys
import re

with open('src/pca_evaluation.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Make sure y_real and y_syn are cast to strings right after loading
content = content.replace(
    'X_real, _, y_real, _ = stratified_split(X, y, TRAIN_RATIO, SEED)',
    'X_real, _, y_real, _ = stratified_split(X, y, TRAIN_RATIO, SEED)\n    y_real = y_real.astype(str)'
)

content = content.replace(
    'X_syn, y_syn = load_synthetic(syn_path, lbl_path)',
    'X_syn, y_syn = load_synthetic(syn_path, lbl_path)\n        y_syn = y_syn.astype(str)'
)

with open('src/pca_evaluation.py', 'w', encoding='utf-8') as f:
    f.write(content)
