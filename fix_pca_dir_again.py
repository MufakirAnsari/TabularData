import sys

with open('src/pancan/pca_evaluation.py', 'r') as f:
    content = f.read()

# Fix the mkdir issue - create it right after loading the dataset in main
content = content.replace(
'''    # "?"? 1. Load Real Data "?"?''',
'''    (cfg.RESULTS_DIR / "pca_plots").mkdir(parents=True, exist_ok=True)
    # "?"? 1. Load Real Data "?"?'''
)

# Fix the matplotlib UserWarning by using 'color=' instead of 'c='
content = content.replace(
'''c=get_colors()[c_type]''',
'''color=get_colors()[c_type]'''
)

with open('src/pancan/pca_evaluation.py', 'w') as f:
    f.write(content)
