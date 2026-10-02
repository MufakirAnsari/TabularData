import sys

with open('src/pancan/pca_evaluation.py', 'r') as f:
    content = f.read()

# Fix the main block properly this time
replacement = '''if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="colon")
    args = parser.parse_args()
    
    # Dynamic config initialization
    import pandas as pd
    import numpy as np
    temp_data_dir = cfg.PROJECT_ROOT / "Data" / args.dataset
    df_l = pd.read_csv(temp_data_dir / "labels.csv", index_col=0)
    col = "Class" if "Class" in df_l.columns else df_l.columns[0]
    discovered_classes = list(np.unique(df_l[col].astype(str).values))
    cfg.set_dataset(args.dataset, discovered_classes)
    
    main()
'''

content = content.replace('if __name__ == "__main__":\n    main()', replacement)

# Let's also remove CLASS_COLORS from being hardcoded, or just make it generate dynamic colors
# Since PANCAN had specific colors, let's use a dynamic colormap inside `plot_2d` and `plot_3d`
color_replacement = '''
# Colors for the 5 cancer types
CLASS_COLORS = {'BRCA': 'tab:blue', 'KIRC': 'tab:orange', 'LUAD': 'tab:green', 'PRAD': 'tab:red', 'COAD': 'tab:purple'}
'''

dynamic_colors = '''
def get_colors():
    colors = plt.cm.tab10.colors
    return {c_type: colors[i % len(colors)] for i, c_type in enumerate(cfg.CANCER_TYPES)}
'''
content = content.replace(color_replacement, dynamic_colors)
content = content.replace('CLASS_COLORS[c_type]', 'get_colors()[c_type]')

with open('src/pancan/pca_evaluation.py', 'w') as f:
    f.write(content)
