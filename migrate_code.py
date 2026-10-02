import os
import shutil
import re
from pathlib import Path

src_pancan = Path('src/pancan')
src_root = Path('src')

# Files to migrate and clean up
files_to_migrate = [
    'config.py',
    'data_utils.py',
    'evaluate.py',
    'pca_evaluation.py',
    'run_experiment.py',
    'plot_tstr.py'
]

for file_name in files_to_migrate:
    src_file = src_pancan / file_name
    dest_file = src_root / file_name
    
    if not src_file.exists():
        continue
        
    with open(src_file, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # 1. Update imports: "src.pancan." -> "src."
    content = content.replace('src.pancan.config', 'src.config')
    content = content.replace('src.pancan.data_utils', 'src.data_utils')
    content = content.replace('src.pancan.evaluate', 'src.evaluate')
    content = content.replace('src.pancan.models', 'src.models')
    
    # 2. Specifically fix pca_evaluation.py's mkdir bug
    if file_name == 'pca_evaluation.py':
        # Remove any lingering broken module-level mkdirs
        content = content.replace('OUTPUT_DIR = cfg.RESULTS_DIR / "pca_plots"', '')
        content = content.replace('OUTPUT_DIR.mkdir(parents=True, exist_ok=True)', '')
        
        # Ensure mkdir is called immediately at the start of main()
        # Find def main():
        main_match = re.search(r'def main\(\):', content)
        if main_match:
            main_start = main_match.end()
            insert_mkdir = '\n    (cfg.RESULTS_DIR / "pca_plots").mkdir(parents=True, exist_ok=True)\n'
            content = content[:main_start] + insert_mkdir + content[main_start:]
            
    # Write to destination
    with open(dest_file, 'w', encoding='utf-8') as f:
        f.write(content)
        
    print(f"Migrated and updated: {file_name}")

# Now update the SLURM bash scripts
slurm_dir = Path('slurm')
for sh_file in ['submit_pipeline.sh', 'submit_eval.sh']:
    sh_path = slurm_dir / sh_file
    if sh_path.exists():
        with open(sh_path, 'r', encoding='utf-8') as f:
            content = f.read()
        # Update python module target
        content = content.replace('src.pancan.run_experiment', 'src.run_experiment')
        with open(sh_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Updated SLURM script: {sh_file}")

print("Done migrating code.")
