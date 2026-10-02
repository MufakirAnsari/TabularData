import sys

with open('src/pancan/run_experiment.py', 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if line.startswith('from src.pancan.config import ('):
        new_lines.append('import src.pancan.config as cfg\n')
        new_lines.append(line)
    else:
        new_lines.append(line)

with open('src/pancan/run_experiment.py', 'w') as f:
    f.writelines(new_lines)
