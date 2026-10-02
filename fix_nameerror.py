import sys

with open('src/pancan/run_experiment.py', 'r') as f:
    content = f.read()

# Fix the CANCER_TYPES NameError
content = content.replace(
'''parser.add_argument("--cancer_type", type=str, choices=CANCER_TYPES + ["all"], default="all")''',
'''parser.add_argument("--cancer_type", type=str, default="all")'''
)

with open('src/pancan/run_experiment.py', 'w') as f:
    f.write(content)
