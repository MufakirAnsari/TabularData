import sys

with open('src/pancan/run_experiment.py', 'r') as f:
    content = f.read()

content = content.replace(
'''parser.add_argument("--model", type=str, required=True, choices=MODEL_NAMES + ["all"])''',
'''parser.add_argument("--dataset", type=str, default="colon")
    parser.add_argument("--model", type=str, required=True, choices=MODEL_NAMES + ["all"])'''
)

with open('src/pancan/run_experiment.py', 'w') as f:
    f.write(content)
