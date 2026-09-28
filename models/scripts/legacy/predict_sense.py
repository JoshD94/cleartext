import sys,json,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cleartext.sense_pipeline import SenseClearText
p=argparse.ArgumentParser()
p.add_argument('text')
p.add_argument('--full',action='store_true',help='Also apply experimental phrase and structure rules')
args=p.parse_args()
print(json.dumps(SenseClearText().analyze(args.text,structure=args.full,phrases=args.full),indent=2))
