import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cleartext.pipeline import ClearText
p=argparse.ArgumentParser()
p.add_argument('text');p.add_argument('--mode',choices=['unchanged','dictionary','learned','context','guarded'],default='guarded')
p.add_argument('--threshold',type=float,default=.3)
args=p.parse_args()
print(json.dumps(ClearText().analyze(args.text,args.mode,args.threshold),indent=2))
