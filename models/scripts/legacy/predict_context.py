import sys,json,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cleartext.context_pipeline import ContextClearText
p=argparse.ArgumentParser(description='Experimental context, phrase and syntax pipeline')
p.add_argument('text')
p.add_argument('--no-structure',action='store_true')
p.add_argument('--no-phrases',action='store_true')
args=p.parse_args()
print(json.dumps(ContextClearText().analyze(args.text,structure=not args.no_structure,phrases=not args.no_phrases),indent=2))
