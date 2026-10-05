import csv, hashlib, json, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'data/raw'

def fetch(url, path):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={'User-Agent':'ClearText-CS4701-research'})
        with urllib.request.urlopen(req, timeout=90) as r:
            path.write_bytes(r.read())
    return path

def download():
    config = {
      'MMU-TDMLab/CompLex': ('master', [(f'{d}/lcp_single_{s}.tsv',f'complex_{s}.tsv') for d,s in [('train','train'),('trial','trial'),('test-labels','test')]]),
      'yukiar/CEFR-SP': ('main', [(f'CEFR-SP/{d}/CEFR-SP_{stem}_{s}.txt',f'cefr_{d}_{s}.tsv') for d,stem in [('SCoRE','SCoRE'),('Wiki-Auto','Wikiauto')] for s in ['train','dev','test']]),
      'LaSTUS-TALN-UPF/TSAR-2022-Shared-Task': ('main',[(f'datasets/{s}/tsar2022_en_{s}_gold.tsv',f'tsar_{s}.tsv') for s in ['trial','test']] + [('tsar_eval.py','tsar_eval.py')]),
      'p-lambda/swords': ('main',[(f'assets/parsed/swords-v1.1_{s}.json.gz',f'swords_{s}.json.gz') for s in ['dev','test']])
    }
    manifest=[]
    for repo,(branch,files) in config.items():
        commit_file = RAW / (repo.split('/')[-1]+'.commit.json')
        commit = json.loads(fetch(f'https://api.github.com/repos/{repo}/commits/{branch}',commit_file).read_text())['sha']
        for source,name in files:
            url=f'https://raw.githubusercontent.com/{repo}/{commit}/{source}'
            p=fetch(url,RAW/name)
            manifest.append({'file':name,'url':url,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'commit':commit})
    (RAW/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest

def word_data():
    out={}
    for split,name in [('train','train'),('dev','trial'),('test','test')]:
        with open(RAW/f'complex_{name}.tsv',newline='') as f:
            rows=list(csv.DictReader(f,delimiter='\t',quoting=csv.QUOTE_NONE))
        for r in rows:
            r['label']=float(r['complexity'])
            r['text']=r['sentence']
        out[split]=rows
    assert [len(out[s]) for s in ['train','dev','test']]==[7662,421,917]
    return out

def sentence_data():
    out={s:[] for s in ['train','dev','test']}
    for split in out:
        for domain in ['SCoRE','Wiki-Auto']:
            for i,line in enumerate((RAW/f'cefr_{domain}_{split}.tsv').read_text().splitlines()):
                text,a,b=line.rsplit('\t',2)
                out[split].append({'id':f'{domain}-{split}-{i}','text':text,'a':int(a),'b':int(b),'label':(int(a)+int(b))/2,'corpus':domain})
    # Preserve released splits; remove exact normalized overlap from earlier splits.
    norm=lambda s:' '.join(s.lower().split())
    test={norm(r['text']) for r in out['test']}
    dev={norm(r['text']) for r in out['dev']}
    removed={}
    for split,excluded in [('train',test|dev),('dev',test)]:
        old=out[split]
        out[split]=[r for r in old if norm(r['text']) not in excluded]
        removed[split]=len(old)-len(out[split])
    return out,removed

def tsar_data(split):
    rows=[]
    for i,line in enumerate((RAW/f'tsar_{split}.tsv').read_text().splitlines()):
        fields=line.split('\t')
        rows.append({'id':f'tsar-{split}-{i}','text':fields[0],'target':fields[1],'gold':[x.lower().strip() for x in fields[2:] if x.strip()]})
    return rows

BENCHLS_URL='https://zenodo.org/api/records/2552393/files/BenchLS.zip/content'
BENCHLS_SHA256='3299fd8113e3da65d149c9327c56bc53bab44fd891cc167e0ed85546f9cb653a'  # Zenodo record 2552393

def benchls_data():
    """BenchLS (Paetzold & Specia 2016, CC BY 4.0): 929 sentences, a target word, and simpler substitutes
    ranked by annotators. Split 50/50 by target lemma into dev and holdout, so no target word crosses."""
    import zipfile,random
    path=fetch(BENCHLS_URL,RAW/'BenchLS.zip')
    digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest==BENCHLS_SHA256,digest
    (RAW/'benchls.manifest.json').write_text(json.dumps({'url':BENCHLS_URL,'sha256':digest,'license':'CC BY 4.0'},indent=2))
    text=zipfile.ZipFile(path).read('BenchLS/BenchLS.txt').decode('utf8')
    rows=[]
    for i,line in enumerate(text.splitlines()):
        f=line.split('\t')
        subs=[(int(x.split(':',1)[0]),x.split(':',1)[1].strip()) for x in f[3:] if ':' in x]
        rows.append({'id':f'benchls-{i}','text':f[0],'target':f[1],'index':int(f[2]),
                     'gold':[w.lower() for r,w in sorted(subs) if w.lower()!=f[1].lower()]})
    lemmas=sorted({r['target'].lower() for r in rows});random.Random(4701).shuffle(lemmas)
    dev=set(lemmas[:len(lemmas)//2])
    return {'dev':[r for r in rows if r['target'].lower() in dev],'holdout':[r for r in rows if r['target'].lower() not in dev]}

CWI_COMMIT='74816aec86e899f6f6a24e9a19d5d50607802d8d'  # sheffieldnlp/cwisharedtask2018-teaching
CWI_SHA256={'Train':'9e25bf48359d0f42d5ba58597fe2711817b3cad19393f92e25847e81b9f9bb26','Dev':'684dcaa8214a77b51d7811c3cf00d5d486e03f88bc545fdfba09ea431e4af56a','Test':'827c8fbbf34403b72f9b08ee9bb4c4eb05317bb71089e4df6b4ed89166199d1a'}

def cwi_download():
    """CWI 2018 English (News, WikiNews, Wikipedia) from the organizers' teaching mirror at a pinned commit.
    The repository has no license file: research use only, do not redistribute."""
    for split,digest in CWI_SHA256.items():
        url=f'https://raw.githubusercontent.com/sheffieldnlp/cwisharedtask2018-teaching/{CWI_COMMIT}/datasets/english/English_{split}.tsv'
        path=fetch(url,RAW/f'cwi_english_{split}.tsv')
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,(split,'CWI checksum mismatch')
