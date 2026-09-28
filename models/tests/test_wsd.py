import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from cleartext.wsd import SenseModel,context_features
def test_target_identity_does_not_enter_context():
    a=context_features(['They','saw','a','bank','.'],3)
    b=context_features(['They','saw','a','shore','.'],3)
    assert a==b
def test_training_frequency_baseline_is_train_only():
    m=SenseModel()
    m.counts={('bank','n'):{'bank.n.01':3,'depository_financial_institution.n.01':1}}
    prediction=m.predict(['river','bank'],1,'bank','n')
    assert prediction['sense']=='bank.n.01'
    assert all(r['lemma_examples']==4 for r in prediction['ranked'])
def test_context_can_change_selected_sense():
    m=SenseModel()
    rows=[{'sense':'river','prior':-1.,'nb':-3.,'vector':.1},
          {'sense':'financial','prior':-2.,'nb':-1.,'vector':.8}]
    assert m.rank_components(rows,{'nb':0.,'vector':0.})[0]['sense']=='river'
    assert m.rank_components(rows,{'nb':1.,'vector':2.})[0]['sense']=='financial'
def test_empty_context_does_not_produce_nan():
    m=SenseModel()
    p=m.predict(['bank'],0,'bank','n')
    assert not p['trained'] and all(np.isfinite(r['score']) for r in p['ranked'])
