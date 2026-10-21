import json,pickle,time,warnings
import numpy as np
from scipy.stats import pearsonr,spearmanr
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge,ElasticNet
from sklearn.ensemble import RandomForestRegressor,HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error

def metrics(y,p):
    y=np.asarray(y);p=np.asarray(p)
    return {'mae':float(mean_absolute_error(y,p)),'rmse':float(mean_squared_error(y,p)**.5),
            'pearson':float(pearsonr(y,p).statistic) if np.std(p)>1e-10 else None,
            'spearman':float(spearmanr(y,p).statistic) if np.std(p)>1e-10 else None}

def fit_compare(features,rows,run_dir,prefix,kind='word'):
    y={s:np.array([r['label'] for r in rows[s]]) for s in rows}
    low,high=(0,1) if kind=='word' else (1,6)
    specs=[('Mean',DummyRegressor(),None),
           ('Frequency' if kind=='word' else 'Length',Ridge(alpha=1),['zipf' if kind=='word' else 'length']),
           ('Ridge',Ridge(alpha=10),None),('Elastic net',ElasticNet(alpha=.0005,l1_ratio=.3,max_iter=3000),None),
           ('Random forest',RandomForestRegressor(n_estimators=140,min_samples_leaf=8,max_features=.8,n_jobs=4,random_state=4701),None),
           ('Gradient boosting',HistGradientBoostingRegressor(max_iter=150,max_leaf_nodes=15,l2_regularization=5,early_stopping=True,random_state=4701),None),
           ('SVR',SVR(C=1,epsilon=.03 if kind=='word' else .15),None),
           ('kNN',KNeighborsRegressor(n_neighbors=30,weights='distance',n_jobs=4),None),
           ('Small MLP',MLPRegressor(hidden_layer_sizes=(32,16),alpha=.1,max_iter=150,early_stopping=True,random_state=4701),None)]
    if kind=='sentence':
        specs.insert(2,('Word average',Ridge(alpha=1),['word_mean']))
        specs.insert(3,('Lexical ridge',Ridge(alpha=10),['length','word_mean','word_max','word_p90','mean_zipf','rare_fraction','mean_syllables','mean_word_length','lexical_density']))
    results=[]; fitted={}; matrices={}
    for name,est,subset in specs:
        use={s:([{k:r[k] for k in subset} for r in features[s]] if subset else features[s]) for s in features}
        pipe=make_pipeline(DictVectorizer(sparse=False),StandardScaler(),est)
        start=time.time()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');pipe.fit(use['train'],y['train'])
        dev=np.clip(pipe.predict(use['dev']),low,high)
        result={'name':name,'dev':metrics(y['dev'],dev),'fit_seconds':round(time.time()-start,3),'warnings':list(dict.fromkeys(str(w.message) for w in caught)),'features':subset or list(features['train'][0])}
        results.append(result);fitted[name]=pipe;matrices[name]=use
        print(prefix,name,'dev MAE',round(result['dev']['mae'],4),flush=True)
    selected=min(results,key=lambda r:r['dev']['mae'])['name']
    # Lock choice before exposing held-out metrics. Never refit to test labels.
    (run_dir/f'{prefix}_selection.json').write_text(json.dumps({'selected':selected,'criterion':'development MAE','comparisons':results},indent=2))
    for r in results:
        p=np.clip(fitted[r['name']].predict(matrices[r['name']]['test']),low,high)
        r['test']=metrics(y['test'],p)
        r['selected']=r['name']==selected
    best=fitted[selected]
    # Best model needs to retain selected feature subset for downstream inference.
    chosen=next(s for n,e,s in specs if n==selected)
    bundle=Scorer(best,chosen,low,high)
    with open(run_dir/f'{prefix}_model.pkl','wb') as f:pickle.dump(bundle,f)
    pred=bundle.predict(features['test'])
    domains={d:metrics([r['label'] for r in rows['test'] if r['corpus']==d],[p for r,p in zip(rows['test'],pred) if r['corpus']==d]) for d in sorted({r['corpus'] for r in rows['test']})}
    records=[{**r,'prediction':float(p),'absolute_error':float(abs(r['label']-p))} for r,p in zip(rows['test'],pred)]
    (run_dir/f'{prefix}_predictions.json').write_text(json.dumps(records,indent=2))
    return bundle,{'selected':selected,'results':results,'domains':domains,'test_predictions':records,'counts':{s:len(rows[s]) for s in rows}}

class Scorer:
    def __init__(self,model,subset,low,high):self.model,self.subset,self.low,self.high=model,subset,low,high
    def predict(self,features):
        use=[{k:r[k] for k in self.subset} for r in features] if self.subset else features
        return np.clip(self.model.predict(use),self.low,self.high)
