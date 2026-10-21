import numpy as np
from .features import base_features
def phrase_features(text):
    words=[w for w in text.split() if any(c.isalpha() for c in w)]
    rows=[base_features(w) for w in words]
    return {'tokens':len(rows),'characters':len(text),
            **{key+'_'+agg:float(fn([r[key] for r in rows])) if rows else 0
               for key in ['zipf','length','syllables','frequency_missing']
               for agg,fn in [('mean',np.mean),('max',np.max),('min',np.min)]}}
