# STEP 3: select & tune model

import sys
import getopt
import json
import time
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score



scoring = {'scoring': 'f1_weighted', 'cv': 5}
parameters = LogisticRegression(solver='liblinear', multi_class='auto').get_params()

param_options = [ f"{k}=" for k in parameters ]
score_options = [ f"{k}=" for k in scoring ]

options  = getopt.getopt(sys.argv[1:], "", param_options + score_options)

parameters.update({ k[2:] : v for k, v in options[0] if k[2:] in param_options })
scoring.update({ k[2:] : v for k, v in options[0] if k[2:] in scoring })

clf = LogisticRegression(**parameters)

X, y = np.load('offline_db/train-feature.npy'), np.load('offline_db/train-target.npy')

results = cross_val_score(clf, X, y, **scoring)
results = {
    'score_val': list(results),
    'score_val_mean': results.mean(),
    'score_val_std': results.std()
}

log = {
    'eval': results,
    'parameters': parameters,
    'scoring': scoring
}

output = json.dumps(log)
now = int(time.time())

with open(f'offline_db/validations/{now}.json', 'w') as f:
    f.write(output)

print(now)
print()
print(output)
