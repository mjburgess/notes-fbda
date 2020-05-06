# STEP 4: train on best model parameters & final eval score

import sys
import getopt
import json
import time

import numpy as np 
import json 

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

parameters = LogisticRegression(solver='liblinear', multi_class='auto').get_params()

if len(sys.argv) < 2:
    print('train.py VALIDATION')
    print()
    print(list(parameters.keys()))
    exit("need validation ID")


with open(f"offline_db/validations/{sys.argv[1]}.json") as conf:
    parameters.update(json.load(conf)['parameters'])


X_train, y_train = np.load('offline_db/train-feature.npy'), np.load('offline_db/train-target.npy')
X_test, y_test = np.load('offline_db/test-feature.npy'), np.load('offline_db/test-target.npy')
X, y = np.r_[X_train, X_test], np.r_[y_train, y_test]


# score on test
clf = LogisticRegression(**parameters)
clf.fit(X_train, y_train)
score_test = clf.score(X_test, y_test)

# retrain on *all* data
clf_all = LogisticRegression(**parameters)
clf_all.fit(X, y)
score_all = clf_all.score(X, y)

# only save fully-trained model
initial_type = [('float_input', FloatTensorType([1, 4]))]
onnx_model = convert_sklearn(clf_all, initial_types=initial_type)

# report partially-trained model score
log = {
    'parameters': parameters,
    'eval': {
        'score_test': score_test,
        'score_all': score_all
    }
}

output = json.dumps(log)
now = int(time.time())

with open(f'offline_db/models/{now}.json', 'w') as f:
    f.write(output)

with open(f'offline_db/models/{now}.onnx', 'wb') as f:
    f.write(onnx_model.SerializeToString())

print(now)
print(output)
