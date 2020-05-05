# STEP 5.2: use predictive model (eg., via docker container, via cloud, ...)

import sys 
import onnxruntime as rt
import numpy as np
import json
import time

from B_process import *

if len(sys.argv) < 6:
    print('predict.py MODEL X1 X2 X3 X4')

model = sys.argv[1]
X = online_prepare(sys.argv[2:])

# here we use the ONNX runtime (a library-agnostic runtime)
# however this could equally just be sklearn/tensorflow/etc.


sess = rt.InferenceSession(f'offline_db/models/{model}.onnx')

input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

prediction = sess.run([label_name], {
    input_name: X.reshape(1, 4)
})[0].tolist()


output = {
    'X': sys.argv[2:],
    'yhat': prediction,
    'model': model
}

now = int(time.time())
with open(f'online_db/predictions/{now}.json', 'w') as f:
    json.dump(output, f)

print(output)