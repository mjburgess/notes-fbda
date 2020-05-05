# STEP 5.1 -- prepare data for prediction
# -- in general, data will need to be pre-prep'd before the model can accept it
# -- even online! eg., a user submits a form, but their details arent valid

import sys
import numpy as np


def offline_prepare(X):
    # ie., assume this is shared code across online/offline
    def normalize(X):
        # eg., apply sklearn transform; PCA, etc.
        return X.astype(np.float32)
        
    return normalize(X)
    

def online_prepare(features):
    # eg., high-d image -> low-d image
    # this will share code with offline/process.py ! 
    return offline_prepare(np.array(features))