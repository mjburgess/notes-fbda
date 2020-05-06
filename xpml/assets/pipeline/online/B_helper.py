# shared preprocessing library
# deployed systems will need to transform alike offline training systems

import numpy as np

def normalize(X):
    # eg., apply sklearn transform; PCA, etc.
    return X.astype(np.float32)
    
