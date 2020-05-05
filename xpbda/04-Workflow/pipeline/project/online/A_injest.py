# STEP 1: injest data

from time import sleep
import numpy as np 
import random 

from sklearn import datasets
iris = datasets.load_iris()
data = np.c_[iris['data'], iris['target']]

# simulate, eg. a streaming API which injests data for training over time
def injest():
    for observation in data:
        missing = random.randint(0,100) >= 95
        if missing:
            observation[4] = np.nan

        print("Observed a y=", observation[4])
        sleep(0.1) 
        yield observation
    

## eg., this could be parquet or some big data db
## esp., a relational db
np.save('online_db/injest-raw', np.array(list(injest())))