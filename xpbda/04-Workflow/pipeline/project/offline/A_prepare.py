# STEP 2: prepare data
import shutil

import numpy as np

from sklearn.model_selection import train_test_split

from A_helper import *

# load data in an unprocessed format
shutil.copyfile(
    'online_db/injest-raw.npy', 
    'offline_db/features-db.npy'
)

raw = np.load('offline_db/features-db.npy', allow_pickle=True)

# X is every row and 4 col of the live data
X_raw = normalize(raw[:, 0:4])
y_raw = raw[:, 4]

#preprocess -- here simply, drop nan
keep = ~np.isnan(y_raw)

X = X_raw[keep]
y = y_raw[keep]

# process for validation/training/testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

print(f'Kept: {100 * sum(keep)/len(y_raw): .2f} %')


#TODO: data quality checks

print('Sample post X:', X[5])
print('Sample post y:', y[5])

print('Sample X_train:', X_train[5])
print('Sample y_train:', y_train[5])

print('Sample X_test:', X_train[5])
print('Sample y_test:', y_train[5])

# save processing output
# numpy save/load used as illustration of data file format 

np.save('offline_db/train-feature', X_train)
np.save('offline_db/train-target', y_train)

np.save('offline_db/test-feature', X_test)
np.save('offline_db/test-target', y_test)