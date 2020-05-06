import seaborn as sns
import numpy as np

def f(x, a, b):
    return a*x + b

def plot_compare(df, features, target):
    X = df[features].values
    y = df[target].values

    sns.distplot(X, hist=False, label='All')
    
    for e in np.unique(y):
        sns.distplot(X[ y == e ], hist=False, label=e)