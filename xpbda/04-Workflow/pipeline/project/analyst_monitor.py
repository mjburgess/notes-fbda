from json import load
import glob

def top3(folder):
    return sorted(glob.glob(f'{folder}/*.json'))[:3]

records = { r: [load(open(f"{file}")) for file in top3(r)]
    for r in ['offline_db/validations', 'offline_db/models', 'online_db/predictions'] 
}

# this should be graphical UI tracking measures of runs/time

print('Latest Predictions:')
print([(r['X'], r['yhat']) for r in records['online_db/predictions']])

print('Latest Models:')
print([r['eval']['score_test'] for r in records['offline_db/models']])

print('Latest Validations:')
print([r['eval']['score_val_mean'] for r in records['offline_db/validations']])
