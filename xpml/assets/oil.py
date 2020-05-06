import pickle

model = pickle.load(open('oil_model.pkl', 'rb'))

x_d = float(input('Dense? '))
x_b = float(input('Boil? '))

yhat = model.predict([
 [x_d, x_b]
])[0]

class_names = [
    'light',
    'arabian light',
    'heavy',
    'superlight'
]

print(class_names[yhat])