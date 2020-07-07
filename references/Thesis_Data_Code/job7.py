# learning rate test
from main import main

kwargs_list1 = [
    {'input-size': 90000, 'test-size': 1000, 'hidden-dims': [200, 100, 50, 25, 20], 'output-dimension': 10, 'lambda': 1,
     'epochs': 40,
     'to_plot': True, 'learning_rate': 5e-7, 'momentum': .9, 'data': 'artificial3_10_200',
     'filename_prefix': '200-100-50-25-20'}
    ]
main(kwargs_list1)


