from main import main

kwargs_list1 = [
    {'input-size': 90000, 'test-size': 1000, 'hidden-dims': [512, 512, 128, 128, 128, 64, 64, 64, 32, 32, 20, 20], 'output-dimension': 10,
     'lambda': 1, 'epochs': 20,
     'to_plot': True, 'learning_rate': 0.01, 'momentum': .9, 'data': 'artificial3_10_200',
     'filename_prefix': '2*512_3*128_3*64_2*32_2*20'}
    ]
main(kwargs_list1)

kwargs_list1 = [
    {'input-size': 90000, 'test-size': 1000, 'hidden-dims': [512, 512, 64, 64, 64, 32, 32, 20, 20], 'output-dimension': 10,
     'lambda': 1, 'epochs': 20,
     'to_plot': True, 'learning_rate': 0.01, 'momentum': .9, 'data': 'artificial3_10_200',
     'filename_prefix': '2*512_3*64_2*32_2*20s'}
    ]
main(kwargs_list1)

