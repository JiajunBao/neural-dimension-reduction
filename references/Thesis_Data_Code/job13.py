from main import main

# recheck paper results
kwargs_list1 = [
    {'input-size': 90000, 'test-size': 1000, 'hidden-dims': [5000, 1000, 200, 200, 200], 'output-dimension': 10,
     'lambda': 1, 'epochs': 20,
     'to_plot': True, 'learning_rate': 0.0005, 'momentum': .7, 'data': 'artificial3_10_200',
     'filename_prefix': 'thesis_1_1e-3_.7'}
    ]
main(kwargs_list1)

kwargs_list1 = [
    {'input-size': 90000, 'test-size': 1000, 'hidden-dims': [5000, 2000, 300, 300, 300], 'output-dimension': 10,
     'lambda': 1, 'epochs': 20,
     'to_plot': True, 'learning_rate': 0.0005, 'momentum': .7, 'data': 'artificial3_10_200',
     'filename_prefix': 'thesis_1_1e-3_.7'}
    ]
main(kwargs_list1)

kwargs_list1 = [
    {'input-size': 90000, 'test-size': 1000, 'hidden-dims': [5000, 3000, 400, 400, 400], 'output-dimension': 10,
     'lambda': 1, 'epochs': 20,
     'to_plot': True, 'learning_rate': 0.0005, 'momentum': .7, 'data': 'artificial3_10_200',
     'filename_prefix': 'thesis_1_1e-3_.7'}
    ]
main(kwargs_list1)

kwargs_list1 = [
    {'input-size': 90000, 'test-size': 1000, 'hidden-dims': [5000, 4000, 500, 500, 500], 'output-dimension': 10,
     'lambda': 1, 'epochs': 20,
     'to_plot': True, 'learning_rate': 0.0005, 'momentum': .7, 'data': 'artificial3_10_200',
     'filename_prefix': 'thesis_1_1e-3_.7'}
    ]
main(kwargs_list1)

kwargs_list1 = [
    {'input-size': 90000, 'test-size': 1000, 'hidden-dims': [5000, 5000, 600, 600, 600], 'output-dimension': 10,
     'lambda': 1, 'epochs': 20,
     'to_plot': True, 'learning_rate': 0.0005, 'momentum': .7, 'data': 'artificial3_10_200',
     'filename_prefix': 'thesis_1_1e-3_.7'}
    ]
main(kwargs_list1)