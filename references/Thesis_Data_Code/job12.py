from main import main

# recheck paper results
kwargs_list1 = [
    {'input-size': 90000, 'test-size': 1000, 'hidden-dims': [500, 100, 20], 'output-dimension': 10,
     'lambda': 1, 'epochs': 20,
     'to_plot': True, 'learning_rate': 0.001, 'momentum': .7, 'data': 'artificial3_10_200',
     'filename_prefix': 'thesis_1_1e-3_.7'}
    ]
main(kwargs_list1)


kwargs_list1 = [
    {'input-size': 90000, 'test-size': 1000, 'hidden-dims': [500, 100, 20, 20], 'output-dimension': 10, 'lambda': 1,
     'epochs': 20,
     'to_plot': True, 'learning_rate': 0.001, 'momentum': .7, 'data': 'artificial3_10_200',
     'filename_prefix': 'thesis_2_1e-3_.7'}
    ]
main(kwargs_list1)

kwargs_list1 = [
    {'input-size': 90000, 'test-size': 1000, 'hidden-dims': [5000, 1000, 200, 200], 'output-dimension': 10, 'lambda': 1,
     'epochs': 20,
     'to_plot': True, 'learning_rate': 0.001, 'momentum': .7, 'data': 'artificial3_10_200',
     'filename_prefix': 'thesis_3_1e-3_.7'}
    ]
main(kwargs_list1)

kwargs_list1 = [
    {'input-size': 90000, 'test-size': 1000, 'hidden-dims': [5000, 1000, 200, 200, 200], 'output-dimension': 10, 'lambda': 1,
     'epochs': 20,
     'to_plot': True, 'learning_rate': 0.001, 'momentum': .7, 'data': 'artificial3_10_200',
     'filename_prefix': 'thesis_4_1e-3_.7'}
    ]
main(kwargs_list1)



# recheck paper results
kwargs_list1 = [
    {'input-size': 90000, 'test-size': 1000, 'hidden-dims': [500, 100, 20], 'output-dimension': 10,
     'lambda': 1, 'epochs': 20,
     'to_plot': True, 'learning_rate': 0.005, 'momentum': .7, 'data': 'artificial3_10_200',
     'filename_prefix': 'thesis_1_5e-3_.7'}
    ]
main(kwargs_list1)


kwargs_list1 = [
    {'input-size': 90000, 'test-size': 1000, 'hidden-dims': [500, 100, 20, 20], 'output-dimension': 10, 'lambda': 1,
     'epochs': 20,
     'to_plot': True, 'learning_rate': 0.005, 'momentum': .7, 'data': 'artificial3_10_200',
     'filename_prefix': 'thesis_2_5e-3_.7'}
    ]
main(kwargs_list1)

kwargs_list1 = [
    {'input-size': 90000, 'test-size': 1000, 'hidden-dims': [5000, 1000, 200, 200], 'output-dimension': 10, 'lambda': 1,
     'epochs': 20,
     'to_plot': True, 'learning_rate': 0.005, 'momentum': .7, 'data': 'artificial3_10_200',
     'filename_prefix': 'thesis_3_5e-3_.7'}
    ]
main(kwargs_list1)

kwargs_list1 = [
    {'input-size': 90000, 'test-size': 1000, 'hidden-dims': [5000, 1000, 200, 200, 200], 'output-dimension': 10, 'lambda': 1,
     'epochs': 20,
     'to_plot': True, 'learning_rate': 0.005, 'momentum': .7, 'data': 'artificial3_10_200',
     'filename_prefix': 'thesis_4_5e-3_.7'}
    ]
main(kwargs_list1)

