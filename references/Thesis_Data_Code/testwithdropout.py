from main import main


kwargs_list1 = [
    {'input-size': 90000, 'test-size': 1000, 'hidden-dims': [500, 100, 20, 20, 20], 'output-dimension': 10,
     'lambda': 1, 'epochs': 20,
     'to_plot': True, 'learning_rate': 0.01, 'momentum': .9, 'data': 'artificial_with_drop_out',
     'filename_prefix': 'withdropout'}
    ]
main(kwargs_list1)

