from models.vanilla.settings import Global
from models.vanilla.data import load_data, load_test_data_df
from models.vanilla.model import DeepNet, Net, device
from models.vanilla.utils import Environment


def run_model(data, test_data, args):
    ### INITIALIZE SETTINGS ###
    G = Global()
    G.set_debug_true() ####
    if data is None:
        return 0
    ### DEFINE MODEL ###
    n, m = data.shape
    out_dim = args['output-dimension']
    hidden_model = DeepNet
    hidden_sizes = args['hidden-dims']
    index = Net(hidden_model, m, out_dim, hidden_sizes).to(device)
    ### TRAIN MODEL ###
    index = index.fit(data, args)
    ### TEST MODEL ###
    results = index.test(data, test_data)
    if args['to_plot']:
        index.create_plot(data, filename='train')
        ## Plot test data ##
        index.create_plot(test_data, name='Test Data', filename='test')
    ## Save plots ##
    name = args['filename_prefix']
    G.save_figs(name)
    ## Release Space ##
    del G, index
    ## Return neighbors ##
    return results

def run_job_for_env(**kwargs):
    #####################
    ### DOWNLOAD DATA ###
    #####################
    if not 'data' in kwargs:
        print('Data not provided')
        return 0
    data = load_data(kwargs['data'], kwargs['input-size'])
    print('Data Loaded')
    ## Test data ##
    test_data = load_test_data_df(kwargs['data'], kwargs['input-size'], kwargs['test-size'])
    print('Test Data Loaded')
    ################
    ### RUN MAIN ###
    ################
    results = run_model(data, test_data, kwargs)
    return results

def main(kwargs_list):
    func = run_job_for_env
    schedule = []
    for kwargs in kwargs_list:
        job = (func, [], kwargs)
        schedule.append(job)
    ENV = Environment(schedule)
    ENV.run()
    ENV.save_states('states.dmp')

if __name__ == '__main__':
    kwargs_list1 = [{'input-size':90000,'test-size':1000,'hidden-dims':[5000, 1000, 200, 200],'output-dimension':10,'lambda':1,'epochs':20,
                    'to_plot':True,'learning_rate':0.01,'momentum':.9,'data':'artificial3_10_200','filename_prefix':'3-normal_data'}
                    ]
    main(kwargs_list1)

    kwargs_list1 = [{'input-size':90000,'test-size':1000,'hidden-dims':[5000, 1000, 200, 200, 200],'output-dimension':10,'lambda':1,'epochs':20,
                    'to_plot':True,'learning_rate':0.01,'momentum':.9,'data':'artificial3_10_200','filename_prefix':'4-normal_data'}
                    ]
    main(kwargs_list1)

    kwargs_list1 = [{'input-size':90000,'test-size':1000,'hidden-dims':[5000, 1000, 200, 200, 200, 200],'output-dimension':10,'lambda':1,'epochs':20,
                    'to_plot':True,'learning_rate':0.01,'momentum':.9,'data':'artificial3_10_200','filename_prefix':'5-normal_data'}
                    ]
    main(kwargs_list1)

