import matplotlib.pyplot as plt
from tqdm import tqdm

class Global:
    ########################
    ### GLOBAL VARIABLES ###
    ########################
    debug = False
    num_of_plots = 0
    DISABLE_TSNE = True # True for omittibg tSNE
    IN_SAMPLE_TESTING = True # default is False
    eps = .1
    fig_names = {}

    def __init__(self):
        self.update()

    def update(self):
        #################
        ### VARIABLES ###
        #################
        self.debug = Global.debug
        #################
        ### FUNCTIONS ###
        #################
        self.tqdm = lambda iter: tqdm(iter) if Global.debug else iter
        self.out = lambda s='\n': print(s) if Global.debug else None

    def set_debug_true(self):
        Global.debug = True
        self.update()

    @classmethod
    def increase_plots(cls):
        cls.num_of_plots = cls.num_of_plots + 1 

    @classmethod
    def show(cls):
        if cls.num_of_plots > 0:
            plt.show()

    @classmethod
    def save_plot_name(cls, key, name):
        cls.fig_names[key] = name

    @classmethod
    def save_figs(cls, name):
        for key in cls.fig_names.keys():
            new_name = '{}_{}.png'.format(name, cls.fig_names[key])
            plt.figure(key).savefig(new_name)