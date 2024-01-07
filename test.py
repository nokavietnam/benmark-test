import os
import pickle

import numpy as np



if __name__ == '__main__':
    data = pickle.load(open("pareto/data_fgsm@Linf.p", "rb"))
    print(data['042324'])