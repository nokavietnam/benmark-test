import os
import pickle
import json
import numpy as np


def main():
    with open('data/[CIFAR-10]_data.p', 'rb') as f:
        data = pickle.load(f)
        print(data['200']['042324']['val_acc'])


if __name__ == '__main__':
    main()