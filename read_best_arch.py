import os
import json
import pickle


def main():
    with open('data/best_architecture_each_gen.p', 'rb') as f:
        nGens_history, best_arch_history = pickle.load(f)
        print(nGens_history)
        for arch in best_arch_history:
            print(arch)


if __name__ == '__main__':
    main()