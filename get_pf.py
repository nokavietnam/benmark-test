import numpy as np
import json
import os
import pickle as p

list_ops = np.array(['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3'])


def decode_arch(arch):
    return ''.join(
        str(np.where(list_ops == item.split('~')[0])[0][0]) for item in filter(None, arch.split('|')) if item != '+')


def get_map_ids(meta):
    return {m["nb201-string"]: k for k, m in meta["ids"].items()}


def get_iso_morph_ids(meta):
    return [i for i, d in meta["ids"].items() if d["isomorph"] == i]

def find_the_better(x, y):
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(y, list):
        y = np.array(y)
    sub_ = x - y
    x_better = np.all(sub_ <= 0)
    y_better = np.all(sub_ >= 0)
    if x_better == y_better:  # True - True
        return -1
    if y_better:  # False - True
        return 1
    return 0  # True - False


def get_front_0(F):
    l = len(F)
    r = np.zeros(l, dtype=np.int8)
    for i in range(l):
        if r[i] == 0:
            for j in range(i + 1, l):
                better_sol = find_the_better(F[i], F[j])
                if better_sol == 0:
                    r[j] += 1
                elif better_sol == 1:
                    r[i] += 1
                    break
    return r == 0


if __name__ == '__main__':
    data = p.load(open('data.p', 'rb'))
    print(data)

    nb201_database = p.load(open('data/data/NASBench201/[CIFAR-10]_data.p', 'rb'))

    F = []
    for arch, info in nb201_database['200'].items():
        flops = info['FLOPs']
        pgd_acc = data[arch]
        F.append([flops, -pgd_acc])
    F = np.array(F)
    F = np.unique(F, axis=0)
    idx_pof = get_front_0(F)
    pof = F[idx_pof]
    p.dump(pof, open('pof_pgd-acc-0.5_flops.p', 'wb'))
    import matplotlib.pyplot as plt

    plt.scatter(pof[:, 0], pof[:, 1])
    plt.xlabel('FLOPs')
    plt.ylabel('-PGD_ACC-0.5')
    plt.savefig('PGD-ACC-0.5_FLOPS.jpg')
    plt.show()

    print(pof)
