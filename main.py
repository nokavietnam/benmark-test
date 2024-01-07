import numpy as np
import matplotlib.pyplot as plt
import json
import os
import pickle


list_ops = np.array(['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3'])


def decode_arch(arch):
    return ''.join(
        str(np.where(list_ops == item.split('~')[0])[0][0]) for item in filter(None, arch.split('|')) if item != '+')


def get_map_ids(meta):
    return {m["nb201-string"]: k for k, m in meta["ids"].items()}


def get_iso_morph_ids(meta):
    return [i for i, d in meta["ids"].items() if d["isomorph"] == i]


def main(path_file):
    dict_ids = dict()
    with open(os.path.join(path_file, "meta.json")) as f:
        meta = json.load(f)
        dict_ids = get_map_ids(meta)

    dict_temp = dict()

    for item in dict_ids:
        dict_temp.update({decode_arch(item): dict_ids[item]})

    for item in dict_temp:
       print(dict_temp[item])


    data_value = dict()
    pgd_acc = []
    d = 'cifar10'
    k = 'pgd@Linf'
    #k = 'clean'
    m = 'accuracy'
    file = os.path.join('data/robustness-data/cifar10', f"{k}_{m}.json")
    with open(file, "r") as f:
        r = json.load(f)
        ja_acc = r[d][k][m]
        for item in ja_acc:
            pgd_acc.append(ja_acc[item][1])
            data_value.update({item: ja_acc[item][1]})

    #for item in data_value:
    #    print(item)

    result = {}

    for key, value in dict_temp.items():
        try:
            acc = data_value[value]
        except:
            iso_idx = meta['ids'][value]['isomorph']
            acc = data_value[iso_idx]
        result[key] = acc
    # print(result_new)

    #for item in result:
        # print(f"{item}: {result[item]}")
    # print(f"test {result.__sizeof__()}")
    # print(result)

    pickle.dump(result, open("data_clean.p", "wb"))
    #print(result)

    # data = pickle.load(open('data.p', 'rb'))
    # print(data)

    #print(result.__sizeof__())
    #print(dict_temp.__sizeof__())
    #print(data_value.__sizeof__())


if __name__ == '__main__':
    path = "data/robustness-data"
    main(path)
