import os
import pickle
import numpy as np
import copy
import random
import collections
from nasbench import api
from nas_201_api import NASBench201API as API201
from exp import get_toy_data
from models import try_different_method, model, method

# basic matrix for nas_bench 201
BASIC_MATRIX = [[0, 1, 1, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0]]

MAX_NUMBER = 15625
NULL = 'null'
CONV1X1 = 'nor_conv_1x1'
CONV3X3 = 'nor_conv_3x3'
AP3X3 = 'avg_pool_3x3'


def delete_useless_node(ops):
    # delete the skip connections nodes and the none nodes
    # output the pruned metrics
    # start to change matrix
    matrix = copy.deepcopy(BASIC_MATRIX)
    for i, op in enumerate(ops, start=1):
        m = []
        n = []

        if op == 'skip_connect':
            for m_index in range(8):
                ele = matrix[m_index][i]
                if ele == 1:
                    # set element to 0
                    matrix[m_index][i] = 0
                    m.append(m_index)

            for n_index in range(8):
                ele = matrix[i][n_index]
                if ele == 1:
                    # set element to 0
                    matrix[i][n_index] = 0
                    n.append(n_index)

            for m_index in m:
                for n_index in n:
                    matrix[m_index][n_index] = 1

        elif op == 'none':
            for m_index in range(8):
                matrix[m_index][i] = 0
            for n_index in range(8):
                matrix[i][n_index] = 0

    ops_copy = copy.deepcopy(ops)
    ops_copy.insert(0, 'input')
    ops_copy.append('output')

    # start pruning
    model_spec = api.ModelSpec(matrix=matrix, ops=ops_copy)
    return model_spec.matrix, model_spec.ops


def save_arch_str2op_list(save_arch_str):
    op_list = []
    save_arch_str_list = API201.str2lists(save_arch_str)
    op_list.append(save_arch_str_list[0][0][0])
    op_list.append(save_arch_str_list[1][0][0])
    op_list.append(save_arch_str_list[1][1][0])
    op_list.append(save_arch_str_list[2][0][0])
    op_list.append(save_arch_str_list[2][1][0])
    op_list.append(save_arch_str_list[2][2][0])
    return op_list


def padding_zeros(matrix, op_list):
    assert len(op_list) == len(matrix)
    padding_matrix = matrix
    len_operations = len(op_list)
    if not len_operations == 8:
        for j in range(len_operations, 8):
            op_list.insert(j - 1, NULL)
        adjecent_matrix = copy.deepcopy(matrix)
        padding_matrix = np.insert(adjecent_matrix, len_operations - 1, np.zeros([8 - len_operations, len_operations]),
                                   axis=0)
        padding_matrix = np.insert(padding_matrix, [len_operations - 1], np.zeros([8, 8 - len_operations]), axis=1)

    return padding_matrix, op_list


def operation2integers(op_list):
    dict_oper2int = {NULL: 0, CONV1X1: 1, CONV3X3: 2, AP3X3: 3}
    module_integers = np.array([dict_oper2int[x] for x in op_list[1: -1]])
    return module_integers


def get_metrics_from_index_list(index_list, ordered_dic, metrics_num, dataset, upper_limit_time=100000000000):
    metrics = {}
    times = 0
    total_time = 0
    for index in index_list:
        if times == metrics_num:
            break
        final_test_acc = ordered_dic[index][dataset]
        epoch12_time = ordered_dic[index]['cifar10_all_time']
        total_time += epoch12_time
        if total_time > upper_limit_time:
            break
        op_list = save_arch_str2op_list(ordered_dic[index]['arch_str'])
        pruned_matrix, pruned_op = delete_useless_node(op_list)
        if pruned_matrix is None:
            continue
        else:
            times += 1
        padding_matrix, padding_op = padding_zeros(pruned_matrix, pruned_op)
        op_integers = operation2integers(padding_op)

        metrics[index] = {'final_training_time': epoch12_time, 'final_test_accuracy': final_test_acc / 100}
        metrics[index]['fixed_metrics'] = {'module_adjacency': padding_matrix, 'module_integers': op_integers,
                                           'trainable_parameters': -1}
    return metrics

def N_K(target, predict, k):
    out_1 = np.argsort(-target)
    out_2 = np.argsort(-predict)
    best_k = out_2[0:k]
    rank = []
    for i in best_k:
        a = np.argwhere(out_1 == i)[0][0] + 1
        rank.append(a)
    return min(rank)

def calculate_MSE(x, y):
    # input two list, x: predict, y: ground truth
    # output MSE
    mse_list = np.array([(element_x - element_y) ** 2 for element_x, element_y in zip(x, y)])
    mse = np.mean(mse_list)
    return mse

def experiment_on_201(train_num, test_num, dataset, create_more_metrics, integers2one_hot):
    expand = 1.1
    expand_train_num = int(train_num * expand)
    expand_test_num = int(test_num * expand)

    print('Loading original nas bench architecture and acc.')
    tidy_file = r'data/tidy_nas_bench_201.pkl'
    if not os.path.exists(tidy_file):
        nasbench201 = API201(r'data/NAS-Bench-201-v1_1-096897.pth')
        ordered_dic = collections.OrderedDict()
        for index in range(len(nasbench201.evaluated_indexes)):
            info = nasbench201.query_meta_info_by_index(index, '12')
            arch_str = info.arch_str
            cifar10_valid = info.get_metrics('cifar10-valid', 'x-valid')['accuracy']
            cifar10_all_time = info.get_metrics('cifar10-valid', 'x-valid')['all_time']

            info = nasbench201.query_meta_info_by_index(index, '200')
            cifar10 = info.get_metrics('cifar10', 'ori-test')['accuracy']
            cifar10_valid200 = info.get_metrics('cifar10-valid', 'x-valid')['accuracy']
            index_info = {'arch_str': arch_str, 'cifar10': cifar10, 'cifar10_valid': cifar10_valid,
                          'cifar10_all_time': cifar10_all_time, 'cifar10_valid200': cifar10_valid200}
            ordered_dic[index] = index_info

        with open(tidy_file, 'wb') as file:
            pickle.dump(ordered_dic, file)
    else:
        with open(tidy_file, 'rb') as file:
            ordered_dic = pickle.load(file)

    print('Selecting train and test index.')
    train_index_save_path = r'pkl/fixed_train_data_201_{}.pkl'.format(train_num)
    if os.path.exists(train_index_save_path):
        with open(train_index_save_path, 'rb') as file:
            train_list = pickle.load(file)
    else:
        sample_list = list(range(0, MAX_NUMBER))
        train_list = random.sample(sample_list, expand_train_num)
        train_list.sort()
        with open(train_index_save_path, 'wb') as file:
            pickle.dump(train_list, file)

    list_remove_train = list(range(0, MAX_NUMBER))
    for i in range(expand_train_num):
        list_remove_train.remove(train_list[i])
    if expand_test_num >= len(list_remove_train):
        expand_test_num = len(list_remove_train)
    test_list = random.sample(list_remove_train, expand_test_num)
    test_list.sort()

    print('Generating metrics like nas-bench-101.')
    train_metrics = get_metrics_from_index_list(train_list, ordered_dic, train_num, dataset, upper_limit_time=10000000)
    test_metrics = get_metrics_from_index_list(test_list, ordered_dic, test_num, dataset, upper_limit_time=10000000)

    print('----------------------train---------------------')
    X, y, _ = get_toy_data(train_metrics, create_more_metrics=create_more_metrics, select_upper_tri=False,
                           additional_metrics=False,
                           integers2one_hot=integers2one_hot)
    print('----------------------test----------------------')
    testX, testy, _ = get_toy_data(test_metrics, create_more_metrics=False, select_upper_tri=False,
                                   additional_metrics=False,
                                   integers2one_hot=integers2one_hot)

    for i in range(0,len(model)):
        try_different_method(X, y, testX, testy, model[i], method[i], show_fig=False)

def experiment_on_E2EPP_201(train_num, test_num, dataset, create_more_metrics, integers2one_hot):
    expand = 1.1
    expand_train_num = int(train_num * expand)
    expand_test_num = int(test_num * expand)

    print('Loading original nas bench architecture and acc.')
    tidy_file = r'path/tidy_nas_bench_201.pkl'
    if not os.path.exists(tidy_file):
        nasbench201 = API201(r'path/NAS-Bench-201-v1_1-096897.pth')
        ordered_dic = collections.OrderedDict()
        for index in range(len(nasbench201.evaluated_indexes)):
            info = nasbench201.query_meta_info_by_index(index, '12')
            arch_str = info.arch_str
            cifar10_valid = info.get_metrics('cifar10-valid', 'x-valid')['accuracy']
            cifar10_all_time = info.get_metrics('cifar10-valid', 'x-valid')['all_time']

            info = nasbench201.query_meta_info_by_index(index, '200')
            cifar10 = info.get_metrics('cifar10', 'ori-test')['accuracy']
            cifar10_valid200 = info.get_metrics('cifar10-valid', 'x-valid')['accuracy']
            index_info = {'arch_str': arch_str, 'cifar10': cifar10, 'cifar10_valid': cifar10_valid,
                          'cifar10_all_time': cifar10_all_time, 'cifar10_valid200': cifar10_valid200}
            ordered_dic[index] = index_info

        with open(tidy_file, 'wb') as file:
            pickle.dump(ordered_dic, file)
    else:
        with open(tidy_file, 'rb') as file:
            ordered_dic = pickle.load(file)

    print('Selecting train and test index.')
    train_index_save_path = r'pkl/fixed_train_data_201_{}.pkl'.format(train_num)
    if os.path.exists(train_index_save_path):
        with open(train_index_save_path, 'rb') as file:
            train_list = pickle.load(file)
    else:
        sample_list = list(range(0, MAX_NUMBER))
        train_list = random.sample(sample_list, expand_train_num)
        train_list.sort()
        with open(train_index_save_path, 'wb') as file:
            pickle.dump(train_list, file)

    list_remove_train = list(range(0, MAX_NUMBER))
    for i in range(expand_train_num):
        list_remove_train.remove(train_list[i])
    if expand_test_num >= len(list_remove_train):
        expand_test_num = len(list_remove_train)
    test_list = random.sample(list_remove_train, expand_test_num)
    test_list.sort()

    print('Generating metrics like nas-bench-101.')
    train_metrics = get_metrics_from_index_list(train_list, ordered_dic, train_num, dataset, upper_limit_time=10000000)
    test_metrics = get_metrics_from_index_list(test_list, ordered_dic, test_num, dataset, upper_limit_time=10000000)

    print('----------------------train---------------------')
    X, y, _ = get_toy_data(train_metrics, create_more_metrics=create_more_metrics, select_upper_tri=False,
                           additional_metrics=False,
                           integers2one_hot=integers2one_hot)
    print('----------------------test----------------------')
    testX, testy, _ = get_toy_data(test_metrics, create_more_metrics=False, select_upper_tri=False,
                                   additional_metrics=False,
                                   integers2one_hot=integers2one_hot)

    from sklearn.metrics import r2_score
    from scipy.stats import kendalltau
    e2epp_tree, e2epp_features = train_e2epp(X, y)
    predictor = e2epp_tree, e2epp_features
    result = test_e2epp(testX, predictor[0], predictor[1])
    result = list(result)
    score = r2_score(testy, result)
    result_arg = np.argsort(result)
    y_test_arg = np.argsort(testy)
    result_rank = np.zeros(len(y_test_arg))
    y_test_rank = np.zeros(len(y_test_arg))
    for i in range(len(y_test_arg)):
        result_rank[result_arg[i]] = i
        y_test_rank[y_test_arg[i]] = i
    KTau, _ = kendalltau(result_rank, y_test_rank)
    print('method: {:}, KTau: {:}, MSE: {:}, R2score: {:}'.format("E2EPP", KTau, calculate_MSE(testy, result), score))
    print('N@5: {:}, N@10: {:}'.format(N_K(np.array(testy), np.array(result), 5),
                                       N_K(np.array(testy), np.array(result), 10)))
    print('--------------------try-end---------------------\n')



def main(train_num, test_num):
    # datasets: cifar10, cifar100, ImageNet
    dataset = 'cifar10'
    create_more_metrics = True
    integers2one_hot = True
    print("create_more_metric:{:}".format(create_more_metrics))
    print("integers2one_hot:{:}".format(integers2one_hot))
    experiment_on_201(train_num, test_num, dataset, create_more_metrics, integers2one_hot)
    # experiment_on_E2EPP_201(train_num, test_num, dataset, create_more_metrics, integers2one_hot)


if __name__ == '__main__':
    valid_train_splits = [0.005, 0.01, 0.05, 0.1, 0.15, 0.2]
    num_all = 15625

    for i in valid_train_splits:
        num_train = int(i * num_all)
        num_test = num_all - num_train
        print("--------%f|%d train-----%d test------------"%(i,num_train,num_test))
        main(num_train, num_test)
            