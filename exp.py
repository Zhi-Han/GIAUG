import get_data_from_101 as get_data
import Create_more_metrics
import copy
import os
import pickle
from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn import neighbors
from sklearn import ensemble
from models import method, calculate_MSE
from sklearn.metrics import r2_score
from scipy.stats import kendalltau
from tensorflow.python.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt



# MAX full training time
MAX_FTT = 5521.803059895833
# MAX trainable parameters
MAX_TP = 49979274


def Random_forestAndmore_test(method, x_train, y_train, x_test, y_test, more_x_test, more_y_test, parameter_scale,
                              step=10):
    print('First run without more test data')
    no_more_test_KTau_list, no_more_test_MSE_list = Ablation_study(method, x_train, y_train, x_test, y_test,
                                                                   parameter_scale, step, more_test_data=False,
                                                                   show_figure=False)

    print('Second run with more test data')
    more_test_KTau_list, more_test_MSE_list = Ablation_study(method, x_train, y_train, more_x_test, more_y_test,
                                                             parameter_scale, step, more_test_data=True,
                                                             show_figure=False)

    # save KTau and MSE list
    if not os.path.isdir('pkl'):
        os.makedirs('pkl')
    save_path = r'pkl\ktau_and_mse_list.pkl'
    save_dic = {'no_more_test_KTau_list': no_more_test_KTau_list, 'no_more_test_MSE_list': no_more_test_MSE_list,
                'more_test_KTau_list': more_test_KTau_list, 'more_test_MSE_list': more_test_MSE_list}
    with open(save_path, 'wb') as file:
        pickle.dump(save_dic, file)
        print('Save KTau and MSE list successfully!')

    # Because running the above code is a time-consuming process. To speed up the process,
    # we recommend using the function make_plot_for_KTau_and_MSE with the saving data.
    plt.figure(figsize=(13, 5))
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(parameter_scale[0], parameter_scale[-1], step), no_more_test_KTau_list[4:], "hotpink",
             linestyle=':', marker='^', label="original test")
    plt.plot(np.arange(parameter_scale[0], parameter_scale[-1], step), more_test_KTau_list[4:], "skyblue",
             linestyle=':', marker='D', label="committee prediction")
    plt.xlabel("number of estimators")
    plt.ylabel("KTau")
    plt.legend(loc="best")
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(parameter_scale[0], parameter_scale[-1], step), no_more_test_MSE_list[4:], "hotpink",
             linestyle=':', marker='^', label="original test")
    plt.plot(np.arange(parameter_scale[0], parameter_scale[-1], step), more_test_MSE_list[4:], "skyblue",
             linestyle=':', marker='D', label="committee prediction")
    plt.xlabel("number of estimators")
    plt.ylabel("MSE")
    plt.legend(loc="best")
    plt.show()


def make_plot_for_KTau_and_MSE(KTau_list, MSE_list):
    # show the results of estimators = {50, 60, ..., 300}
    parameter_scale = [10, 310]
    # don't change the step
    step = 10
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(parameter_scale[0], parameter_scale[-1], step), KTau_list, "hotpink", linestyle=':',
             marker='^')
    plt.xlabel("number of estimators")
    plt.ylabel("KTau")

    plt.subplot(2, 1, 2)
    plt.plot(np.arange(parameter_scale[0], parameter_scale[-1], step), MSE_list, "skyblue", linestyle=':',
             marker='D')
    plt.xlabel("number of estimators")
    plt.ylabel("MSE")
    plt.show()


def Ablation_study(method, x_train, y_train, x_test, y_test, parameter_scale, step=1, more_test_data=False,
                   show_figure=True, upper_tri_num=120):
    score_list, KTau_list, MSE_list = [], [], []
    y_test_copy = copy.deepcopy(y_test)
    print(f'Ablation study, method: {method}')
    for parameter in range(parameter_scale[0], parameter_scale[-1], step):
        if method == 'random_forest':
            Model = ensemble.RandomForestRegressor(n_estimators=parameter)
        elif method == 'knn':
            Model = neighbors.KNeighborsRegressor(n_neighbors=parameter)
        elif method == 'GBRT':
            Model = ensemble.GradientBoostingRegressor(n_estimators=parameter)
        elif method == 'Bagging':
            Model = ensemble.BaggingRegressor(n_estimators=parameter)
        else:
            raise ValueError
        Model.fit(x_train, y_train)
        result = Model.predict(x_test)
        if more_test_data:
            if isinstance(upper_tri_num, list):
                print('Select upper tri matrix!!!')
                mean_result = []
                first_ground_truth = []
                last_num = 0
                for num in upper_tri_num:
                    mean_result.extend([np.mean(result[last_num:last_num + num])])
                    first_ground_truth.extend([y_test_copy[last_num]])
                    last_num = last_num + num
            else:
                # the number is fixed 120 now, use the mean value
                # that is one metric can produce 5! = 120 same architecture representation
                mean_result = []
                first_ground_truth = []
                for i in range(len(x_test) // 120):
                    mean_result.extend([np.mean(result[i * 120:(i + 1) * 120])])
                    first_ground_truth.extend([y_test_copy[i * 120]])
                    # because the ground truth in [i*120:(i+1)*120] are all [i*120]
                result = mean_result
                y_test = first_ground_truth
        result = list(result)
        score = r2_score(y_test, result)
        result_arg = np.argsort(result)
        y_test_arg = np.argsort(y_test)
        result_rank = np.zeros(len(y_test_arg))
        y_test_rank = np.zeros(len(y_test_arg))
        for i in range(len(y_test_arg)):
            result_rank[result_arg[i]] = i
            y_test_rank[y_test_arg[i]] = i
        KTau, _ = kendalltau(result_rank, y_test_rank)
        MSE = calculate_MSE(y_test, result)
        print(
            'parameter: {:}, KTau: {:}, MSE: {:}, R2score: {:}'.format(parameter, KTau, MSE, score))
        score_list.append([score])
        MSE_list.append([MSE])
        KTau_list.append([KTau])

    # save KTau and MSE list
    if not os.path.isdir('pkl'):
        os.makedirs('pkl')
    save_path = r'pkl\ktau_and_mse_list.pkl'
    save_dic = {'KTau_list': KTau_list, 'MSE_list': MSE_list}
    with open(save_path, 'wb') as file:
        pickle.dump(save_dic, file)
        print('Save KTau and MSE list successfully!')

    if show_figure:
        plt.subplot(2, 1, 1)
        plt.plot(np.arange(parameter_scale[0], parameter_scale[-1], step), KTau_list, "hotpink", linestyle=':',
                 marker='^')
        plt.xlabel("number of estimators")
        plt.ylabel("KTau")

        plt.subplot(2, 1, 2)
        plt.plot(np.arange(parameter_scale[0], parameter_scale[-1], step), MSE_list, "skyblue", linestyle=':',
                 marker='D')
        plt.xlabel("number of estimators")
        plt.ylabel("MSE")

        plt.title(f"Different number of estimators in {method}")
        plt.show()



# return: 1. X: a linear array: flattened adjacent matrix + integer operations
#         2. y: accuracy
def get_toy_data(important_metrics, create_more_metrics = True, select_upper_tri=False, max_creation=-1,integers2one_hot=True, additional_metrics=False, normalization=True):
    X = []
    y = []
    more_metrics_num = []
    for index in important_metrics:
        fixed_metrics = important_metrics[index]['fixed_metrics']
        adjacent_matrix = fixed_metrics['module_adjacency']
        module_integers = fixed_metrics['module_integers']
        trainable_parameters = fixed_metrics['trainable_parameters']
        final_training_time = important_metrics[index]['final_training_time']
        accuracy = important_metrics[index]['final_test_accuracy']
        if create_more_metrics:
            more_metrics = Create_more_metrics.create_new_metrics(adjacent_matrix, module_integers, select_upper_tri,
                                                                  max_creation, InOut=False)
            num_new_metrics = len(more_metrics)
            more_metrics_num.extend([num_new_metrics])
            for same_metric in more_metrics:
                adjacent_matrix, module_integers = same_metric['module_adjacency'], same_metric['module_integers']
                adjacent_matrix = get_data.delete_margin(adjacent_matrix)
                flattened_adjacent = adjacent_matrix.flatten()
                input_metrics = []
                input_metrics.extend(flattened_adjacent)
                if integers2one_hot:
                    module_integers = to_categorical(module_integers, 4)
                    module_integers = module_integers.flatten()
                input_metrics.extend(module_integers)
                if additional_metrics:
                    norm_trainable_parameters = trainable_parameters / MAX_TP
                    norm_final_training_time = final_training_time / MAX_FTT
                    if normalization:
                        input_metrics.extend([norm_trainable_parameters])
                        input_metrics.extend([norm_final_training_time])
                    else:
                        input_metrics.extend([trainable_parameters])
                        input_metrics.extend([final_training_time])
                X.append(input_metrics)
                y.append(accuracy)
        else:
            adjacent_matrix = get_data.delete_margin(adjacent_matrix)
            flattened_adjacent = adjacent_matrix.flatten()
            input_metrics = []
            input_metrics.extend(flattened_adjacent)
            if integers2one_hot:
                module_integers = to_categorical(module_integers, 4)
                module_integers = module_integers.flatten()
            input_metrics.extend(module_integers)
            if additional_metrics:
                if normalization:
                    trainable_parameters = trainable_parameters / MAX_TP
                    final_training_time = final_training_time / MAX_FTT
                input_metrics.extend([trainable_parameters])
                input_metrics.extend([final_training_time])
            X.append(input_metrics)
            y.append(accuracy)

    assert len(X) == len(y)
    print('Input {:} metrics, obtain {:} metrics'.format(len(important_metrics), len(X)))
    return X, y, more_metrics_num

def get_GCN_data(important_metrics, create_more_metrics=True, select_upper_tri=False, max_creation=-1):
    X = []
    y = []
    more_metrics_num = []
    for index in important_metrics:
        fixed_metrics = important_metrics[index]['fixed_metrics']
        adjacent_matrix = fixed_metrics['module_adjacency']
        module_integers = fixed_metrics['module_integers']
        zero = 0
        mi = list(module_integers)
        while zero in mi:
            mi.remove(zero)
        num_vertices = len(mi)
        trainable_parameters = fixed_metrics['trainable_parameters']
        final_training_time = important_metrics[index]['final_training_time']
        accuracy = important_metrics[index]['final_test_accuracy']
        if create_more_metrics:
            more_metrics = Create_more_metrics.create_new_metrics(adjacent_matrix, module_integers, select_upper_tri,
                                                                  max_creation, InOut=False)
            num_new_metrics = len(more_metrics)
            more_metrics_num.extend([num_new_metrics])
            for same_metric in more_metrics:
                adjacent_matrix, module_integers = same_metric['module_adjacency'], same_metric['module_integers']
                adjacent_matrix = get_data.delete_margin(adjacent_matrix)
                input_metrics = {"num_vertices": num_vertices, "adjacency": adjacent_matrix,"operations": module_integers}
                X.append(input_metrics)
                y.append(accuracy)
        else:
            adjacent_matrix = get_data.delete_margin(adjacent_matrix)
            input_metrics = {"num_vertices": num_vertices, "adjacency": adjacent_matrix, "operations": module_integers}
            X.append(input_metrics)
            y.append(accuracy)

    assert len(X) == len(y)
    print('Input {:} metrics, obtain {:} metrics'.format(len(important_metrics), len(X)))
    return X, y, more_metrics_num

def get_upper_triangular_data(important_metrics, integers2one_hot=True, double_upper=True, additional_metrics=True, normalization=True):
    # upper triangular and additional metrics (including trainable parameters and final training time)
    # double_upper denotes flatting the upper triangular matrix into a one-dimensional vector based the axis = 0 and 1
    X = []
    y = []
    for index in important_metrics:
        fixed_metrics = important_metrics[index]['fixed_metrics']
        adjacent_matrix = fixed_metrics['module_adjacency']
        module_integers = fixed_metrics['module_integers']
        if integers2one_hot:
            module_integers = to_categorical(module_integers, 4, dtype='int8')
            module_integers = module_integers.flatten()

        trainable_parameters = fixed_metrics['trainable_parameters']
        final_training_time = important_metrics[index]['final_training_time']
        adjacent_matrix = get_data.delete_margin(adjacent_matrix)
        array_adjacent_matrix = np.array(adjacent_matrix)
        flattened_adjacent = []
        matrix_size = len(adjacent_matrix)
        # get upper triangular data in matrix and flat it
        # Noting that this doesn't contains the elements as main diagonal
        for i in range(matrix_size):
            flattened_adjacent.extend(adjacent_matrix[i][i:])

        if double_upper:
            # add information from column
            for i in range(matrix_size):
                extend_array = array_adjacent_matrix[:i + 1, i]
                flattened_adjacent.extend(extend_array.tolist())
        input_metrics = []
        input_metrics.extend(flattened_adjacent)
        input_metrics.extend(module_integers)
        if additional_metrics:
            if normalization:
                trainable_parameters = trainable_parameters / MAX_TP
                final_training_time = final_training_time / MAX_FTT
            input_metrics.extend([trainable_parameters])
            input_metrics.extend([final_training_time])
        accuracy = important_metrics[index]['final_test_accuracy']
        X.append(input_metrics)
        y.append(accuracy)

    return X, y


def get_toy_metrics(num, type='train', train_num=2000):
    index = get_data.get_data_index_from_101(num, type=type, train_num=train_num)
    metrics = get_data.get_corresponding_metrics_by_index(index, type=type)
    metrics = get_data.padding_zero_in_matrix(metrics)
    metrics = get_data.operations2integers(metrics)
    return metrics
