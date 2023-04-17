from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn import neighbors
from sklearn import ensemble
from sklearn.tree import ExtraTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import r2_score
from scipy.stats import kendalltau
import numpy as np
import matplotlib.pyplot as plt

# different methods
# 1.decision tree regression
model_decision_tree_regression = tree.DecisionTreeRegressor()

# 2.linear regression
model_linear_regression = LinearRegression()

# 3.SVM regression
model_svm = svm.SVR()

# 4.kNN regression
model_k_neighbor = neighbors.KNeighborsRegressor()

# 5.random forest regression
model_random_forest_regressor = ensemble.RandomForestRegressor(n_estimators=230,max_features='auto')

# 6.Adaboost regression
model_adaboost_regressor = ensemble.AdaBoostRegressor()

# 7.GBRT regression
model_gradient_boosting_regressor = ensemble.GradientBoostingRegressor()

# 8.Bagging regression
model_bagging_regressor = ensemble.BaggingRegressor()

# 9.ExtraTree regression
model_extra_tree_regressor = ExtraTreeRegressor()

# 10.Gaussian Process Regression
model_gaussian_process_regressor = GaussianProcessRegressor()

# 11.MLP Regression
model_MLP_regressor = MLPRegressor()

model = [model_decision_tree_regression, model_linear_regression, model_svm, model_k_neighbor,
         model_random_forest_regressor, model_adaboost_regressor, model_gradient_boosting_regressor,
         model_bagging_regressor, model_extra_tree_regressor, model_gaussian_process_regressor, model_MLP_regressor]

model_random_forest_regressor1 = ensemble.RandomForestRegressor(n_estimators=230,max_features='auto')
model_random_forest_regressor2 = ensemble.RandomForestRegressor(n_estimators=230,max_features='sqrt')
model_random_forest_regressor3 = ensemble.RandomForestRegressor(n_estimators=230,max_features='log2')     
model_finetune_randoms = [model_random_forest_regressor1,model_random_forest_regressor2,model_random_forest_regressor3]

method = ['decision_tree', 'linear_regression', 'svm', 'knn', 'random_forest', 'adaboost', 'GBRT', 'Bagging', 'ExtraTree', 'Gaussian_Process', 'MLP']

def try_different_method(x_train, y_train, x_test, y_test, model, method, show_fig=True, return_flag=False):
    model.fit(x_train, y_train)
    result = model.predict(x_test)
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
    print('method: {:}, KTau: {:}, MSE: {:}, R2score: {:}'.format(method, KTau, calculate_MSE(y_test, result), score))
    print('N@5: {:}, N@10: {:}'.format(N_K(np.array(y_test), np.array(result), 5),
                                       N_K(np.array(y_test), np.array(result), 10)))
    print('--------------------try-end---------------------\n')

    if show_fig:
        x = np.arange(0, 1, 0.01)
        y = x
        plt.figure(figsize=(5, 5))
        plt.plot(x, y, 'g', label='y_test = result')
        plt.scatter(result, y_test, s=1)
        plt.xlabel("predict_result")
        plt.ylabel("y_test")
        plt.title(f"method:{method}---score:{score}")
        plt.legend(loc="best")
        plt.show()

        x = np.arange(0, len(y_test), 0.1)
        y = x
        plt.figure(figsize=(6, 6))
        line_color = '#1F77D0'
        plt.plot(x, y, c=line_color, linewidth=1)
        point_color = '#FF4400'
        plt.scatter(result_rank, y_test_rank, c=point_color, s=2)
        plt.xlabel("predict_result")
        plt.ylabel("y_test")
        plt.title(f"method:{method}---KTau:{KTau}")
        plt.xlim(xmax=5000, xmin=0)
        plt.ylim(ymax=5000, ymin=0)
        plt.show()

    if return_flag:
        return KTau, calculate_MSE(y_test, result)


def calculate_MSE(x, y):
    # input two list, x: predict, y: ground truth
    # output MSE
    mse_list = np.array([(element_x - element_y) ** 2 for element_x, element_y in zip(x, y)])
    mse = np.mean(mse_list)
    return mse

def N_K(target, predict, k):
    out_1 = np.argsort(-target)
    out_2 = np.argsort(-predict)
    best_k = out_2[0:k]
    rank = []
    for i in best_k:
        a = np.argwhere(out_1 == i)[0][0] + 1
        rank.append(a)
    return min(rank)