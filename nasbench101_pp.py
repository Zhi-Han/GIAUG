'''
This is for Table 1. And figure 4.
'''
import numpy as np

import exp
from sklearn import ensemble
from exp import get_toy_data
from models import try_different_method, model, method
# def calculate_MSE(x, y):
#     # input two list, x: predict, y: ground truth
#     # output MSE
#     mse_list = np.array([(element_x - element_y) ** 2 for element_x, element_y in zip(x, y)])
#     mse = np.mean(mse_list)
#     return mse


def main(train_num, test_num):
    # DO NOT change the following parameters.
    integers2one_hot = True
    data_augmentation = True
    # model = ensemble.RandomForestRegressor(n_estimators=230)
    # method = 'Random_Forest'

    metrics = exp.get_toy_metrics(train_num)
    print('----------------------train---------------------')
    X, y, _ = exp.get_toy_data(metrics, create_more_metrics=data_augmentation, integers2one_hot=integers2one_hot)

    print('----------------------test----------------------')
    # You could change the type='random_test' to resample test data.
    test_metrics = exp.get_toy_metrics(test_num, type='fixed_test', train_num=train_num)
    testX, testy, num_new_metrics = exp.get_toy_data(test_metrics, create_more_metrics =False, integers2one_hot=integers2one_hot)
    
    for i in range(0,len(method)):
        print("------------------Method {:}-----------------".format(method[i]))
        try_different_method(X, y, testX, testy, model[i], method[i], show_fig=True)

if __name__ == '__main__':
    valid_train_splits = [0.0001,0.0003,0.0005,0.0007,0.001,0.003,0.005,0.007,0.01]
    num_all = 423624
    for i in valid_train_splits:
        num_train = int(i * num_all)
        num_test = num_all - num_train
        print("--------%f|%d train-----%d test------------"%(i,num_train,num_test))
        main(num_train, num_test)

