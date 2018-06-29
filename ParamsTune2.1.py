
### 这个调参文件是对于在分类器之后的全1的值进行回归预测
### 最后依据调参后的值，rmse可达到0.2303

import pandas as pd
import lightgbm as lgb
import numpy as np
import gc
import pickle

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics


with open('./temp_file/df_train2.pickle','rb') as file:
    df_train =pickle.load(file)
with open('./temp_file/df_valid2.pickle','rb') as file:
    df_test =pickle.load(file)
with open('./temp_file/y_train2.pickle','rb') as file:
    y_train=pickle.load(file)
with open('./temp_file/y_test2.pickle','rb') as file:
    y_test=pickle.load(file)




cat_cols = {"region","city","parent_category_name","category_name","user_type","image_top_1","param_2","param_3"}

train_data = lgb.Dataset(df_train, label=y_train, categorical_feature=list(cat_cols),free_raw_data=False)
valid_data = lgb.Dataset(df_test, label=y_test, categorical_feature=list(cat_cols), reference=train_data, free_raw_data=False)
watchlist2 = [valid_data]

print('Params Tuning...')
params = {
    "objective": "regression",
    "boosting_type": "gbdt",
    'metric': 'rmse',
    'num_threads':3,
    'verbose': -1,       #####可以避免[LightGBM] [Warning] No further splits with positive gain, best gain: -inf的出现

    'min_data_in_leaf':1200,
    'max_bin':255,
    'learning_rate': 0.01,
    'bagging_fraction':0.8,
    'feature_fraction':0.7
}


print('提高准确率1')

rmse_state1 = []
for max_depth in [14,15,17]:
    for num_leaves in [250,500,600,700,800]:
        params['num_leaves'] = num_leaves
        params['max_depth'] = max_depth


        bst = lgb.train(params, train_data, num_boost_round=10000, valid_sets=valid_data, verbose_eval=500, early_stopping_rounds=50)
        ### verbose_eval表示多少次迭代显示一次

        pred_y = bst.predict(df_test)
        curr_score = np.sqrt(metrics.mean_squared_error(y_test, pred_y))

        print('RMSE score:', curr_score, '[num_leaves]', num_leaves,'[max_depth]', max_depth)
        rmse_state1.append([curr_score, num_leaves, max_depth])

best = min(rmse_state1, key=lambda x: x[0])
print('########################The best score:##########', best[0], '[num_leaves]', best[1], '[max_depth]', best[2])

# 'min_data_in_leaf':100,
# 'max_bin':255,
# 'learning_rate': 0.08
# for max_depth in [9,11,15,18,21]:
# for num_leaves in [300,500,700,900]:
# 15, 700
#
params['num_leaves'] = best[1]
params['max_depth'] = best[2]
# params['num_leaves'] = 700
# params['max_depth'] = 17

print('过拟合2')
rmse_state2 = []
for max_bin in [255, 280, 330]:
    for min_data_in_leaf in [1100,1200,1300]:
        params['max_bin'] = max_bin
        params['min_data_in_leaf'] = min_data_in_leaf

        bst = lgb.train(params, train_data, num_boost_round=10000, valid_sets=valid_data, verbose_eval=500, early_stopping_rounds=50)

        pred_y = bst.predict(df_test)
        curr_score = np.sqrt(metrics.mean_squared_error(y_test, pred_y))

        print('RMSE score:', curr_score, '[max_bin]', max_bin,'[min_data_in_leaf]', min_data_in_leaf)
        rmse_state2.append([curr_score, max_bin, min_data_in_leaf])

best = min(rmse_state2, key=lambda x: x[0])
print('########################The best score:##########', best[0], '[max_bin]', best[1], '[min_data_in_leaf]', best[2])

params['max_bin'] = best[1]
params['min_data_in_leaf'] = best[2]

# min_data_in_leaf --[700,900,1200,1500,2000], (255上,1200):0.231048
# params['max_bin'] = 255
# params['min_data_in_leaf'] = 1200

print('过拟合3')
rmse_state3 = []
for bagging_fraction in [0.7,0.9]:   #### sub_features
    for feature_fraction in [0.55,0.7,0.9]:
        params['feature_fraction'] = feature_fraction
        params['bagging_fraction'] = bagging_fraction


        bst = lgb.train(params, train_data, num_boost_round=10000, valid_sets=valid_data, verbose_eval=500, early_stopping_rounds=50)

        pred_y = bst.predict(df_test)
        curr_score = np.sqrt(metrics.mean_squared_error(y_test, pred_y))

        print('RMSE score:', curr_score, '[feature_fraction]', feature_fraction,'[bagging_fraction]', bagging_fraction)
        rmse_state3.append([curr_score, feature_fraction, bagging_fraction])

best = min(rmse_state3, key=lambda x: x[0])
print('########################The best score:##########', best[0], '[feature_fraction]', best[1], '[bagging_fraction]', best[2])
#
params['feature_fraction'] = best[1]
params['bagging_fraction'] = best[2]
# params['feature_fraction'] = 0.7
# params['bagging_fraction'] = 0.9



bst = lgb.train(params, train_data, num_boost_round=5000, valid_sets=valid_data, verbose_eval=500, early_stopping_rounds=50)

pred_y = bst.predict(df_test)
curr_score = np.sqrt(metrics.mean_squared_error(y_test, pred_y))
#

# print('过拟合4')
# rmse_state4 = []
# for lambda_l1 in [0.5,0.7,0.9]:   #### sub_features
#     for lambda_l2 in [0.6,0.8,1.0]:
#         for min_split_gain in [0.5, 0.7, 0.9]:
#             params['lambda_l1'] = lambda_l1
#             params['lambda_l2'] = lambda_l2
#             params['min_split_gain'] = min_split_gain
#
#             bst = lgb.train(params, train_data, num_boost_round=5000, valid_sets=valid_data, verbose_eval=500, early_stopping_rounds=20)
#
#             pred_y = bst.predict(df_test)
#             curr_score = np.sqrt(metrics.mean_squared_error(y_test, pred_y))
#
#             print('RMSE score:', curr_score, '[lambda_l1]', lambda_l1,'[lambda_l2]', lambda_l2,'[min_split_gain]',min_split_gain)
#             rmse_state4.append([curr_score, lambda_l1, lambda_l2, min_split_gain])
#
# best = min(rmse_state4, key=lambda x: x[0])
# print('########################The best score:##########', best[0], '[lambda_l1]', best[1], '[lambda_l2]', best[2], '[min_split_gain]', best[3])
#
# params['lambda_l1'] = best[1]
# params['lambda_l2'] = best[2]
# params['min_split_gain'] = best[3]






