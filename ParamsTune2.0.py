## 这个文件是针对merge1.4.py文件进行的调参，也就是说在没有加入分类器之前

import pandas as pd
import lightgbm as lgb
import numpy as np
import gc

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics

all_train = pd.read_csv('./temp_file/train_data_before_model_0623.csv', index_col = "item_id")
all_test = pd.read_csv('./temp_file/test_data_before_model_0623.csv', index_col = "item_id")
all_y_train = pd.read_csv('./temp_file/y_train_before_model_0623.csv', names=['item_id', 'deal_probability'], index_col=0 )['deal_probability']


all_y_train1 = all_y_train.apply(lambda x: 1 if x>0 else 0)
all_train = all_train[all_y_train1 == 1]
all_y_train1 = all_y_train[all_y_train1 == 1]

df_train, df_test, y_train, y_test = train_test_split(all_train, all_y_train1, test_size=0.2, random_state=144)
del all_train , all_y_train
gc.collect()




print("----------类别特征---------")
cat_cols = {"region","city","parent_category_name","category_name","user_type","image_top_1","param_2","param_3"}

print('Params Tuning...')
params = {
    "objective": "regression",
    "boosting_type": "gbdt",
    'learning_rate': 0.01,
    'metric': 'rmse',
    'num_threads':3,
    'verbose': -1       #####可以避免[LightGBM] [Warning] No further splits with positive gain, best gain: -inf的出现
}
print('提高准确率1')


rmse_state1 = []
for max_depth in [14,15,16]:
    for num_leaves in [250,300,400]:
        params['num_leaves'] = num_leaves
        params['max_depth'] = max_depth

        train_data = lgb.Dataset(df_train, y_train)
        valid_data = lgb.Dataset(df_test, y_test, reference=train_data)

        bst = lgb.train(params, train_data, num_boost_round=4000, valid_sets=valid_data, verbose_eval=500, early_stopping_rounds=20)
        ### verbose_eval表示多少次迭代显示一次

        pred_y = bst.predict(df_test)
        curr_score = np.sqrt(metrics.mean_squared_error(y_test, pred_y))

        print('RMSE score:', curr_score, '[num_leaves]', num_leaves,'[max_depth]', max_depth)
        rmse_state1.append([curr_score, num_leaves, max_depth])

best = min(rmse_state1, key=lambda x: x[0])
print('########################The best score:##########', best[0], '[num_leaves]', best[1], '[max_depth]', best[2])

params['num_leaves'] = best[1]
params['max_depth'] = best[2]
# params['num_leaves'] = 15
# params['max_depth'] = 300

print('过拟合2')
rmse_state2 = []
for max_bin in [100,200,300]:
    for min_data_in_leaf in [50,100,200]:
        params['max_bin'] = max_bin
        params['min_data_in_leaf'] = min_data_in_leaf

        train_data = lgb.Dataset(df_train, y_train)
        valid_data = lgb.Dataset(df_test, y_test, reference=train_data)

        bst = lgb.train(params, train_data, num_boost_round=5000, valid_sets=valid_data, verbose_eval=500, early_stopping_rounds=20)

        pred_y = bst.predict(df_test)
        curr_score = np.sqrt(metrics.mean_squared_error(y_test, pred_y))

        print('RMSE score:', curr_score, '[max_bin]', max_bin,'[min_data_in_leaf]', min_data_in_leaf)
        rmse_state2.append([curr_score, max_bin, min_data_in_leaf])

best = min(rmse_state2, key=lambda x: x[0])
print('########################The best score:##########', best[0], '[max_bin]', best[1], '[min_data_in_leaf]', best[2])

params['max_bin'] = best[1]
params['min_data_in_leaf'] = best[2]
# params['max_bin'] = 100
# params['min_data_in_leaf'] = 200

print('过拟合3')
rmse_state3 = []
for feature_fraction in [0.5,0.7,0.9]:   #### sub_features
    for bagging_fraction in [0.6,0.8,1.0]:
        params['feature_fraction'] = feature_fraction
        params['bagging_fraction'] = bagging_fraction

        train_data = lgb.Dataset(df_train, y_train)
        valid_data = lgb.Dataset(df_test, y_test, reference=train_data)

        bst = lgb.train(params, train_data, num_boost_round=5000, valid_sets=valid_data, verbose_eval=500, early_stopping_rounds=20)

        pred_y = bst.predict(df_test)
        curr_score = np.sqrt(metrics.mean_squared_error(y_test, pred_y))

        print('RMSE score:', curr_score, '[feature_fraction]', feature_fraction,'[bagging_fraction]', bagging_fraction)
        rmse_state3.append([curr_score, feature_fraction, bagging_fraction])

best = min(rmse_state3, key=lambda x: x[0])
print('########################The best score:##########', best[0], '[feature_fraction]', best[1], '[bagging_fraction]', best[2])

params['feature_fraction'] = best[1]
params['bagging_fraction'] = best[2]
# params['feature_fraction'] = 0.7
# params['bagging_fraction'] = 0.6


# print('过拟合4')
# rmse_state4 = []
# for lambda_l1 in [0.5,0.7,0.9]:   #### sub_features
#     for lambda_l2 in [0.6,0.8,1.0]:
#         for min_split_gain in [0.5, 0.7, 0.9]:
#             params['lambda_l1'] = lambda_l1
#             params['lambda_l2'] = lambda_l2
#             params['min_split_gain'] = min_split_gain
#
#
#             train_data = lgb.Dataset(df_train, y_train)
#             valid_data = lgb.Dataset(df_test, y_test, reference=train_data)
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






