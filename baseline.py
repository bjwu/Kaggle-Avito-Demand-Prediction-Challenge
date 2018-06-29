import time
notebookstart = time.time()

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
print("Data:\n", os.listdir("../input"))

# Models Packages
from sklearn import metrics
from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD
# Gradient Boosting
import lightgbm as lgb
# Tf-Idf
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords



print("\nData Load Stage")
training = pd.read_csv('../input/train.csv',  parse_dates=["activation_date"]).sample(50000)# index_col="item_id",
traindex = training.shape[0]
testing = pd.read_csv('../input/test.csv', parse_dates=["activation_date"]).sample(50000)
testdex = testing.shape[0]

train_y = training["deal_probability"]
# training.drop("", axis=1, inplace=True)

print('Train shape: {} Rows, {} Columns'.format(*training.shape))
print('Test shape: {} Rows, {} Columns'.format(*testing.shape))

print("Combine Train and Test")
df = pd.concat([training, testing], axis=0,ignore_index=True)
# del training, testing
gc.collect()
print('\nAll Data shape: {} Rows, {} Columns'.format(*df.shape))


print(df.isnull().sum())


print("-----------价格特征----------")
df["price_new"]=np.log(df["price"].values)
df["price"].fillna(np.nanmean(df["price"].values))

df["price_new"].fillna(np.nanmean(df["price_new"].values), inplace=True) #log变换

df["image_top_1"].fillna(-999, inplace=True)
print("----------时间特征-----------")
df["Weekday"] = df['activation_date'].dt.weekday
df["Weekd of Year"] = df['activation_date'].dt.week
df["Day of Month"] = df['activation_date'].dt.day

print("----------类别特征---------")
categorical = ["user_id", "region", "city", "parent_category_name", "category_name", "user_type",
               "image_top_1", "param_1", "param_2", "param_3"]
print("Encoding :", categorical)
lbl = preprocessing.LabelEncoder()
for col in categorical:
    df[col] = lbl.fit_transform(df[col].astype(str))


print("文本特征")
textfeats = ["description", "title"]
df["description"].fillna("NA", inplace=True)

for cols in textfeats:
    df[cols] = df[cols].astype(str)
    df[cols] = df[cols].astype(str).fillna('missing')  # FILL NA
    df[cols] = df[cols].str.lower()  # Lowercase all text, so that capitalized words dont get treated differently
    df[cols + '_num_words'] = df[cols].apply(lambda comment: len(comment.split()))  # Count number of Words
    df[cols + '_num_unique_words'] = df[cols].apply(lambda comment: len(set(w for w in comment.split())))
    df[cols + '_words_vs_unique'] = df[cols + '_num_unique_words'] / df[cols + '_num_words'] * 100  # Count Unique Words



## TFIDF Vectorizer ###
russian_stop = set(stopwords.words('russian'))
params = {
    "stop_words": russian_stop,
    "analyzer": 'word',
    "token_pattern": r'\w{1,}',
    "norm": 'l2',
    "max_features": 1000,
    "smooth_idf": False,
    "ngram_range":(1,1)
}
n_comp = 10
for cols in textfeats:
    # tfidf_vec = TfidfVectorizer(ngram_range=(1,1))
    tfidf_vec = TfidfVectorizer(params)
    full_tfidf = tfidf_vec.fit_transform(df[cols].values.tolist())
    ### SVD Components ###
    svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
    svd_obj.fit(full_tfidf)
    full_svd = pd.DataFrame(svd_obj.transform(full_tfidf))
    df = pd.concat([df, full_svd], axis=1)



cols_to_drop = ["item_id", "user_id", "title", "description",'deal_probability',"activation_date", "image"]
df = df.drop(cols_to_drop, axis=1)


train_X= df.loc[:traindex-1,:]
test_X= df.loc[traindex:,:]
print(train_X.shape)
print(test_X.shape)


categorical = ["region", "city", "parent_category_name", "category_name", "user_type",
               "image_top_1", "param_1", "param_2", "param_3"]
############################### 调参 ###################################
from sklearn.model_selection import KFold

NFold = 5

skf = KFold(n_splits=NFold, shuffle=False, random_state=218)

rmse_state = []
for n_round in range(3500,8000,200):
    for learning_rate in [0.01, 0.05, 0.01]:

        params = {
                "objective": "regression",
                "boosting_type": "gbdt",
                'metric': 'rmse',
                "learning_rate": 0.01,  ###

                'max_depth': 7,                  ###
                "num_leaves": 100,  ###

                'max_bin':  255,
                "min_data_in_leaf": 50,

                'feature_fraction': 0.70,  ###
                'bagging_fraction': 0.70,  ###
                'bagging_freq': 5,

                # "drop_rate": 0.1,
                # "max_drop": 50,
                "min_child_samples": 10,        ###
                "min_child_weight": 150,       ###

                'verbose': 1
        }
        curr_score_mes = []
        kf = skf.split(train_X, train_y)
        for train_index, valid_index in kf:
            X_train_x, X_train_y, valid_x, valid_y = train_X.iloc[train_index], train_y.iloc[train_index], train_X.iloc[valid_index], train_y.iloc[valid_index]
            train_data = lgb.Dataset(X_train_x, X_train_y,categorical_feature=categorical)
            valid_data = lgb.Dataset(valid_x, valid_y, reference=train_data,categorical_feature=categorical)
            bst = lgb.train(params, train_data,
                            num_boost_round=n_round,
                            valid_sets=valid_data,
                            verbose_eval=1,
                            early_stopping_rounds=50)

            print("Model Evaluation Stage")
            pred_y = bst.predict(valid_x)

            curr_score = np.sqrt(metrics.mean_squared_error(valid_y, pred_y))
            #print( 'RMSE Score of the single stage:', curr_score,'[learnig_rate]', learning_rate,'[n_round]',n_round)
            curr_score_mes.append(curr_score)

        curr_score_mes=np.sum(curr_score_mes)/ NFold
        print('RMSE score:', curr_score_mes, '[learnig_rate]', learning_rate,'[n_round]',n_round)
        rmse_state.append([curr_score_mes, learning_rate, n_round])

best = min(rmse_state, key=lambda x: x[0])
print('The best score is', best[0], '[learning_rate]', best[1], '[n_round]', best[2])

###################################################################
learning_rate=0.01
n_round=8000
for max_depth in range(15,20,1):
    for num_leaves in [250, 500, 20]:

        params = {
                "objective": "regression",
                "boosting_type": "gbdt",
                'metric': 'rmse',
                "learning_rate": learning_rate,  ###

                'max_depth': max_depth,                  ###
                "num_leaves": num_leaves,  ###

                'max_bin':  255,
                "min_data_in_leaf": 50,

                'feature_fraction': 0.90,  ###
                'bagging_fraction': 0.90,  ###
                'bagging_freq': 5,

                # "drop_rate": 0.1,
                # "max_drop": 50,
                "min_child_samples": 10,        ###
                "min_child_weight": 150,       ###

                'verbose': 10
        }
        curr_score_mes = []
        kf = skf.split(train_X, train_y)
        for train_index, valid_index in kf:
            X_train_x, X_train_y, valid_x, valid_y = train_X.iloc[train_index], train_y.iloc[train_index], train_X.iloc[valid_index], train_y.iloc[valid_index]
            train_data = lgb.Dataset(X_train_x, X_train_y,categorical_feature=categorical)
            valid_data = lgb.Dataset(valid_x, valid_y, reference=train_data,categorical_feature=categorical)
            bst = lgb.train(params, train_data,
                            num_boost_round=n_round,
                            valid_sets=valid_data,
                            verbose_eval=1,
                            early_stopping_rounds=50)

            print("Model Evaluation Stage")
            pred_y = bst.predict(valid_x)

            curr_score = np.sqrt(metrics.mean_squared_error(valid_y, pred_y))
            #print( 'RMSE Score of the single stage:', curr_score,'[learnig_rate]', learning_rate,'[n_round]',n_round)
            curr_score_mes.append(curr_score)

        curr_score_mes=np.sum(curr_score_mes)/ NFold
        print('RMSE score:', curr_score_mes, '[learnig_rate]', max_depth,'[n_round]',num_leaves)
        rmse_state.append([curr_score_mes, max_depth, num_leaves])

best = min(rmse_state, key=lambda x: x[0])
print('The best score is', best[0], '[max_depth]', best[1], '[num_leaves]', best[2])

max_depth=16
num_leaves=250

for max_bin in range(5,255,50):
    for min_data_in_leaf in [30, 300, 20]:

        params = {
                "objective": "regression",
                "boosting_type": "gbdt",
                'metric': 'rmse',
                "learning_rate": learning_rate,  ###

                'max_depth': max_depth,                  ###
                "num_leaves": num_leaves,  ###

                'max_bin':  max_bin,
                "min_data_in_leaf": min_data_in_leaf,

                'feature_fraction': 0.90,  ###
                'bagging_fraction': 0.90,  ###
                'bagging_freq': 5,

                # "drop_rate": 0.1,
                # "max_drop": 50,
                "min_child_samples": 10,        ###
                "min_child_weight": 150,       ###

                'verbose': 10
        }
        curr_score_mes = []
        kf = skf.split(train_X, train_y)
        for train_index, valid_index in kf:
            X_train_x, X_train_y, valid_x, valid_y = train_X.iloc[train_index], train_y.iloc[train_index], train_X.iloc[valid_index], train_y.iloc[valid_index]
            train_data = lgb.Dataset(X_train_x, X_train_y,categorical_feature=categorical)
            valid_data = lgb.Dataset(valid_x, valid_y, reference=train_data,categorical_feature=categorical)
            bst = lgb.train(params, train_data,
                            num_boost_round=n_round,
                            valid_sets=valid_data,
                            verbose_eval=1,
                            early_stopping_rounds=500)

            print("Model Evaluation Stage")
            pred_y = bst.predict(valid_x)

            curr_score = np.sqrt(metrics.mean_squared_error(valid_y, pred_y))
            #print( 'RMSE Score of the single stage:', curr_score,'[learnig_rate]', learning_rate,'[n_round]',n_round)
            curr_score_mes.append(curr_score)

        curr_score_mes=np.sum(curr_score_mes)/ NFold
        print('RMSE score:', curr_score_mes, '[max_bin]', max_bin,'[min_data_in_leaf]',min_data_in_leaf)
        rmse_state.append([curr_score_mes, max_bin, min_data_in_leaf])

best = min(rmse_state, key=lambda x: x[0])
print('The best score is', best[0], '[max_bin]', best[1], '[min_data_in_leaf]', best[2])


# print("-------------------Modeling Stage-------------------------------")
# print("Light Gradient Boosting Regressor")

# n_rounds = 3000
# lgbm_params = {
#     'task': 'train',
#     'boosting_type': 'gbdt',
#     'objective': 'regression',
#     'metric': 'rmse',
#     'max_depth': 7,
#     'num_leaves': 50,
#     'feature_fraction': 0.70,
#     'bagging_fraction': 0.70,
#     'bagging_freq': 5,
#     'learning_rate': 0.005,
#     'verbose': 100
# }
#
# # Training and Validation Set
# modelstart = time.time()
#
# X_train, X_valid, y_train, y_valid = train_test_split(train_X, train_y, test_size=0.20, random_state=23)
#
# categorical = ["region", "city", "parent_category_name", "category_name", "user_type",
#                "image_top_1", "param_1", "param_2", "param_3"]
# # LGBM Dataset Formatting
# lgtrain = lgb.Dataset(X_train, y_train ,categorical_feature=categorical)
# lgvalid = lgb.Dataset(X_valid, y_valid,categorical_feature=categorical)
#
#
#
# lgb_clf = lgb.train(
#     params=lgbm_params,
#     train_set=lgtrain,
#     num_boost_round=n_rounds,
#     valid_sets=[lgtrain, lgvalid],
#     valid_names=['train', 'valid'],
#     early_stopping_rounds=500,
#     verbose_eval=100
# )
#
# print("Model Evaluation Stage")
# y_pre=lgb_clf.predict(X_valid)
# print('RMSE:', np.sqrt(metrics.mean_squared_error(y_valid,y_pre)))
#
#
#
# # Feature Importance Plot
# # f, ax = plt.subplots(figsize=[7, 10])
# lgb.plot_importance(lgb_clf, max_num_features=50)
# plt.title("Light GBM Feature Importance")
# plt.savefig('feature_import.png')
#
# print('----------提交--------------')
# lgpred = lgb_clf.predict(testing)
# del testing;
# gc.collect()
# lgsub = pd.DataFrame(lgpred, columns=["deal_probability"], index=testdex)
# lgsub['deal_probability'].clip(0.0, 1.0, inplace=True)  # Between 0 and 1
# lgsub.to_csv("lgsub.csv", index=True, header=True)
#
#
#
# # print("Model Runtime: %0.2f Minutes" % ((time.time() - modelstart) / 60))
# # print("Notebook Runtime: %0.2f Minutes" % ((time.time() - notebookstart) / 60))