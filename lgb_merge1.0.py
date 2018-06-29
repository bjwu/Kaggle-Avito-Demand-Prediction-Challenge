



import pandas as pd
import numpy as np
import gc

import lightgbm as lgb
from sklearn import metrics
from sklearn import preprocessing
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

data_proc = False
modeling = True

if data_proc:

    train_set = pd.read_csv('./input/'+'train.csv', parse_dates=["activation_date"])
    test_set = pd.read_csv('./input/'+'test.csv', parse_dates=["activation_date"])

    traindex, testdex = train_set.shape[0], test_set.shape[0]
    print('The shape of train data is: {}'.format(traindex))
    print('The shape of test data is: {}'.format(testdex))

    # city_clust = pd.read_csv('./temp_file/city_cluster.csv', names=['origin', 'now'], index_col=0).to_dict()['now']
    # pm1_clust = pd.read_csv('./temp_file/param_1_cluster.csv', names=['origin', 'now'], index_col=0).to_dict()['now']
    # pm2_clust = pd.read_csv('./temp_file/param_2_cluster.csv', names=['origin', 'now'], index_col=0).to_dict()['now']
    # pm3_clust = pd.read_csv('./temp_file/param_3_cluster.csv', names=['origin', 'now'], index_col=0).to_dict()['now']

    train_y = train_set['deal_probability']
    train_set.drop('deal_probability', axis=1, inplace = True)

    df = pd.concat([train_set, test_set], axis=0, ignore_index=True)
    del train_set, test_set
    gc.collect()

    # df['city_org'] = df['city']
    # df['param_1_org'] = df['param_1']
    # df['param_2_org'] = df['param_2']
    # df['param_3_org'] = df['param_3']
    #
    # print('------------类别特征聚类——————————————')
    # df.replace({'city':city_clust, 'param_1':pm1_clust, 'param_2':pm2_clust, 'param_3':pm3_clust}, inplace=True)

    print("----------时间特征-----------")
    df["Weekday"] = df['activation_date'].dt.weekday
    df["Weekd of Year"] = df['activation_date'].dt.week
    df["Day of Month"] = df['activation_date'].dt.day

    print("----------类别特征encode---------")
    categorical = ["user_id", "region", "parent_category_name", "category_name", "user_type","image_top_1", 'city', 'param_1', 'param_2', 'param_3']
    print("Encoding :", categorical)
    lbl = preprocessing.LabelEncoder()
    for col in categorical:
        df[col] = lbl.fit_transform(df[col].astype(str))


    print("----------文本特征处理----------------")
    textfeats = ["description", "title"]
    df["description"].fillna("NA", inplace=True)

    for cols in textfeats:
        df[cols] = df[cols].astype(str)
        df[cols] = df[cols].astype(str).fillna('missing')  # FILL NA
        df[cols] = df[cols].str.lower()  # Lowercase all text, so that capitalized words dont get treated differently
        df[cols + '_num_words'] = df[cols].apply(lambda comment: len(comment.split()))  # Count number of Words
        df[cols + '_num_unique_words'] = df[cols].apply(lambda comment: len(set(w for w in comment.split())))
        df[cols + '_words_vs_unique'] = df[cols + '_num_unique_words'] / df[cols + '_num_words'] * 100  # Count Unique Words
        df[cols + '_num_letters'] = df[cols].apply(lambda comment: len(comment))  # Count number of Letters

    ## TFIDF Vectorizer ###
    russian_stop = set(stopwords.words('russian'))
    params = {
        "stop_words": russian_stop,
        "analyzer": 'word',
        "token_pattern": r'\w{1,}',
        "norm": 'l2',
        # "min_df":,
        # "max_df":,
        "max_features": 1000,
        "smooth_idf": False,
        "ngram_range": (1, 1)
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
        full_svd.columns = [cols + '_svd_' + str(i + 1) for i in range(n_comp)]
        df = pd.concat([df, full_svd], axis=1)

    # del tfidf_vec, full_tfidf, full_svd
    # gc.collect()

    print('Drop features of no use')
    cols_to_drop = ['activation_date', "description", 'title', 'item_id', 'user_id', 'image','item_seq_number']
    df = df.drop(cols_to_drop, axis=1)

    df.to_csv('./temp_file/lgb_featclusts_df.csv')

############################### model ###################################

if modeling:

    categorical_feats = ["region", "city", "parent_category_name", "category_name", "user_type",
                         "image_top_1", "param_1", "param_2", "param_3", 'Weekday', "Weekday","Weekd of Year","Day of Month"]

    df = pd.read_csv('./temp_file/lgb_featclusts_df.csv',index_col=0)
    train_title_w2v_cnn = pd.read_csv('./temp_file/train_title_word2vec_CNN.csv',squeeze=True).rename(columns={'deal_probability':'title_w2v_cnn'})
    test_title_w2v_cnn = pd.read_csv('./temp_file/test_title_word2vec_CNN.csv',  squeeze=True).rename(columns={'deal_probability':'title_w2v_cnn'})

    train_y = pd.read_csv('./input/' + 'train.csv', usecols=['deal_probability'], squeeze=True)


    traindex = train_y.shape[0]
    # 将df特征的dtype顺势变得正常

    train_X= df[:traindex]
    train_X = pd.concat([train_X, train_title_w2v_cnn['title_w2v_cnn']], axis=1)
    print('在model之前所有的特征：', train_X.columns)
    test_X= df[traindex:].reset_index()
    del test_X['index']
    test_X = pd.concat([test_X, test_title_w2v_cnn['title_w2v_cnn']], axis=1)

    print('train_X shape: ',train_X.shape)
    print('test_X shape:', test_X.shape)

    # train_X[["param_1", "param_2", "param_3"]] = train_X[["param_1", "param_2", "param_3"]].astype(float)
    # train_X[['city']] = train_X[['city']].astype(int)

    del df
    gc.collect()



    params = {
            "objective": "regression",
            "boosting_type": "gbdt",
            'metric': 'rmse',
            "learning_rate": 0.01,  ###

            'max_depth': 16,                  ###
            "num_leaves": 250,  ###

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

    X_train, X_valid, y_train, y_valid = train_test_split(train_X, train_y, test_size=0.2, random_state=42)
    train_data = lgb.Dataset(X_train, y_train, categorical_feature=categorical_feats)
    valid_data = lgb.Dataset(X_valid, y_valid, reference=train_data, categorical_feature=categorical_feats)
    bst = lgb.train(params, train_data,
                    num_boost_round=8000,
                    valid_sets=valid_data,
                    verbose_eval=1,
                    early_stopping_rounds=50)

    print("Model Evaluation Stage")
    y_pred = bst.predict(test_X)


