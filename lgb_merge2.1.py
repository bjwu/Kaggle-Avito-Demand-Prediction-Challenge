
####2.0 采用先分类后递归的方法进行预测

import time
start_time = time.time()

# SUBMIT_MODE = True  ##### 输出test结果
SUBMIT_MODE = False ##### 仅仅验证

data_preproc = True

import pandas as pd
import numpy as np
import time
import gc
import string
import re
import random
import matplotlib.pyplot as plt
random.seed(2018)

from nltk.corpus import stopwords

from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import  LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import recall_score, f1_score, fbeta_score
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import lightgbm as lgb


############# Help string & functions ##################

stopwords_en = {x: 1 for x in stopwords.words('english')}
stopwords_ru = {x: 1 for x in stopwords.words('russian')}
non_alphanums = re.compile(u'[^A-Za-z0-9]+')
non_alphanumpunct = re.compile(u'[^A-Za-z0-9\.?!,; \(\)\[\]\'\"\$]+')
RE_PUNCTUATION = '|'.join([re.escape(x) for x in string.punctuation])

def rmse(predicted, actual):
    return np.sqrt(((predicted - actual) ** 2).mean())

def plot_feat_importance(model):
    fig, ax = plt.subplots(figsize=(12,18))
    lgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
    ax.grid(False)
    plt.title("LightGBM - Feature Importance", fontsize=15)
    plt.show()

def submit(coat, body, name):
    coat['deal_probability'] = body
    coat['deal_probability'] = coat['deal_probability'].clip(0.0, 1.0) # Between 0 and 1
    coat.to_csv('./output/'+name+'.csv', index=False)

def fbeta(y_true, pred):
    # 调整阈值
    for thershold in [0.3, 0.4, 0.45, 0.5,]:
        preds = [1 if i > thershold else 0 for i in pred]
        print('For thershold', thershold, ' ,the fbeta score is', fbeta_score(y_true, preds, 2))
    return 'f1_score', fbeta_score(y_true, preds, 2), True

def lgb_recall(y_true, pred):
    # preds = np.argmax(pred, axis=0)

    #preds = [1 for i in pred ]
    return 'recall_score', recall_score(y_true, pred), True

def clean_name(x):
    if len(x):
        x = non_alphanums.sub(' ', x).split()
        if len(x):
            return x[0].lower()
    return ''

def to_number(x):
    try:
        if not x.isdigit():
            return 0
        x = int(x)
        if x > 100:
            return 100
        else:
            return x
    except:
        return 0

def sum_numbers(desc):
    if not isinstance(desc, str):
        return 0
    try:
        return sum([to_number(s) for s in desc.split()])
    except:
        return 0

def ridge_proc(train_data, test_data, y_train):
    X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split(train_data, y_train,
                                                                  test_size = 0.5,
                                                                  shuffle = False)
    print('[{}] Finished splitting'.format(time.time() - start_time))

    model = Ridge(solver="sag", fit_intercept=True, random_state=205, alpha=3.3)
    model.fit(X_train_1, y_train_1)
    print('[{}] Finished to train ridge (1)'.format(time.time() - start_time))
    ridge_preds1 = model.predict(X_train_2)
    ridge_preds1f = model.predict(test_data)
    print('[{}] Finished to predict ridge (1)'.format(time.time() - start_time))
    model = Ridge(solver="sag", fit_intercept=True, random_state=205, alpha=3.3)
    model.fit(X_train_2, y_train_2)
    print('[{}] Finished to train ridge (2)'.format(time.time() - start_time))
    ridge_preds2 = model.predict(X_train_1)
    ridge_preds2f = model.predict(test_data)
    print('[{}] Finished to predict ridge (2)'.format(time.time() - start_time))
    ridge_preds_oof = np.concatenate((ridge_preds2, ridge_preds1), axis=0)
    ridge_preds_test = (ridge_preds1f + ridge_preds2f) / 2.0
    print('RMSLE OOF: {}'.format(rmse(ridge_preds_oof, y_train)))

    return ridge_preds_oof, ridge_preds_test

print('[{}] Finished defining stuff'.format(time.time() - start_time))

if data_preproc:

    ######################## Loading data #################################

    train = pd.read_csv('./input/train.csv', index_col = "item_id", parse_dates = ["activation_date"])
    test = pd.read_csv('./input/test.csv', index_col = "item_id", parse_dates = ["activation_date"])
    print('[{}] Finished load data'.format(time.time() - start_time))


    train['is_train'] = 1
    test['is_train'] = 0
    print('[{}] Compiled train / test'.format(time.time() - start_time))
    print('Train shape: ', train.shape)
    print('Test shape: ', test.shape)

    y = train.deal_probability.copy()
    train.drop('deal_probability', axis=1, inplace = True)
    nrow_train = train.shape[0]

    merge = pd.concat([train, test])

    print('[{}] Compiled merge'.format(time.time() - start_time))
    print('Merge shape: ', merge.shape)
    print('merge\'s columns:', merge.columns)

    del train, test
    gc.collect()
    print('[{}] Garbage collection'.format(time.time() - start_time))

    ####################### Basic Feature Engineering #####################

    merge["price"] = np.log(merge["price"]+0.001)
    merge["price"].fillna(-999,inplace=True)
    merge["image_top_1"].fillna(-999,inplace=True)

    # print("\nCreate Time Variables")
    # merge["activation_weekday"] = merge['activation_date'].dt.weekday
    # merge["Weekd_of_Year"] = merge['activation_date'].dt.week
    # merge["Day_of_Month"] = merge['activation_date'].dt.day

    print(merge.head(5))
    gc.collect()

    merge.drop(["activation_date", "image", "user_id"],axis=1,inplace=True)

    merge['param_1_copy'] = merge['param_1']


    ################# Feature from text ##############################
    print("\nText Features")
    textfeats = ["description", "title", "param_1_copy"]

    for cols in textfeats:
        merge[cols] = merge[cols].astype(str)
        merge[cols] = merge[cols].astype(str).fillna('missing') # FILL NA
        merge[cols] = merge[cols].str.lower() # Lowercase all text, so that capitalized words dont get treated differently
        merge[cols + '_num_stopwords_en'] = merge[cols].apply(lambda x: len([w for w in x.split() if w in stopwords_en]))  # Count number of Stopwords
        merge[cols + '_num_stopwords'] = merge[cols].apply(lambda x: len([w for w in x.split() if w in stopwords_ru])) # Count number of Stopwords
        merge[cols + '_num_punctuations'] = merge[cols].apply(lambda comment: (comment.count(RE_PUNCTUATION))) # Count number of Punctuations
        merge[cols + '_num_alphabets'] = merge[cols].apply(lambda comment: (comment.count(r'[a-zA-Z]'))) # Count number of Alphabets
        merge[cols + '_num_alphanumeric'] = merge[cols].apply(lambda comment: (comment.count(r'[A-Za-z0-9]'))) # Count number of AlphaNumeric
        merge[cols + '_num_digits'] = merge[cols].apply(lambda comment: (comment.count('[0-9]'))) # Count number of Digits
        merge[cols + '_num_letters'] = merge[cols].apply(lambda comment: len(comment)) # Count number of Letters
        merge[cols + '_num_words'] = merge[cols].apply(lambda comment: len(comment.split())) # Count number of Words
        merge[cols + '_num_unique_words'] = merge[cols].apply(lambda comment: len(set(w for w in comment.split())))
        merge[cols + '_words_vs_unique'] = merge[cols+'_num_unique_words'] / merge[cols+'_num_words'] # Count Unique Words
        merge[cols + '_letters_per_word'] = merge[cols+'_num_letters'] / merge[cols+'_num_words'] # Letters per Word
        merge[cols + '_punctuations_by_letters'] = merge[cols+'_num_punctuations'] / merge[cols+'_num_letters'] # Punctuations by Letters
        merge[cols + '_punctuations_by_words'] = merge[cols+'_num_punctuations'] / merge[cols+'_num_words'] # Punctuations by Words
        merge[cols + '_digits_by_letters'] = merge[cols+'_num_digits'] / merge[cols+'_num_letters'] # Digits by Letters
        merge[cols + '_alphanumeric_by_letters'] = merge[cols+'_num_alphanumeric'] / merge[cols+'_num_letters'] # AlphaNumeric by Letters
        merge[cols + '_alphabets_by_letters'] = merge[cols+'_num_alphabets'] / merge[cols+'_num_letters'] # Alphabets by Letters
        merge[cols + '_stopwords_by_letters'] = merge[cols+'_num_stopwords'] / merge[cols+'_num_letters'] # Stopwords by Letters
        merge[cols + '_stopwords_by_words'] = merge[cols+'_num_stopwords'] / merge[cols+'_num_words'] # Stopwords by Letters
        merge[cols + '_stopwords_by_letters_en'] = merge[cols+'_num_stopwords_en'] / merge[cols+'_num_letters'] # Stopwords by Letters
        merge[cols + '_stopwords_by_words_en'] = merge[cols+'_num_stopwords_en'] / merge[cols+'_num_words'] # Stopwords by Letters
        merge[cols + '_mean'] = merge[cols].apply(lambda x: 0 if len(x) == 0 else float(len(x.split())) / len(x)) * 10 # Mean
        merge[cols + '_num_sum'] = merge[cols].apply(sum_numbers)
        print(cols +' Feature done')

    # Extra Feature Engineering
    merge['title_desc_len_ratio'] = merge['title_num_letters']/(merge['description_num_letters']+1)
    merge['title_param1_len_ratio'] = merge['title_num_letters']/(merge['param_1_copy_num_letters']+1)
    merge['param_1_copy_desc_len_ratio'] = merge['param_1_copy_num_letters']/(merge['description_num_letters']+1)

    print('[{}] Getting feature from text'.format(time.time() - start_time))
    gc.collect()

    ############### Set columns type ##############################

    cols = set(merge.columns.values)
    cat_cols = {"region","city","parent_category_name","category_name","user_type","image_top_1","param_2","param_3"}
    basic_cols = {"region","city","parent_category_name","category_name","user_type","image_top_1",
                   "description","title","param_1_copy","param_1","param_2","param_3", "price", "item_seq_number"}
    text_cols = ["description","title","param_1_copy"]
    drop_clos = ['param_1','param_1_copy','title','description']
    num_features = list(cols - (basic_cols)).remove('is_train')

    ############### Label Encode ##############################
    print("----------LabelEncode--------")
    print("Encoding :", cat_cols)
    lbl = LabelEncoder()
    for col in cat_cols:
        merge[col] = lbl.fit_transform(merge[col].astype(str))

    print('[{}] Label Encode.'.format(time.time() - start_time))

    ################# Split train and test data ###############################
    df_test = merge.loc[merge['is_train'] == 0]
    df_train = merge.loc[merge['is_train'] == 1]
    del merge
    gc.collect()

    all_test = df_test.drop(['is_train'], axis=1)
    all_train = df_train.drop(['is_train'], axis=1)
    all_y_train = y

    print(all_train.shape)
    print(all_y_train.shape)


    ############### StandardScaler ############################
    # scaler = StandardScaler()
    #
    # train_num_features = scaler.fit_transform(df_train[num_features].drop('deal_probability', axis=1))
    # test_num_features = scaler.fit_transform(df_test[num_features].drop('deal_probability', axis=1))
    #
    # train_num_features = csr_matrix(train_num_features)
    # test_num_features = csr_matrix(test_num_features)


    ############### TFIDF ######################

    tv = TfidfVectorizer(max_features=100000,
                         ngram_range=(1, 2),
                         stop_words=set(stopwords.words('russian')))
    X_name_train = tv.fit_transform(all_train['title'])
    print('[{}] Finished TFIDF vectorize `title` (1/2)'.format(time.time() - start_time))
    X_name_test = tv.transform(all_test['title'])
    print('[{}] Finished TFIDF vectorize `title` (2/2)'.format(time.time() - start_time))

    tv = TfidfVectorizer(max_features=250000,
                         ngram_range=(1, 3),
                         stop_words=set(stopwords.words('russian')))
    X_description_train = tv.fit_transform(all_train['description'])
    print('[{}] Finished TFIDF vectorize `description` (1/2)'.format(time.time() - start_time))
    X_description_test = tv.transform(all_test['description'])
    print('[{}] Finished TFIDF vectorize `description` (2/2)'.format(time.time() - start_time))

    tv = TfidfVectorizer(max_features=50000,
                         ngram_range=(1, 2),
                         stop_words=None)
    X_param1_train = tv.fit_transform(all_train['param_1_copy'])
    print('[{}] Finished TFIDF vectorize `param_1_copy` (1/2)'.format(time.time() - start_time))
    X_param1_test = tv.transform(all_test['param_1_copy'])
    print('[{}] Finished TFIDF vectorize `param_1_copy` (2/2)'.format(time.time() - start_time))


    sparse_merge_train = hstack((X_description_train, X_param1_train, X_name_train)).tocsr()
    del X_description_train, X_param1_train, X_name_train
    gc.collect()
    print('[{}] Create sparse merge train completed'.format(time.time() - start_time))

    sparse_merge_test = hstack((X_description_test, X_param1_test, X_name_test)).tocsr()
    del X_description_test, X_param1_test, X_name_test
    gc.collect()
    print('[{}] Create sparse merge test completed'.format(time.time() - start_time))

    text_ridge_pred_train, text_ridge_pred_test = ridge_proc(sparse_merge_train, sparse_merge_test, y)
    del sparse_merge_test, sparse_merge_train
    gc.collect()

    ################## Features Merge and Drop########################

    all_train.drop(drop_clos, axis=1, inplace=True)
    all_test.drop(drop_clos, axis=1, inplace=True)

    all_train['text_ridge'] = text_ridge_pred_train
    all_test['text_ridge'] = text_ridge_pred_test

    all_train.to_csv('./temp_file/train_data_before_model.csv')
    all_test.to_csv('./temp_file/test_data_before_model.csv')
    all_y_train.to_csv('./temp_file/y_train_before_model.csv')
else:
    all_train = pd.read_csv('./temp_file/train_data_before_model.csv', index_col = "item_id")
    all_test = pd.read_csv('./temp_file/test_data_before_model.csv', index_col = "item_id")
    all_y_train = pd.read_csv('./temp_file/y_train_before_model.csv', names=['item_id', 'deal_probability'], index_col=0 )['deal_probability']
    cat_cols = {"region", "city", "parent_category_name", "category_name", "user_type", "image_top_1", "param_2", "param_3"}

print('Add external files')
train_imgtop_jh = pd.read_csv('./temp_file/train_image_top_1_features.csv',index_col="item_id", squeeze=True)
test_imgtop_jh = pd.read_csv('./temp_file/test_image_top_1_features.csv',index_col="item_id", squeeze=True)
train_des_w2vRnn_jh = pd.read_csv('./temp_file/train_description_word2vec_RNN3.0.csv', index_col="item_id", squeeze=True)
test_des_w2vRnn_jh = pd.read_csv('./temp_file/test_description_word2vec_RNN3.0.csv', index_col="item_id", squeeze=True)

print('merge external files to df')
all_train['img_top_complete'] = train_imgtop_jh
all_train['des_w2vRnn'] = train_des_w2vRnn_jh
all_test['img_top_complete'] = test_imgtop_jh
all_test['des_w2vRnn'] = test_des_w2vRnn_jh
submission = pd.DataFrame(all_test.index)


################## LightGBM #############################

print('Modeling begins...')

################# MODEL 1 ####################

###################### For classification ################
#
all_y_train1 = all_y_train.apply(lambda x: 1 if x>0 else 0)
# # lbl = OneHotEncoder()
# # all_y_train1 = lbl.fit_transform(all_y_train1)
#
# df_train, df_test, y_train, y_test = train_test_split(all_train, all_y_train1, test_size=0.15, random_state=144)
# # del all_y_train1
# # gc.collect()
#
# modelC = lgb.LGBMClassifier(num_leaves=400,
#                             max_depth=18,
#                             learning_rate=0.02,
#                             max_bin=150,
#                             objective='binary',
#                             subsample=0.9,
#                             random_state=218,
#                             n_estimators=1000
#                             )
# modelC.fit(df_train, y_train, eval_set= [(df_test, y_test)], eval_metric=fbeta, categorical_feature=list(cat_cols), early_stopping_rounds=50, verbose=50)
#
# ### 确定阈值thershold
# tmp = modelC.predict_proba(df_test)
# fbeta(y_test, tmp)
#
#
# test_pred1 = modelC.predict_proba(all_test)
#
# print('The number of 1 in test_pred1:', test_pred1.sum())
#
# del df_train, df_test, y_train, y_test
# gc.collect()

################# MODEL 2 #######################

all_train = all_train[all_y_train1 == 1]
all_y_train1 = all_y_train1[all_y_train1 == 1]

df_train2, df_test2, y_train2, y_test2 = train_test_split(all_train, all_y_train, test_size=0.15, random_state=244)

d_train = lgb.Dataset(df_train2, label=y_train2, categorical_feature=list(cat_cols))
d_valid = lgb.Dataset(df_test2, label=y_test2, categorical_feature=list(cat_cols), reference=d_train)
watchlist = [d_valid]

params = {
    'learning_rate': 0.01,
    'application': 'regression',
    'metric': 'RMSE',

    'max_depth': 10,
    'num_leaves': 200,

    'max_bin': 150,
    "min_data_in_leaf": 50,

    'bagging_fraction': 0.9,
    'feature_fraction': 0.5,

    "min_child_samples": 10,
    "min_child_weight": 150,

    'lambda_l1': 0.8,
    'lambda_l2': 0.8,

    'verbose': -1
}
modelR = lgb.train(params,
                  train_set=d_train,
                  num_boost_round=8000,
                  valid_sets=watchlist,
                  early_stopping_rounds=50,
                  verbose_eval=50)

test_pred2 = modelR.predict(all_test)
#
# del all_train, all_y_train
# gc.collect()
#
# ################## Blending ##############################
#
# test_pred = [test_pred2[i] if s == 1 else 0 for i, s in enumerate(test_pred1)]




