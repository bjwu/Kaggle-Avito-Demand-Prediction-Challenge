
####2.0 采用先分类后递归的方法进行预测
#### 这个是在1.4的基础上，完善了tfidf的fit过程，且加入了一个分类器，用于预先分类设定好的0，1值，然后取全部的预测1值进行下一步的回归，具体调参见ParamTune2.1

import time
start_time = time.time()

# SUBMIT_MODE = True  ##### 输出test结果
SUBMIT_MODE = False ##### 仅仅验证

data_preproc = False

import pandas as pd
import numpy as np
import time
import gc
import string
import re
import random
import pickle
import matplotlib.pyplot as plt
random.seed(2018)

from nltk.corpus import stopwords

from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import  LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import recall_score, f1_score, fbeta_score,precision_recall_fscore_support,mean_squared_error
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import Ridge
import lightgbm as lgb


############# Help string & functions ##################

stopwords_en = {x: 1 for x in stopwords.words('english')}
stopwords_russian = {x: 1 for x in stopwords.words('russian')}
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


def ridge_proc(data, y_train):
    ###data：train和test加起来的特征集合
    ###y_train: train的target

    ### 输出为一行特征（包括train和test）
    train_data = data[:nrow_train]
    test_data = data[nrow_train:]
    X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split(train_data, y_train,
                                                                  test_size=0.5,
                                                                  shuffle=False)
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

    return np.concatenate((ridge_preds_oof, ridge_preds_test), axis=0)

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
        merge[cols + '_num_stopwords'] = merge[cols].apply(lambda x: len([w for w in x.split() if w in stopwords_russian])) # Count number of Stopwords
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

    ############### TFIDF ######################
    # russian_stop = set(stopwords.words('russian'))
    tv = TfidfVectorizer(max_features=100000,
                         ngram_range=(1, 2),
                         stop_words=set(stopwords.words('russian')))
    X_name = tv.fit_transform(merge['title'])
    print('[{}] Finished TFIDF vectorize `title`'.format(time.time() - start_time))

    tv = TfidfVectorizer(max_features=250000,
                         ngram_range=(1, 3),
                         stop_words=set(stopwords.words('russian')))
    X_description = tv.fit_transform(merge['description'])
    print('[{}] Finished TFIDF vectorize `description` (1/2)'.format(time.time() - start_time))

    tv = TfidfVectorizer(max_features=50000,
                         ngram_range=(1, 2),
                         stop_words=None)
    X_param1 = tv.fit_transform(merge['param_1_copy'])
    print('[{}] Finished TFIDF vectorize `param_1_copy`'.format(time.time() - start_time))

    sparse_merge = hstack((X_description, X_param1, X_name)).tocsr()
    del X_description, X_param1, X_name
    gc.collect()
    print('[{}] Create sparse merge train completed'.format(time.time() - start_time))


    text_ridge_pred= ridge_proc(sparse_merge, y)
    merge['text_ridge'] = text_ridge_pred

    del sparse_merge
    gc.collect()

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


    ################## Features Merge and Drop########################

    all_train.drop(drop_clos, axis=1, inplace=True)
    all_test.drop(drop_clos, axis=1, inplace=True)

    all_train.to_csv('./temp_file/train_data_before_model_0623.csv')
    all_test.to_csv('./temp_file/test_data_before_model_0623.csv')
    all_y_train.to_csv('./temp_file/y_train_before_model_0623.csv')
else:
    all_train = pd.read_csv('./temp_file/train_data_before_model_0623.csv', index_col = "item_id")
    all_test = pd.read_csv('./temp_file/test_data_before_model_0623.csv', index_col = "item_id")
    all_y_train = pd.read_csv('./temp_file/y_train_before_model_0623.csv', names=['item_id', 'deal_probability'], index_col=0 )['deal_probability']
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

df_train, df_test, y_train, y_test = train_test_split(all_train, all_y_train, test_size=0.15, random_state=144)

y_train1 = y_train.apply(lambda x: 1 if x>0 else 0)
y_test1 = y_test.apply(lambda x: 1 if x>0 else 0)
# del all_y_train1
# gc.collect()
d_train = lgb.Dataset(df_train, label=y_train1, categorical_feature=list(cat_cols),free_raw_data=False)
d_valid = lgb.Dataset(df_test, label=y_test1, categorical_feature=list(cat_cols), reference=d_train,free_raw_data=False)
watchlist = [d_valid]
cw={0: 1, 1: 2}
def fbeta(y_true, pred):
    # 调整阈值
    best_recall=0
    precision_recall = []
    best_thershold=0
    for thershold in [0.15,0.2, 0.25,0.3]:
        preds = [1 if i > thershold else 0 for i in pred]
        precision, recall, f_score, true_sum=precision_recall_fscore_support(y_true, preds)
        if recall[1]>best_recall:
            best_thershold=thershold
            precision_recall = []
            best_recall=recall[1]
            precision_recall.append(precision)
            precision_recall.append(recall)
            precision_recall.append(f_score)
            precision_recall.append(true_sum)
    print('For thershold', best_thershold, ' ,the fbeta score is',precision_recall)
    return 'best_recall', best_recall, True
params = {
    'learning_rate': 0.01,
    'application': 'binary',
    'is_unbalance':True,
    'metric': ['auc','binary_error','fbeta'],

    'max_depth': 14,
    'num_leaves': 160,


    "min_data_in_leaf": 500,
    #

    'max_bin': 255,
    'bagging_fraction': 0.6,
    # 'feature_fraction': 0.9,

    # "min_child_weight": 5,
    #
    # 'lambda_l1': 0.8,
    # 'lambda_l2': 0.8,
    'num_threads':3,
    'verbose': -1
}
modelC = lgb.train(params,
                  train_set=d_train,
                  num_boost_round=1000,
                  valid_sets=watchlist,
                  early_stopping_rounds=50,
                  verbose_eval=500)

pre_modelC_train = modelC.predict(df_train)
pre_modelC_test = modelC.predict(df_test)
# fbeta(y_test, tmp)
#
preds_modelC_train = [1 if i > 0.15 else 0 for i in pre_modelC_train]      #0.9639
sum(preds_modelC_train)

preds_modelC_test = [1 if i > 0.15 else 0 for i in pre_modelC_test]      #0.9639
sum(preds_modelC_test)

df_train['preds_modelC']=preds_modelC_train
df_test['preds_modelC']=preds_modelC_test

df_train2=df_train[df_train['preds_modelC']==1] #回归模型的train
df_train2.drop('preds_modelC',axis=1, inplace=True)
print('回归模型的train',df_train2.shape)
df_test2=df_test[df_test['preds_modelC']==1] #回归模型的验证集
df_test2.drop('preds_modelC',axis=1, inplace=True)
print('回归模型的验证集',df_test2.shape)

y_train2=[]
for i, s in enumerate(preds_modelC_train):
    if s==1:
        y_train2.append(y_train[i])
y_test2=[]
y_test2_0=[]
for i, s in enumerate(preds_modelC_test):
    if s==1:
        y_test2.append(y_test[i])
    else:
        y_test2_0.append(y_test[i])

with open('./temp_file/df_train2.pickle','wb') as file1:
    pickle.dump(df_train2, file1)
with open('./temp_file/df_valid2.pickle','wb') as file2:
    pickle.dump(df_test2, file2)
with open('./temp_file/y_train2.pickle','wb') as file3:
    pickle.dump(y_train2, file3)
with open('./temp_file/y_test2.pickle','wb') as file4:
    pickle.dump(y_test2, file4)

print('pickle bingo')

d_train2 = lgb.Dataset(df_train2, label=y_train2, categorical_feature=list(cat_cols),free_raw_data=False)
d_valid2 = lgb.Dataset(df_test2, label=y_test2, categorical_feature=list(cat_cols), reference=d_train2,free_raw_data=False)
watchlist2 = [d_valid2]



params = {
    'learning_rate': 0.01,
    'application': 'regression',
    'metric': 'RMSE',

    'max_depth': 12,
    'num_leaves': 150,

    'max_bin': 150,
    "min_data_in_leaf": 200,

    'bagging_fraction': 0.9,
    'feature_fraction': 0.9,

    # "min_child_samples": 10,
    "min_child_weight": 1,
    'num_threads': 3,
    # 'lambda_l1': 0.8,
    # 'lambda_l2': 0.8,

    'verbose': -1
}
modelR = lgb.train(params,
                  train_set=d_train2,
                  num_boost_round=5000,
                  valid_sets=watchlist2,
                  early_stopping_rounds=50,
                  verbose_eval=100)

test_pred2 = modelR.predict(df_test2)

score_R=mean_squared_error(y_test2,test_pred2)
score_C=mean_squared_error(y_test2_0,np.zeros(len(y_test2_0)))
score=np.sqrt((score_R*len(y_test2)+score_C*len(y_test2_0)) /(len(y_test2)+len(y_test2_0)))



#阈值0.2 0.2213




