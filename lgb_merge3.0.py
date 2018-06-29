
### 采用merge2.2的数据，使用fastFM 的模型做尝试

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
import pickle
import matplotlib.pyplot as plt
random.seed(2018)

from nltk.corpus import stopwords
from fastFM import als
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


if data_preproc:

    ######################## Loading data #################################

    train = pd.read_csv('./input/train.csv', index_col = "item_id", parse_dates = ["activation_date"])
    test = pd.read_csv('./input/test.csv', index_col = "item_id", parse_dates = ["activation_date"])
    print('[{}] Finished load data'.format(time.time() - start_time))



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

    # print("\nCreate Time Variables")
    # merge["activation_weekday"] = merge['activation_date'].dt.weekday
    # merge["Weekd_of_Year"] = merge['activation_date'].dt.week
    # merge["Day_of_Month"] = merge['activation_date'].dt.day

    print(merge.head(5))
    gc.collect()

    merge.drop(["activation_date", "image", "user_id"],axis=1,inplace=True)



    # ################# Feature from text ##############################
    # print("\nText Features")
    # textfeats = ["description", "title", "param_1_copy"]
    #
    # for cols in textfeats:
    #     merge[cols] = merge[cols].astype(str)
    #     merge[cols] = merge[cols].astype(str).fillna('missing') # FILL NA
    #     merge[cols] = merge[cols].str.lower() # Lowercase all text, so that capitalized words dont get treated differently
    #     merge[cols + '_num_stopwords_en'] = merge[cols].apply(lambda x: len([w for w in x.split() if w in stopwords_en]))  # Count number of Stopwords
    #     merge[cols + '_num_stopwords'] = merge[cols].apply(lambda x: len([w for w in x.split() if w in stopwords_russian])) # Count number of Stopwords
    #     merge[cols + '_num_punctuations'] = merge[cols].apply(lambda comment: (comment.count(RE_PUNCTUATION))) # Count number of Punctuations
    #     merge[cols + '_num_alphabets'] = merge[cols].apply(lambda comment: (comment.count(r'[a-zA-Z]'))) # Count number of Alphabets
    #     merge[cols + '_num_alphanumeric'] = merge[cols].apply(lambda comment: (comment.count(r'[A-Za-z0-9]'))) # Count number of AlphaNumeric
    #     merge[cols + '_num_digits'] = merge[cols].apply(lambda comment: (comment.count('[0-9]'))) # Count number of Digits
    #     merge[cols + '_num_letters'] = merge[cols].apply(lambda comment: len(comment)) # Count number of Letters
    #     merge[cols + '_num_words'] = merge[cols].apply(lambda comment: len(comment.split())) # Count number of Words
    #     merge[cols + '_num_unique_words'] = merge[cols].apply(lambda comment: len(set(w for w in comment.split())))
    #     merge[cols + '_words_vs_unique'] = merge[cols+'_num_unique_words'] / merge[cols+'_num_words'] # Count Unique Words
    #     merge[cols + '_letters_per_word'] = merge[cols+'_num_letters'] / merge[cols+'_num_words'] # Letters per Word
    #     merge[cols + '_punctuations_by_letters'] = merge[cols+'_num_punctuations'] / merge[cols+'_num_letters'] # Punctuations by Letters
    #     merge[cols + '_punctuations_by_words'] = merge[cols+'_num_punctuations'] / merge[cols+'_num_words'] # Punctuations by Words
    #     merge[cols + '_digits_by_letters'] = merge[cols+'_num_digits'] / merge[cols+'_num_letters'] # Digits by Letters
    #     merge[cols + '_alphanumeric_by_letters'] = merge[cols+'_num_alphanumeric'] / merge[cols+'_num_letters'] # AlphaNumeric by Letters
    #     merge[cols + '_alphabets_by_letters'] = merge[cols+'_num_alphabets'] / merge[cols+'_num_letters'] # Alphabets by Letters
    #     merge[cols + '_stopwords_by_letters'] = merge[cols+'_num_stopwords'] / merge[cols+'_num_letters'] # Stopwords by Letters
    #     merge[cols + '_stopwords_by_words'] = merge[cols+'_num_stopwords'] / merge[cols+'_num_words'] # Stopwords by Letters
    #     merge[cols + '_stopwords_by_letters_en'] = merge[cols+'_num_stopwords_en'] / merge[cols+'_num_letters'] # Stopwords by Letters
    #     merge[cols + '_stopwords_by_words_en'] = merge[cols+'_num_stopwords_en'] / merge[cols+'_num_words'] # Stopwords by Letters
    #     merge[cols + '_mean'] = merge[cols].apply(lambda x: 0 if len(x) == 0 else float(len(x.split())) / len(x)) * 10 # Mean
    #     merge[cols + '_num_sum'] = merge[cols].apply(sum_numbers)
    #     print(cols +' Feature done')
    #
    # # Extra Feature Engineering
    # merge['title_desc_len_ratio'] = merge['title_num_letters']/(merge['description_num_letters']+1)
    # merge['title_param1_len_ratio'] = merge['title_num_letters']/(merge['param_1_copy_num_letters']+1)
    # merge['param_1_copy_desc_len_ratio'] = merge['param_1_copy_num_letters']/(merge['description_num_letters']+1)
    #
    # print('[{}] Getting feature from text'.format(time.time() - start_time))
    # gc.collect()

    ############### Set columns type ##############################

    cols = set(merge.columns.values)
    cat_cols = ["region","city","parent_category_name","category_name","user_type","image_top_1","param_2","param_3"]
    basic_cols = ["region","city","parent_category_name","category_name","user_type","image_top_1",
                   "description","title","param_1","param_2","param_3", "price", "item_seq_number"]
    text_cols = ["description","title","param_1"]
    drop_clos = ['param_1','title','description']
    # num_features = list(cols - (basic_cols)).remove('is_train')

    ############### Label Encode ##############################
    print("----------LabelEncode--------")
    print("Encoding :", cat_cols)
    lbl = LabelEncoder()
    for col in cat_cols:
        merge[col] = lbl.fit_transform(merge[col].astype(str))

    print('[{}] Label Encode.'.format(time.time() - start_time))

    ############### Onehot Encode ##############################
    print("----------Onehot Encode--------")
    print("One-hot Encoding :", cat_cols)
    ohe = OneHotEncoder(sparse=True)
    oh_sparse_merge = ohe.fit_transform(merge[cat_cols])

    print('[{}] Label Encode.'.format(time.time() - start_time))

    ############### TFIDF ######################
    for text_col in text_cols:
        merge[text_col].fillna('missing', inplace=True)
    tv = TfidfVectorizer(max_features=80000,
                         ngram_range=(1, 2),
                         )
    X_name = tv.fit_transform(merge['title'])
    print('[{}] Finished TFIDF vectorize `title`'.format(time.time() - start_time))

    tv = TfidfVectorizer(max_features=100000,
                         ngram_range=(1, 2),
                        )
    X_description = tv.fit_transform(merge['description'])
    print('[{}] Finished TFIDF vectorize `description` '.format(time.time() - start_time))

    tv = TfidfVectorizer(max_features=50000,
                         ngram_range=(1, 2),
                         stop_words=None)
    X_param1 = tv.fit_transform(merge['param_1'])
    print('[{}] Finished TFIDF vectorize `param_1`'.format(time.time() - start_time))

    tfidf_sparse_merge = hstack((X_description, X_param1, X_name))
    del X_description, X_param1, X_name
    gc.collect()
    print('[{}] Create sparse merge train completed'.format(time.time() - start_time))


    sparse_merge = hstack((oh_sparse_merge, tfidf_sparse_merge)).tocsr()

    ################# Split train and test data ###############################
    all_train = sparse_merge[:nrow_train]
    all_test = sparse_merge[nrow_train:]
    all_y_train = y

    print(all_train.shape)
    print(all_y_train.shape)


    ################## Features Merge and Drop########################

    all_train.drop(drop_clos, axis=1, inplace=True)
    all_test.drop(drop_clos, axis=1, inplace=True)

    all_train.to_csv('./temp_file/train_data_before_model_0625.csv')
    all_test.to_csv('./temp_file/test_data_before_model_0625.csv')
    all_y_train.to_csv('./temp_file/y_train_before_model_0625.csv')
else:
    all_train = pd.read_csv('./temp_file/train_data_before_model_0625.csv', index_col = "item_id")
    all_test = pd.read_csv('./temp_file/test_data_before_model_0625.csv', index_col = "item_id")
    all_y_train = pd.read_csv('./temp_file/y_train_before_model_0625.csv', names=['item_id', 'deal_probability'], index_col=0 )['deal_probability']
    cat_cols = {"region", "city", "parent_category_name", "category_name", "user_type", "image_top_1", "param_2", "param_3"}

# print('Add external files')
# train_imgtop_jh = pd.read_csv('./temp_file/train_image_top_1_features.csv',index_col="item_id", squeeze=True)
# test_imgtop_jh = pd.read_csv('./temp_file/test_image_top_1_features.csv',index_col="item_id", squeeze=True)
# train_des_w2vRnn_jh = pd.read_csv('./temp_file/train_description_word2vec_RNN3.0.csv', index_col="item_id", squeeze=True)
# test_des_w2vRnn_jh = pd.read_csv('./temp_file/test_description_word2vec_RNN3.0.csv', index_col="item_id", squeeze=True)
#
# print('merge external files to df')
# all_train['img_top_complete'] = train_imgtop_jh
# all_train['des_w2vRnn'] = train_des_w2vRnn_jh
# all_test['img_top_complete'] = test_imgtop_jh
# all_test['des_w2vRnn'] = test_des_w2vRnn_jh
# submission = pd.DataFrame(all_test.index)

################## LightGBM #############################

print('Modeling begins...')

fm = als.FMRegression(n_iter=1000, init_stdev=0.1, rank=2)
fm.fit(all_train, all_y_train)



