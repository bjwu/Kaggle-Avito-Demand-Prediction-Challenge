import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from nltk.corpus import stopwords

import re

from keras.preprocessing.text import Tokenizer
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from gensim.models.doc2vec import Doc2Vec,LabeledSentence

from keras.layers import InputSpec, Layer, Input, Dense, merge, Conv1D
from keras.layers import Lambda, Activation, Dropout, Embedding, TimeDistributed
from keras.layers.pooling import GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras import metrics
import keras.backend as K
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
from sklearn.cross_validation import KFold
############################ Selection ####################
train_w2v_model = True

feature = 'title'
filename = 'train.csv'

# 处理的最大单词数量
MAX_NUM_WORDS = 5000
# 序列的最大长度, 大于此长度的序列将被截短，小于此长度的序列将在后部填0.
MAX_TEXT_LENGTH = 100

# 特征向量维度
FEAT_VEC_SIZE = 100
#
# ###########################################################
# if train_w2v_model:
#     data_train = pd.read_csv('../input/'+filename, usecols=[feature])
#     data_test = pd.read_csv('../input/' + 'test.csv', usecols=[feature])
#     data=pd.concat([data_train, data_test], axis=0, ignore_index=True)
#     # class MyCorpus(object):
#     #     def __iter__(self):
#     #         for line in data:
#     #             pass
#     #
#
#
#
#
#     model_org = Word2Vec(texts, size=100, window=5, min_count=5, workers=4)
#     model_org.save('w2v_model1.0.model')
train =  pd.read_csv('../input/'+filename, usecols=['title','deal_probability'])
data_x = train['title'].fillna('missing')
# data_y = data["deal_probability"]


def clean_str(org):
        stri = re.sub(r'\W+', ' ', org).lower().split()
        stri_proc = [w for w in stri if w not in stopwords.words('russian')]
        return stri_proc
texts = train[feature].apply(clean_str).tolist()

model = Word2Vec.load('w2v_model1.0.model')
word2vec = model.wv
print('Found %s word vectors of word2vec' % len(word2vec.vocab))
# model.vector 返回所有的词向量

# dictionary = corpora.Dictionary(texts)
# dictionary.save('/tmp/deerwester.dict')

#将一个句子拆分成单词构成的列表
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)

#序列的列表
sequences = tokenizer.texts_to_sequences(texts)
#单词字典
word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_TEXT_LENGTH)

embedding_matrix = np.zeros((MAX_NUM_WORDS, FEAT_VEC_SIZE))
for word, i in word_index.items():
    if word in model.wv and i < MAX_NUM_WORDS:
        embedding_matrix[i] = model.wv.word_vec(word)
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))


labels = train['deal_probability'].values


def build_model(emb_matrix, max_sequence_length):
    # The embedding layer containing the word vectors
    emb_layer = Embedding(
        input_dim=emb_matrix.shape[0],
        output_dim=emb_matrix.shape[1],
        weights=[emb_matrix],
        input_length=max_sequence_length,
        trainable=False
    )

    # 1D convolutions that can iterate over the word vectors
    conv1 = Conv1D(filters=128, kernel_size=1, padding='same', activation='relu')
    conv2 = Conv1D(filters=128, kernel_size=2, padding='same', activation='relu')
    conv3 = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')
    conv4 = Conv1D(filters=128, kernel_size=4, padding='same', activation='relu')
    conv5 = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')
    conv6 = Conv1D(filters=32, kernel_size=6, padding='same', activation='relu')
    conv7 = Conv1D(filters=32, kernel_size=7, padding='same', activation='relu')
    conv8 = Conv1D(filters=32, kernel_size=8, padding='same', activation='relu')

    # Define inputs
    seq = Input(shape=(max_sequence_length,))

    # Run inputs through embedding
    emb = emb_layer(seq)

    # Run through CONV + GAP layers
    conv1 = conv1(emb)
    glob1 = GlobalAveragePooling1D()(conv1)

    conv2 = conv2(emb)
    glob2 = GlobalAveragePooling1D()(conv2)

    conv3 = conv3(emb)
    glob3 = GlobalAveragePooling1D()(conv3)

    conv4 = conv4(emb)
    glob4 = GlobalAveragePooling1D()(conv4)

    conv5 = conv5(emb)
    glob5 = GlobalAveragePooling1D()(conv5)

    conv6 = conv6(emb)
    glob6 = GlobalAveragePooling1D()(conv6)

    conv7 = conv7(emb)
    glob7 = GlobalAveragePooling1D()(conv7)

    conv8 = conv8(emb)
    glob8 = GlobalAveragePooling1D()(conv8)

    merge = concatenate([glob1, glob2, glob3, glob4, glob5, glob6, glob7, glob8])

    # The MLP that determines the outcome
    x = Dropout(0.1)(merge)
    x = BatchNormalization()(x)
    #    x = Dense(1500, activation='relu')(x)
    x = Dense(32, activation='relu')(x)

    x = Dropout(0.1)(x)
    x = BatchNormalization()(x)
    #    pred = Dense(1999, activation='softmax')(x)
    pred = Dense(units=1, activation='linear')(x)

    model = Model(inputs=seq, outputs=pred)
    model.compile(loss="mse", optimizer="adam", metrics=["mae", score])
    model.summary()

    return model

def score(y_true, y_pred):
    return 1.0 / (1.0 + K.sqrt(K.mean(K.square(y_true - y_pred), axis=-1)))


nfolds = 5
folds = KFold(data.shape[0], n_folds=nfolds, shuffle=True, random_state=2017)
pred_results = []


def batch_generator(X, y, batch_size, shuffle):
    number_of_batches = np.ceil(X.shape[0] / batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = X[batch_index]
        y_batch = y[batch_index, :].toarray()
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0

train_scores = []
vali_scores = []
for curr_fold, (idx_train, idx_val) in enumerate(folds):
    data_train = data[idx_train]
    y_train = labels[idx_train]

    data_val = data[idx_val]
    y_val = labels[idx_val]

    model1 = build_model(embedding_matrix, data.shape[1])
#    batch_size = 128
#    epochs = 20
    early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=2)

    history = model1.fit(data_train, y_train,
                      batch_size=128,
                      epochs=10,
                      validation_data=(data_val, y_val))
    trainScore = history.history['loss'][-1]
    valiScore = history.history['val_loss'][-1]
    train_scores.append(trainScore)
    vali_scores.append(valiScore)


    print('Fold: %d' % (curr_fold))
    print('Train Score: %.4f rmse' % (trainScore))
    print('Vali Score: %.4f rmse' % (valiScore))


mean_trainScore = np.mean(train_scores)
mean_valiScore = np.mean(vali_scores)
print('Mean train score: %.4f rmse' %(mean_trainScore))
print('Mean vali score: %.4f rmse' % (mean_valiScore))



