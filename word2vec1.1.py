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

############################ Selection ####################
train_w2v_model = True

feature = 'title'
filename = 'train_sample5k.csv'

# 处理的最大单词数量
MAX_NUM_WORDS = 5000

# 序列的最大长度, 大于此长度的序列将被截短，小于此长度的序列将在后部填0.
MAX_TEXT_LENGTH = 100

# 特征向量维度
FEAT_VEC_SIZE = 200

###########################################################


if train_w2v_model:
    data = pd.read_csv('./input/'+filename, usecols=[feature])
    #
    # class MyCorpus(object):
    #     def __iter__(self):
    #         for line in data:
    #             pass
    #
    data = data.fillna('missing')

    def clean_str(org):
        stri = re.sub(r'\W+', ' ', org).lower().split()
        stri_proc = [w for w in stri if w not in stopwords.words('russian')]
        return stri_proc

    texts = data[feature].apply(clean_str).tolist()

    model_org = Word2Vec(texts, size=FEAT_VEC_SIZE, window=5, min_count=5, workers=4)
    #model_org.save('w2v_model1.0.model')

#model = Word2Vec.load('w2v_model1.0.model')
model = model_org


# model.vector 返回所有的词向量

# dictionary = corpora.Dictionary(texts)
# dictionary.save('/tmp/deerwester.dict')

# num_words: the maximum number of words to keep, based on word frequency. Only the most common num_words words will be kept.
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)


#将一个句子拆分成单词id构成的列表
sequences = tokenizer.texts_to_sequences(texts)
#单词字典
word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

# 填充sequences, 作为最后模型训练数据
traindes = pad_sequences(sequences, maxlen=MAX_TEXT_LENGTH)

embedding_matrix = np.zeros((MAX_NUM_WORDS, FEAT_VEC_SIZE))

for word, i in word_index.items():
    if word in model.wv and i < MAX_NUM_WORDS:
        embedding_matrix[i] = model.wv.word_vec(word)
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))




