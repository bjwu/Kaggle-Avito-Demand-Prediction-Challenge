import pandas as pd
import numpy as np
import gensim
import re
import pickle
import gc

from gensim.models.doc2vec import Doc2Vec, LabeledSentence
from nltk.corpus import stopwords
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN
from sklearn.metrics import calinski_harabaz_score, silhouette_score

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

############################ Selection ####################
text_preproc = True
train_d2v_model = True

save_copy = True

feature = 'title'
filename = 'train.csv'

kmeans = False
dbscan = False

if text_preproc:
    print('loading...')
    data_train = pd.read_csv('./input/'+filename, usecols=[feature])
    data_test = pd.read_csv('./input/'+'test.csv', usecols=[feature])
    data = pd.concat([data_train, data_test], axis=0, ignore_index=True)
    del data_test, data_train
    gc.collect()
    #
    # class MyCorpus(object):
    #     def __iter__(self):
    #         for line in data:
    #             pass
    #
    print('Missing value filling')
    data = data.fillna('missing')

    def clean_str(org):
        stri = re.sub(r'\W+', ' ', org).lower().split()
        stri_proc = [w for w in stri if w not in stopwords.words('russian')]
        return stri_proc

    texts = data[feature].apply(clean_str).tolist()
    print('Cleaning data...')
    if save_copy:
        file = open('./input/title_preproc.pickle','wb')
        pickle.dump(texts, file)
        file.close()

else:

    with open('./input/title_preproc.pickle', 'rb') as file:
        texts = pickle.load(file)


### doc2vec preprocessing
TaggededDocument = gensim.models.doc2vec.TaggedDocument
x_train = []
for i, text in enumerate(texts):
    document = TaggededDocument(text, tags=[i])
    x_train.append(document)
del texts
gc.collect()

if train_d2v_model:
    d2v_model = Doc2Vec(x_train, min_count=1, window = 3, size= 100, sample=1e-3, negative=5, workers=4)
    infered_vectors_list = []
    for text, label in x_train:
        vector = d2v_model.infer_vector(text)
        infered_vectors_list.append(vector)
    if save_copy:
        d2v_model.save('d2v_model1.0.model')
        file = open('./input/title_before_kmeans.pickle', 'wb')
        pickle.dump(infered_vectors_list, file)
        file.close()
    d2v_model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
else:
    with open('./input/title_before_kmeans.pickle', 'rb') as file:
        infered_vectors_list = pickle.load(file)

# kmean_model = KMeans(n_clusters=100, n_jobs=1, verbose=100)
# kmean_model.fit(infered_vectors_list)
# labels = kmean_model.predict(infered_vectors_list)
if kmeans:
    print('Mini Kmeans model building begins...')
    for index, k in enumerate((2, 4, 10, 20, 45)):
        minikmeans_model = MiniBatchKMeans(n_clusters=k, batch_size=256, verbose=0, reassignment_ratio= 0.003, max_no_improvement=10)
        minikmeans_model.fit(infered_vectors_list)
        labels = minikmeans_model.predict(infered_vectors_list)
        #ch_score = calinski_harabaz_score(infered_vectors_list, labels)
        si_score = silhouette_score(infered_vectors_list, labels)
        #print('CH Score:', ch_score, '[n_clusters]:', k)
        print('si Score:', si_score, '[n_clusters]:', k)

# file = open('./input/title_cluster_label.pickle','wb')
# pickle.dump(labels, file)
# file.close()
#
# cluster_centers = minikmeans_model.cluster_centers_
if dbscan:
    db = DBSCAN().fit(infered_vectors_list)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    file = open('./input/dbscan_labels.pickle', 'wb')
    pickle.dump(labels, file)
    file.close()

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print('Estimated number of clusters: %d' % n_clusters_)
