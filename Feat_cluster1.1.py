import pandas as pd
import numpy as np
import gc

from gensim.models import Word2Vec
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


feats_to_proc = ['region', 'city', 'parent_category_name','param_1', 'param_2', 'param_3']

target_c = 'deal_probability'

for c in feats_to_proc:
    print('loading ['+c+'] data...')
    daset = pd.read_csv('./input/'+'train.csv', usecols=[c, target_c])
    
    daset.fillna('missing')
    # prepare w2v sentences with target histories
    gp = daset.loc[: , [c,target_c]].groupby(c)
    #hist = gp.agg(lambda x: ''.join(str(x)))
    gp_index = list(gp.indices.keys())
    print('The feature ['+c,'] has {} unique lebels(including missing)'.format(len(gp_index)))
    if len(gp_index) < 50:
        print('Not too much categorical number, just omit this feature —— '+c)
        del daset
        gc.collect()
        continue

    sentences = []
    for i, cate in enumerate(gp_index):
        sentences.append([])
        for index in gp.indices[cate]:
            sentences[i].append(str(daset[target_c][index]))

    del daset
    gc.collect()

    # sentences = [x.split(' ') for x in hist.values]
    #sentences = [re.sub(r's\+','') for s in hist.values]

    print('word2vec model building begins...')
    n_features = 200
    w2v = Word2Vec(sentences=sentences, min_count=1, size=n_features)

    def transform_to_matrix(sentences, model, num_features):

        def _average_word_vectors(words, model, vocabulary, num_features):
            ### 将句子的每个词的词向量求和平均
            feature_vector = np.zeros((num_features,), dtype='float64')
            n_words = 0.
            for word in words:
                if word in vocabulary:
                    n_words = n_words + 1.
                    feature_vector = np.add(feature_vector, model[word])

            if n_words:
                feature_vector = np.divide(feature_vector, n_words)
            return feature_vector

        vocab = set(model.wv.index2word)
        ### 每个向量代表一个特征feature
        feats = [_average_word_vectors(s, model, vocab, num_features) for s in sentences]
        return csr_matrix(np.array(feats))
        
    print('Transform to matrix begins...')
    w2v_matrix = transform_to_matrix(sentences, w2v, n_features)
    del sentences, w2v
    gc.collect()

    # clustering
    n_min = min(int(np.sqrt(len(gp_index))), len(gp_index)//10)
    n_max = max(int(np.sqrt(len(gp_index))), len(gp_index)//10)
    if n_max - n_min > 100:
        n_clusts = range(n_min, n_max, 3)
    elif n_max - n_min > 30:
        n_clusts = range(n_min, n_max, 2)
    else:
        n_clusts = range(n_min, n_max)
    score_best = -float('inf')
    for n in n_clusts:
        print('Kmeans model begins with n_cluster {}'.format(n))
        kmean_model = KMeans(n_clusters=n, n_jobs=2, verbose=0)
        kmean_model.fit(w2v_matrix)
        cluster_labels = kmean_model.predict(w2v_matrix)
        score_curr = silhouette_score(w2v_matrix, cluster_labels)
        print('si Score:', score_curr, 'for feature ' +c, 'with N: {}'.format(n))
        if score_curr > score_best:
            score_best = score_curr
            label_best = cluster_labels

    ### 暂时保存
    pd.Series(label_best, name=c + '_cluster', index=gp_index).to_csv(c+'_cluster.csv')
    del label_best, cluster_labels, w2v_matrix
    gc.collect()
