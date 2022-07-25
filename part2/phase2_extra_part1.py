from __future__ import unicode_literals
from itertools import combinations
import pandas as pd
from hazm import *
from hazm import stopwords_list
import math
import operator
import numpy as np
import multiprocessing
import time
import numpy as np
from numpy.linalg import norm
from gensim.models import Word2Vec

def pre_proccessing():
    df = pd.read_excel('IR1_7k_news.xlsx', engine='openpyxl')
    dict_tokens = {}
    normalizer = Normalizer()
    for i in range(df.shape[0]):
        df.iloc[i].content = normalizer.normalize(df.iloc[i].content)
        dict_tokens[i] = word_tokenize(df.iloc[i].content)
    stop_words_list = stopwords_list()
    for i in range(df.shape[0]):
        dict_tokens[i] = [x for x in dict_tokens[i] if x not in stop_words_list]
    lemmatizer = Lemmatizer()
    stemmer = Stemmer()
    for i in range(len(dict_tokens)):
        for j in range(len(dict_tokens[i])):
            dict_tokens[i][j] = stemmer.stem(dict_tokens[i][j])
            dict_tokens[i][j] = lemmatizer.lemmatize(dict_tokens[i][j])
    return dict_tokens, df


def create_positional_index(dict_tokens):
    PL = {}
    dict_index = 0
    k = 1
    for i in range(len(dict_tokens)):
        for j in range(len(dict_tokens[i])):
            if dict_tokens[i][j] not in PL.keys():
                dict_index += 1
                PL[dict_tokens[i][j]] = {'total_count': 1, 'DOCID': {i: [k, j]}}
            else:
                PL[dict_tokens[i][j]]['total_count'] += 1
                if i in PL[dict_tokens[i][j]]['DOCID'].keys():
                    PL[dict_tokens[i][j]]['DOCID'][i].append(j)
                    PL[dict_tokens[i][j]]['DOCID'][i][0] += 1
                else:
                    PL[dict_tokens[i][j]]['DOCID'][i] = [k, j]

    return PL


def add_tfidf(pl):
    keys = pl.keys()
    #     print(keys)
    N = len(pl)
    count = 0
    for i in (keys):
        count += 1
        nt = len(pl[i]['DOCID'])
        doc_keys = pl[i]['DOCID'].keys()
        for j in (doc_keys):
            term_frequency_per_doc = pl[i]['DOCID'][j][0]
            tfidf = (1 + math.log(term_frequency_per_doc, 10)) * math.log(N / nt, 10)
            pl[i]['DOCID'][j].insert(1, tfidf)

    return pl


def normalization(pl, max_id):
    for idd in range(max_id):

        weights = []
        for key in pl.keys():
            if idd in pl[key]['DOCID']:
                weights.append(pl[key]['DOCID'][idd][1])
        vector = np.array(weights)
        normalized_v = vector / np.linalg.norm(vector)
        count = 0
        for key in pl.keys():
            if idd in pl[key]['DOCID']:
                pl[key]['DOCID'][idd][1] = normalized_v[count]
                count += 1

    return pl


def create_doc_tf_idf(pl, dt):
    dict_tf_idf = {}
    for i in range(len(dt)):
        dict_tf_idf[str(i)] = {}
        for key in dt[i]:
            dict_tf_idf[str(i)][key] = pl[key]['DOCID'][i][1]

    return dict_tf_idf


def similarity(doc1, doc2):
    similarity = np.dot(doc1, doc2) / (norm(doc1) * norm(doc2))
    return (similarity + 1) * 0.5


def query_handling(pl):
    query = input("enter your query:  ")
    query = normalizer.normalize(query)
    word_before_stop = word_tokenize(query)
    stop_words_list = stopwords_list()
    word_tokens = [x for x in word_before_stop if x not in stop_words_list]
    query_pl = {}

    for i in range(len(word_tokens)):
        word_tokens[i] = stemmer.stem(word_tokens[i])
        word_tokens[i] = lemmatizer.lemmatize(word_tokens[i])
    print(word_tokens)
    for i in range(len(word_tokens)):
        if word_tokens[i] not in query_pl.keys():
            query_pl[word_tokens[i]] = [1, i]
        else:
            query_pl[word_tokens[i]][0] += 1
            query_pl[word_tokens[i]].append(i)
    keys = pl.keys()
    N = len(pl)
    used = set()
    unique = [x for x in word_tokens if x not in used and (used.add(x) or True)]
    for key in (unique):
        nt = len(pl[key]['DOCID'])
        tfidf = (1 + math.log(query_pl[key][0], 10)) * math.log(N / nt, 10)
        query_pl[key].insert(1, tfidf)

    return query_pl
if __name__ == "__main__":
    word_before_stop = []
    word_tokens = []
    normalizer = Normalizer()
    lemmatizer = Lemmatizer()
    stemmer = Stemmer()
    dict_tokens, dataframe = pre_proccessing()
    positional_index = create_positional_index(dict_tokens)
    postional_index_tfidf = add_tfidf(positional_index)
    postional_index_tfidf = normalization(postional_index_tfidf, dataframe.shape[0])
    doc_tf_idf = {}
    doc_tf_idf = create_doc_tf_idf(postional_index_tfidf, dict_tokens)
    cores = multiprocessing.cpu_count()
    print('Number of cores : ', cores)
    docs_number = len(dict_tokens)
    token_number = len(positional_index)
    training_data = []
    for i in dict_tokens.keys():
        training_data.append(dict_tokens[i])
    print('Number of docs : ', docs_number)
    print('Number of tokens : ', token_number)
    w2v_model = Word2Vec(min_count=1, window=5, vector_size=300, alpha=0.03, workers=cores - 1)
    w2v_model.build_vocab(training_data)
    w2v_model_vocab_size = len(w2v_model.wv)
    print('vocab size: ', w2v_model_vocab_size)
    start = time.time()
    w2v_model.train(training_data, total_examples=w2v_model.corpus_count, epochs=20)
    end = time.time()
    print("{} s".format(end - start))
    w2v_model.save('w2v_model_tfidf.model')

    for key in doc_tf_idf.keys():
        v = list(doc_tf_idf[key].values())
        normalized_v = v / np.linalg.norm(v)
        count = 0
        for i in doc_tf_idf[key].keys():
            doc_tf_idf[key][i] = normalized_v[count]
            count += 1
    docs_embedding = []
    for doc in doc_tf_idf:
        doc_vec = np.zeros(300)
        weights_sum = 0
        for token, weight in doc_tf_idf[doc].items():
            doc_vec += w2v_model.wv[token] * weight
            weights_sum += weight
        docs_embedding.append(doc_vec / weights_sum)
    while True :
        query_list = query_handling(postional_index_tfidf)
        query_tfidf_dict = {}
        for key in query_list.keys():
            query_tfidf_dict[key] = query_list[key][1]

        doc_vec = np.zeros(300)
        weights_sum = 0
        for token, weight in query_tfidf_dict.items():
            doc_vec += w2v_model.wv[token] * weight
            weights_sum += weight
        query_embedding = doc_vec / weights_sum
        similarity_dict = {}

        for i in range(len(docs_embedding)):
            similarity_dict[i] = (similarity(docs_embedding[i], query_embedding))
        similarity_dict_sorted = dict(sorted(similarity_dict.items(), key=operator.itemgetter(1), reverse=True))
        count = 0
        for doc_id in similarity_dict_sorted.keys():
            if count == 10:
                break
            else:
                print(doc_id)
                print(dataframe.iloc[doc_id]['title'])
                print(dataframe.iloc[doc_id]['content'])
                print("============")
                count += 1



