from __future__ import unicode_literals
from itertools import combinations
import pandas as pd
from hazm import *
from numpy.linalg import norm
from hazm import stopwords_list
import math
import operator
import numpy as np
import pickle
import random
import copy


def pre_proccessing():
    df1 = pd.read_excel('IR00_3_11k News.xlsx', engine='openpyxl')
    df2 = pd.read_excel('IR00_3_17k News.xlsx', engine='openpyxl')
    df3 = pd.read_excel('IR00_3_20k News.xlsx', engine='openpyxl')
    pdList = [df1, df2, df3]  # List of your dataframes
    df = pd.concat(pdList)
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


def return_df():
    df1 = pd.read_excel('IR00_3_11k News.xlsx', engine='openpyxl')
    df2 = pd.read_excel('IR00_3_17k News.xlsx', engine='openpyxl')
    df3 = pd.read_excel('IR00_3_20k News.xlsx', engine='openpyxl')
    pdList = [df1, df2, df3]  # List of your dataframes
    df = pd.concat(pdList)
    return df


def create_positional_index(dict_tokens):
    PL = {}
    dict_index = 0
    k = 1
    for i in range(len(dict_tokens)):
        for j in range(len(dict_tokens[i])):
            if dict_tokens[i][j] not in PL.keys():
                dict_index += 1
                PL[dict_tokens[i][j]] = {'total_count': 1, 'DOCID': {i: [k]}}
            else:
                PL[dict_tokens[i][j]]['total_count'] += 1
                if i in PL[dict_tokens[i][j]]['DOCID'].keys():
                    PL[dict_tokens[i][j]]['DOCID'][i][0] += 1
                else:
                    PL[dict_tokens[i][j]]['DOCID'][i] = [k]

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


def pre_proccessing_test():
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
def return_df ():
    df1 = pd.read_excel('IR00_3_11k News.xlsx', engine='openpyxl')
    df2 = pd.read_excel('IR00_3_17k News.xlsx', engine='openpyxl')
    df3 = pd.read_excel('IR00_3_20k News.xlsx', engine='openpyxl')
    pdList = [df1, df2, df3]  # List of your dataframes
    df = pd.concat(pdList)
    return df.shape[0], df
def return_df_test ():
    df = pd.read_excel('IR1_7k_news.xlsx', engine='openpyxl')
    return df.shape[0], df
def similarity(doc1, doc2):
    similarity = np.dot(doc1,doc2)/(norm(doc1)* norm(doc2))
    return similarity


def vectorization(pl, max_id):
    vector_data = {}

    for i in range(max_id):
        #         print(i)
        count = 0
        vector_data[i] = {}
        for key in pl.keys():
            if i in pl[key]['DOCID']:
                vector_data[i][count] = pl[key]['DOCID'][i][1]
            count += 1
    return vector_data


def k_means(pl, max_id, doc_vector):
    #     k_num = 3
    k_num = 5
    list_pl_keys = list(pl.keys())

    # انتخاب مراکز اولیه
    first_centroids = random.sample(range(1, max_id), k_num)

    #     first_centroids = [25000, 30000, 45000]
    flag_end = True
    updated_centroids = {1: {}, 2: {}, 3: {}, 4: {}, 0: {}}
    old_centroids = {1: {}, 2: {}, 3: {}, 4: {}, 0: {}}
    clusters_docid = {1: [], 2: [], 3: [], 4: [], 0: []}
    #     updated_centroids = {1:{} , 2:{}  , 0:{}}
    #     old_centroids = {1:{} , 2:{} ,  0:{}}
    #     clusters_docid = {1:[] , 2:[], 0:[]}
    count = 0
    RSS = []
    Sum_rss = []
    # حلقه اصلی الگوریتم
    while count < 100:
        print(count)
        if count != 0:
            old_centroids = copy.deepcopy(updated_centroids)
            # برای بار اول
        if count == 0:
            # به ازای تمام داک های دیتاست
            for j in range(max_id):
                print(j)
                distance = {}
                countt = 0
                # این هم لیست ترم های داک های دیتاست
                doc_key = list(doc_vector[j].keys())
                # به ازای تمام مراکز اولیه
                for i in (first_centroids):
                    # میایم تمام کلید های دیکشنری ینی تمام ترم های داک رو جدا میکنیم تو یک لیست برای مرکز
                    centroid_key = list(doc_vector[i].keys())
                    if i != j:
                        # المنت های مشترک که کم کنیم فاصلشونو
                        match_elements = [value for value in centroid_key if value in doc_key]
                        vector_ditance = []
                        # اینجا مشترکا کم میشن گذاشته میشن تو لیست
                        for k in match_elements:
                            vector_ditance.append(doc_vector[i][k] - doc_vector[j][k])
                        # اینجا کلماتی که توی داک هستند
                        for k in doc_key:
                            if k not in match_elements:
                                vector_ditance.append(doc_vector[j][k])
                        # اینجا هم کلمات مرکز
                        for k in centroid_key:
                            if k not in match_elements:
                                vector_ditance.append(doc_vector[i][k])
                                # فاصله رو حساب میکنیم
                        vector_ditance = np.array(vector_ditance)
                        distance[countt] = np.sqrt(np.sum(np.square(vector_ditance)))
                        countt += 1
                #                 RSS.append(sum(distance.values()))
                # سورت برا حسب فاصله
                sorted_d = dict(sorted(distance.items(), key=operator.itemgetter(1)))
                #                 print(sorted_d)
                sorted_list = list(sorted_d.keys())

                # اینجا هم هر داک به دوتا کلاستر نزدیکش اساین میشه
                clusters_docid[sorted_list[0]].append(j)
                clusters_docid[sorted_list[1]].append(j)
            count += 1

        #             Sum_rss.append(sum(RSS))
        else:
            #             updated_centroids = {1:{} , 2:{},  0:{}}
            updated_centroids = {1: {}, 2: {}, 3: {}, 4: {}, 0: {}}

            # برای بار به جز اول چون مراکز فرق میکنه
            RSS = []
            clusters_docid = {1: [], 2: [], 3: [], 4: [], 0: []}
            #             clusters_docid = {1:[] , 2:[] , 0:[]}
            for j in range(max_id):
                print(j)
                # همون کار با مراکز قدیم مرحله قبل
                doc_key = list(doc_vector[j].keys())
                distance = {}
                countt = 0

                for i in (old_centroids.keys()):

                    centroid_key = list(old_centroids[i].keys())

                    #                     print(len(centroid_key))

                    match_elements = [value for value in centroid_key if value in doc_key]
                    list_centroid = [i for i in centroid_key if i not in match_elements]
                    vector_ditance = []
                    for k in match_elements:
                        vector_ditance.append(old_centroids[i][k] - doc_vector[j][k])
                    for k in doc_key:
                        if k not in match_elements:
                            vector_ditance.append(doc_vector[j][k])
                    for k in list_centroid:
                        vector_ditance.append(old_centroids[i][k])

                    vector_ditance = np.array(vector_ditance)
                    distance[countt] = np.sqrt(np.sum(np.square(vector_ditance)))
                    countt += 1
                #                 RSS.append(sum(distance.values()))

                sorted_d = dict(sorted(distance.items(), key=operator.itemgetter(1)))
                sorted_list = list(sorted_d.keys())

                clusters_docid[sorted_list[0]].append(j)
                clusters_docid[sorted_list[1]].append(j)
            count += 1

        #             Sum_rss.append(sum(RSS))
        #             print(Sum_rss)

        # average
        for i in range(k_num):
            #             print(i)
            list_doc = clusters_docid[i]
            #             print(len(list_doc))
            dict_cent = {}
            for idd in list_doc:
                for j in doc_vector[idd].keys():
                    if j not in dict_cent.keys():
                        dict_cent[j] = []
                        dict_cent[j] = [1, doc_vector[idd][j]]
                    else:
                        dict_cent[j][1] += doc_vector[idd][j]
                        dict_cent[j][0] += 1
            #             print(dict_cent)
            dict_cent = dict(sorted(dict_cent.items()))

            for k in dict_cent.keys():
                updated_centroids[i][list_pl_keys[k]] = dict_cent[j][1] / dict_cent[j][0]

        if updated_centroids == old_centroids:
            break

    return clusters_docid, updated_centroids


def query_handling (pl):
    query = input("enter your query:  ")
    query = normalizer.normalize(query)
    word_before_stop = word_tokenize(query)
    stop_words_list = stopwords_list()
    used = []
    word_tokens = [x for x in word_before_stop if x not in stop_words_list]
    query_pl = {}
    for i in range(len(word_tokens)):
        word_tokens[i] = stemmer.stem(word_tokens[i])
        word_tokens[i] = lemmatizer.lemmatize(word_tokens[i])
    for i in range(len(word_tokens)):
        if word_tokens[i] not in query_pl.keys():
            query_pl[word_tokens[i]] = [1, i]
        else:
            query_pl[word_tokens[i]][0] += 1
            query_pl[word_tokens[i]].append(i)

    used = set()
    unique = [x for x in word_tokens if x not in used and (used.add(x) or True)]
    for key in (unique):
        nt = len(pl[key]['DOCID'])
        tfidf = (1 + math.log(query_pl[key][0], 10))
        query_pl[key].insert(1, tfidf)
    query_vector = []
    for key in pl.keys():
        if key in query_pl.keys():
            query_vector.append(query_pl[key][1])
        else:
            query_vector.append(0)
    return np.array(query_vector)


def check_clusters(query_vector, clusters_docid, updated_centroids, doc_vector, pl):
    centroids_vector = {1: [], 2: [], 3: [], 4: [], 0: []}
    # vazn term haye centroid ha
    for i in range(len(updated_centroids)):
        for key in pl.keys():
            if key in updated_centroids[i].keys():
                centroids_vector[i].append(updated_centroids[i][key])
            else:
                centroids_vector[i].append(0)

    similarity_centroids = {}
    # شباهت بیت کویری و مراکز
    for i in range(len(updated_centroids)):
        similarity_centroids[i] = similarity(query_vector, centroids_vector[i])
    sortes_similarity = dict(sorted(similarity_centroids.items(), key=operator.itemgetter(1)))
    closest_centroids = list(sortes_similarity.keys())[:2]
    print()
    closest_docs = {}

    query_vector_new = []
    doc_vector_complete = []
    for i in range(2):
        for doc_idd in (clusters_docid[closest_centroids[i]]):
            #             print(doc_idd)
            for j in (doc_vector[doc_idd].keys()):
                if query_vector[j] != 0:
                    query_vector_new.append(query_vector[j])
                    doc_vector_complete.append(doc_vector[doc_idd][j])

            #             if doc_idd not in closest_docs.keys():
            closest_docs[doc_idd] = similarity(query_vector_new, doc_vector_complete)
    closest_docs_sorted = dict(sorted(closest_docs.items(), key=operator.itemgetter(1)))

    return closest_docs_sorted
if __name__ == "__main__":
    # word_before_stop = []
    # word_tokens = []
    normalizer = Normalizer()
    lemmatizer = Lemmatizer()
    stemmer = Stemmer()
    # dict_tokens, dataframe = pre_proccessing()
    # positional_index = {}
    # positional_index = create_positional_index(dict_tokens)
    # postional_index_tfidf = add_tfidf(positional_index)
    # dict_tokens_test, dataframe_test = pre_proccessing_test()
    # positional_index_test = {}
    # positional_index_test = create_positional_index(dict_tokens_test)
    # postional_index_tfidf_test = add_tfidf(positional_index_test)
    max_id, df_data = return_df()
    # max_id_test , df_test = return_df_test()
    # vector_data = vectorization(postional_index_tfidf ,max_id)
    # vector_test = vectorization(postional_index_tfidf_test ,max_id_test)
    # pickle_out = open("IR_PHASE3_pl.pickle","wb")
    # pickle.dump(postional_index_tfidf, pickle_out)
    # pickle_out2 = open("IR_PHASE3_test_PL.pickle","wb")
    # pickle.dump(postional_index_tfidf_test, pickle_out2)
    # pickle_out1 = open("vector_test.pickle","wb")
    # pickle.dump(vector_test, pickle_out1)
    # pickle_out3 = open("vector_data.pickle","wb")
    # pickle.dump(vector_data, pickle_out3)
    pickle_in1 = open("vector_data.pickle","rb")
    vector_data = pickle.load(pickle_in1)
    pickle_in = open("IR_PHASE3_pl.pickle","rb")
    postional_index_tfidf = pickle.load(pickle_in)
    # pickle_in12 = open("vector_test.pickle","rb")
    # vector_tset = pickle.load(pickle_in12)
    # pickle_in13 = open("IR_PHASE3_test_PL.pickle","rb")
    # postional_index_tfidf_test = pickle.load(pickle_in13)
    clusters_docid, updated_centroids = k_means(postional_index_tfidf, df_data, vector_data)

    while True:
        query_v = query_handling(postional_index_tfidf)
        listt_cluster = check_clusters(query_v, clusters_docid, updated_centroids, vector_data, postional_index_tfidf)
        for i in range(10):
            print(df_data.iloc[list(listt_cluster.keys())[i]]['url'])
