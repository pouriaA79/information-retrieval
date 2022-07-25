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


def mapp(s):
    if s == "sport":
        return 0
    elif s == "economy":
        return 1
    elif s == "political":
        return 2
    elif s == "cultre":
        return 3
    else:
        return 4


# به طور کلی میاد شباهت هر سند تست یعنی هفت هزار تارو با ترین حساب میکنه 5 تا شبیه ترینو کتگوری عدد شدشو  اونی که بیشتر هستو بر میگردونه یعنی میگه به کردوم شبیه تره
def KNN(v_test, v_data, max_id, max_id_test, df_data):
    category_test = []
    knn = 15
    category_list = {}

    for i in range(max_id_test):

        distance = {}
        doc_key_test = list(v_test[i].keys())
        for j in range(max_id):

            doc_key_data = list(v_data[j].keys())
            match_elements = [value for value in doc_key_data if value in doc_key_test]
            vector_ditance = []
            # اینجا مشترکا کم میشن گذاشته میشن تو لیست
            for k in match_elements:
                vector_ditance.append(v_test[i][k] - v_data[j][k])

            # اینجا کلماتی که توی داک هستند
            for k in doc_key_data:
                if k not in match_elements:
                    vector_ditance.append(v_data[j][k])

            # اینجا هم کلمات مرکز
            for k in doc_key_test:
                if k not in match_elements:
                    vector_ditance.append(v_test[i][k])
                    # فاصله رو حساب میکنیم
            vector_ditance = np.array(vector_ditance)
            distance[j] = np.sqrt(np.sum(np.square(vector_ditance)))
        sorted_d = dict(sorted(distance.items(), key=operator.itemgetter(1)))
        sorted_list = list(sorted_d.keys())

        listt_type = []
        for j in range(knn):
            #             print(df_data.iloc[sorted_list[j]]['topic'])
            #             print(mapp(df_data.iloc[sorted_list[j]]['topic']))
            listt_type.append(mapp(df_data.iloc[sorted_list[j]]['topic']))
        listt_type = np.array(listt_type)
        category_test.append(np.bincount(listt_type).argmax())

    return category_test


# وکتور کردن
def query_handling (pl):
    query = input("enter your query:  ")
    if "sport" in query:
        topic =  0
    elif  "economy" in query:
        topic =  1
    elif "political"in query :
        topic =  2
    elif "cultre" in query :
        topic =  3
    else :
        topic =  4
    query = ""
    query_list = list(query.split(" "))

    for i in range(1,len(query_list)):
        query.join(query_list)
        if i !=len(query_list)-1:
            query.join(" ")
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
    return np.array(query_vector), topic ,query_list
def similiraty(doc1, doc2):
    doc1 = np.array(doc1)
    doc2 = np.array(doc2)
    similarity = np.dot(doc1,doc2)/(norm(doc1)* norm(doc2))
    return similarity


# اگر در هامن کتگوری باشه حساب میکنه 10 تا شبیهو برمیگردونه
def find_doc(query_list, query_vector, topic, category, test_vector, df_test, tf_idf):
    print(query_list)
    scores = {}
    query_vector_new = []
    doc_vector_complete = []
    for i in range(len(category)):
        if category[i] == topic:
            query_vector_new = []
            doc_vector_complete = []
            for j in (range(len(tf_idf))):
                if query_vector[j] != 0:
                    query_vector_new.append(query_vector[j])
                    doc_vector_complete.append(test_vector[i][j])
            scores[i] = similiraty(query_vector_new, doc_vector_complete)
    sorted_d = dict(sorted(scores.items(), key=operator.itemgetter(1), reverse=True))
    sorted_list = list(sorted_d.keys())
    for i in range(10):
        print("TITLE")
        print(df_test.iloc[sorted_list[i]]['title'])
        print("- - - - - - - - - - - - - ")


if __name__ == "__main__":
    word_before_stop = []
    word_tokens = []
    normalizer = Normalizer()
    lemmatizer = Lemmatizer()
    stemmer = Stemmer()
    dict_tokens, dataframe = pre_proccessing()
    positional_index = {}
    positional_index = create_positional_index(dict_tokens)
    postional_index_tfidf = add_tfidf(positional_index)

    max_id, df_data = return_df()
    max_id_test, df_test = return_df_test()
    vector_data = vectorization(postional_index_tfidf ,max_id)
    pickle_in = open("IR_PHASE3_pl.pickle", "rb")
    postional_index_tfidf = pickle.load(pickle_in)

    pickle_in1 = open("vector_data.pickle", "rb")
    vector_dataaa = pickle.load(pickle_in1)

    pickle_in12 = open("vector_test.pickle", "rb")
    vector_tset = pickle.load(pickle_in12)
    pickle_in13 = open("IR_PHASE3_test_PL.pickle", "rb")
    postional_index_tfidf_test = pickle.load(pickle_in13)
    category = KNN(vector_tset, vector_dataaa, max_id, max_id_test, df_data)
    query_vector , topic ,query_list = query_handling(postional_index_tfidf_test)
    find_doc(query_list, query_vector, topic, category, vector_tset, df_test, postional_index_tfidf_test)
