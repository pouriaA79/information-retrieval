from __future__ import unicode_literals
from itertools import combinations
import pandas as pd
from hazm import *
from hazm import stopwords_list
import math
import operator
import numpy as np


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


def handling_champion_lists(pl):
    import operator
    for key in (pl.keys()):
        weights = {}
        pl[key]['champion_list'] = []
        for idd in (pl[key]['DOCID'].keys()):
            weights[idd] = pl[key]['DOCID'][idd][1]
        sorted_d = dict(sorted(weights.items(), key=operator.itemgetter(1), reverse=True))

        if len(sorted_d.keys()) < 10:
            pl[key]['champion_list'] = list(sorted_d.keys())
        else:
            pl[key]['champion_list'] = list(sorted_d.keys())[0:10]
    return pl


def query_champion_list(pl):
    query = input("enter your query:  ")
    query = normalizer.normalize(query)
    word_before_stop = word_tokenize(query)
    stop_words_list = stopwords_list()
    used = []
    word_tokens = [x for x in word_before_stop if x not in stop_words_list]
    query_pl = {}
    print(word_tokens)
    for i in range(len(word_tokens)):
        word_tokens[i] = stemmer.stem(word_tokens[i])
        word_tokens[i] = lemmatizer.lemmatize(word_tokens[i])
    for i in range(len(word_tokens)):
        if word_tokens[i] not in query_pl.keys():
            query_pl[word_tokens[i]] = [1, i]
        else:
            query_pl[word_tokens[i]][0] += 1
            query_pl[word_tokens[i]].append(i)
    keys = pl.keys()
    N = len(pl)
    id_relevant_list = []
    used = set()
    unique = [x for x in word_tokens if x not in used and (used.add(x) or True)]
    for key in (unique):
        if len(pl[key]['champion_list']) > 10:
            for i in range(10):
                if pl[key]['champion_list'][i] not in id_relevant_list:
                    id_relevant_list.append(pl[key]['champion_list'][i])

        else:
            for i in range(len(pl[key]['champion_list'])):
                if pl[key]['champion_list'][i] not in id_relevant_list:
                    id_relevant_list.append(pl[key]['champion_list'][i])

        nt = len(pl[key]['DOCID'])
        tfidf = (1 + math.log(query_pl[key][0], 10)) * math.log(N / nt, 10)
        query_pl[key].insert(1, tfidf)

    final_dict_score = {}
    vector = np.ones(len(query_pl))
    #     print(vector)
    c = 0
    for key in query_pl.keys():
        vector[c] = query_pl[key][1]
        c += 1
    normalized_v = vector / np.linalg.norm(vector)
    c = 0
    for key in query_pl.keys():
        query_pl[key][1] = normalized_v[c]
        c += 1
    #     print(query_pl)
    score_query = 0
    for key in (query_pl.keys()):
        score_query += query_pl[key][1] * query_pl[key][1]
    for idd in id_relevant_list:
        score = 0
        score_2 = 0
        ddoc_pl = {}
        for key in (query_pl.keys()):
            if idd in pl[key]['champion_list']:
                ddoc_pl[key] = pl[key]['DOCID'][idd][1]
                score += pl[key]['DOCID'][idd][1] * query_pl[key][1]
                score_2 += pl[key]['DOCID'][idd][1] * pl[key]['DOCID'][idd][1]
        final_dict_score[idd] = score / (math.sqrt(score_2) * math.sqrt(score_query))

    final_dict_score_sort = dict(sorted(final_dict_score.items(), key=operator.itemgetter(1), reverse=True))
    DOCID_list = list(final_dict_score_sort.keys())
    return DOCID_list


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
    postional_index_tfidf = normalization(postional_index_tfidf, dataframe.shape[0])
    postional_index_tfidf = handling_champion_lists(postional_index_tfidf)

    while True:

        DOCID_list = query_champion_list(postional_index_tfidf)
        if len(DOCID_list) <= 10:
            for i in range(len(DOCID_list)):
                print("TITLE")
                print(dataframe.iloc[DOCID_list[i]]['title'])
                #                     print("CONTENT")
                #                     print(dataframe.iloc[DOCID_list[i]]['content'])
                print("- - - - - - - - - - - - - ")
        else:
            for i in range(10):
                #                     print(DOCID_list[i])
                print("TITLE")
                print(dataframe.iloc[DOCID_list[i]]['title'])
                #                     print("CONTENT")
                #                     print(dataframe.iloc[DOCID_list[i]]['content'])
                print("- - - - - - - - - - - - - ")



