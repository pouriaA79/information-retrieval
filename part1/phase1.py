from __future__ import unicode_literals
from itertools import combinations
import pandas as pd
from hazm import *
from hazm import stopwords_list

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
    positional_index = {}
    dict_index = 0
    k = 1
    for i in range(len(dict_tokens)):
        for j in range(len(dict_tokens[i])):
            if dict_tokens[i][j] not in positional_index.keys():
                dict_index += 1
                positional_index[dict_tokens[i][j]] = {'total_count': 1, 'DOCID': {i: [k, j]}}
            else:
                positional_index[dict_tokens[i][j]]['total_count'] += 1
                if i in positional_index[dict_tokens[i][j]]['DOCID'].keys():
                    positional_index[dict_tokens[i][j]]['DOCID'][i].append(j)
                    positional_index[dict_tokens[i][j]]['DOCID'][i][0] += 1
                else:
                    positional_index[dict_tokens[i][j]]['DOCID'][i] = [k, j]

    return positional_index

def sort_by_values_len(dictionary):
    dict_len= {key: len(value) for key, value in dictionary.items()}
    import operator
    sorted_key_list = sorted(dict_len.items(), key=operator.itemgetter(1), reverse=True)
    sorted_dict = [{item[0]: dictionary[item [0]]} for item in sorted_key_list]
    return sorted_dict


def more_search(list_word_index, PL, word_list, sdnc):
    final_result = []
    for t in range(len(sdnc)):
        answer_dict = sdnc[t]
        doc_id = int((list((answer_dict).keys()))[0])
        len_s = len((list((answer_dict).values()))[0])
        if len_s == 4:
            for j in range(len(list_word_index)):

                if len(list_word_index[j]) == len_s:
                    if doc_id in PL[word_list[list_word_index[j][3]]]['DOCID'] and doc_id in \
                            PL[word_list[list_word_index[j][2]]]['DOCID'] and doc_id in \
                            PL[word_list[list_word_index[j][1]]]['DOCID'] and doc_id in \
                            PL[word_list[list_word_index[j][0]]]['DOCID']:
                        fourth_token_list = PL[word_list[list_word_index[j][3]]]['DOCID'][doc_id][1:]
                        third_token_list = PL[word_list[list_word_index[j][2]]]['DOCID'][doc_id][1:]
                        second_token_list = PL[word_list[list_word_index[j][1]]]['DOCID'][doc_id][1:]
                        for i in range(PL[word_list[list_word_index[j][0]]]['DOCID'][doc_id][0] - 1):
                            if PL[word_list[list_word_index[j][0]]]['DOCID'][doc_id][i + 1] + 1 in second_token_list and \
                                    PL[word_list[list_word_index[j][0]]]['DOCID'][doc_id][
                                        i + 1] + 2 in third_token_list and \
                                    PL[word_list[list_word_index[j][0]]]['DOCID'][doc_id][
                                        i + 1] + 3 in fourth_token_list:
                                final_result.append(doc_id)
                                break
                elif len(list_word_index[j]) < len_s:
                    break
        elif len_s == 3:
            for j in range(len(list_word_index)):

                if len(list_word_index[j]) == len_s:
                    if doc_id in PL[word_list[list_word_index[j][2]]]['DOCID'] and doc_id in \
                            PL[word_list[list_word_index[j][1]]]['DOCID'] and doc_id in \
                            PL[word_list[list_word_index[j][0]]]['DOCID']:
                        third_token_list = PL[word_list[list_word_index[j][2]]]['DOCID'][doc_id][1:]
                        second_token_list = PL[word_list[list_word_index[j][1]]]['DOCID'][doc_id][1:]
                        for i in range(PL[word_list[list_word_index[j][0]]]['DOCID'][doc_id][0] - 1):
                            if PL[word_list[list_word_index[j][0]]]['DOCID'][doc_id][i + 1] + 1 in second_token_list and \
                                    PL[word_list[list_word_index[j][0]]]['DOCID'][doc_id][
                                        i + 1] + 2 in third_token_list:
                                final_result.append(doc_id)
                                break
                elif len(list_word_index[j]) < len_s:
                    break
        elif len_s == 2:
            for j in range(len(list_word_index)):

                if len(list_word_index[j]) == len_s:
                    if doc_id in PL[word_list[list_word_index[j][1]]]['DOCID'] and doc_id in \
                            PL[word_list[list_word_index[j][0]]]['DOCID']:
                        second_token_list = PL[word_list[list_word_index[j][1]]]['DOCID'][doc_id][1:]
                        for i in range(PL[word_list[list_word_index[j][0]]]['DOCID'][doc_id][0] - 1):
                            if PL[word_list[list_word_index[j][0]]]['DOCID'][doc_id][i + 1] + 1 in second_token_list:
                                final_result.append(doc_id)
                                break


                elif len(list_word_index[j]) < len_s:
                    break
        elif len_s == 1:
            final_result.append(doc_id)

    return final_result


def more_than_one_word_2(wordlist, PL):
    length = len(wordlist)
    word_dict = {}
    for i in range(len(wordlist)):
        if wordlist[i] in PL.keys():
            word_dict[wordlist[i]] = list(positional_index[wordlist[i]]['DOCID'].keys())
    total_list = []
    index_list = []
    for i in range(len(wordlist)):
        for j in range(len(word_dict[wordlist[i]])):
            total_list.append(word_dict[wordlist[i]][j])
        index_list.append(len(total_list))
    dict_numbers_of_count = {}
    for j in range(len(total_list)):
        result = []
        elementindex = -1
        while True:
            try:
                elementindex = total_list.index(total_list[j], elementindex + 1)
                result.append(elementindex)
            except  ValueError:
                break
        dict_numbers_of_count[str(total_list[j])] = result

    sort_dict_numbers_of_count = sort_by_values_len(dict_numbers_of_count)
    list_combination = []
    list_word_index = []
    arr = []

    #     print(sort_dict_numbers_of_count)

    for i in range(length):
        arr.append(i)
    for i in reversed(range(length)):
        list_combination.append(list(combinations(arr, i + 1)))
    for i in range(len(list_combination)):
        length_list = len(list_combination[i])
        for j in range(length_list):
            list_word_index.append(list(list_combination[i][j]))

    f_result = more_search(list_word_index, PL, wordlist, sort_dict_numbers_of_count)
    return f_result


def query_handling(positional_index):
    query = input("enter your query:  ")
    query = normalizer.normalize(query)
    word_before_stop = word_tokenize(query)
    stop_words_list = stopwords_list()
    word_tokens = [x for x in word_before_stop if x not in stop_words_list]
    for i in range(len(word_tokens)):
        word_tokens[i] = stemmer.stem(word_tokens[i])
        word_tokens[i] = lemmatizer.lemmatize(word_tokens[i])
    DOCID_list1 = more_than_one_word_2(word_tokens , positional_index)
    if len(DOCID_list1) <= 10:
        for i in range(len(DOCID_list1)):
            print(dataframe.iloc[DOCID_list1[i]]['title'])
            print("- - - - - - - - - - - - - ")
    else:
        for i in range(10):
            print(dataframe.iloc[DOCID_list1[i]]['title'])
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
    while True:
        query_handling(positional_index)
