from copy import copy
import argparse
from collections import defaultdict, Counter
from lxml import etree
from mosestokenizer import MosesTokenizer
import simplejson as json
import codecs
import random


from mydatasets import SemEval, SemEval_TD

rest_train = {14: '../SemEval2014Task4/Restaurants_Train_v2.xml'}
rest_test = {14: '../SemEval2014Task4/Restaurants_Test_Gold.xml'}
rest_dev = {14: '../SemEval2014Task4/Restaurants_Dev_v2.xml'}
laptop_train = {14: '../SemEval2014Task4/Laptop_Train_v2.xml'}
laptop_test = {14: '../SemEval2014Task4/Laptops_Test_Gold.xml'}
ds_train = {'r': rest_train, 'l': laptop_train}
ds_test = {'r': rest_test, 'l': laptop_test}


def read_sentence14(file_path, dedup_set=None):
    dataset = []
    with open(file_path, 'rb') as fopen:
        raw = fopen.read()
        root = etree.fromstring(raw)
        for sentence in root:
            example = dict()
            example["sentence"] = sentence.find('text').text.lower()
            categories = sentence.find("aspectCategories")
            example["aspect_sentiment"] = []
            for c in categories:
                aspect = c.attrib['category'].lower()
                if aspect == 'anecdotes/miscellaneous':
                    aspect = 'misc'
                example["aspect_sentiment"].append((aspect, c.attrib['polarity']))
            dataset.append(example)
    return dataset


def read_sentence14_target(file_path, max_offset_len=83):
    tk = MosesTokenizer()
    with open(file_path, 'rb') as fopen:
        raw = fopen.read()
        root = etree.fromstring(raw)
        for sentence in root:
            example = dict()
            example["sentence"] = sentence.find('text').text.lower()

            # for RAN
            tokens = tk.tokenize(example['sentence'])

            terms = sentence.find('aspectTerms')
            if terms is None:
                continue
            example["aspect_sentiment"] = []
            example["left_right"] = []
            example['offset'] = []

            for c in terms:
                target = c.attrib['term'].lower()
                example["aspect_sentiment"].append((target, c.attrib['polarity']))

                # for td lstm
                left_index = int(c.attrib['from'])
                right_index = int(c.attrib['to'])
                example["left_right"].append((example['sentence'][:right_index],
                                              example['sentence'][left_index:],
                                              c.attrib['polarity']))

                # for RAN
                left_word_offset = len(tk.tokenize(example['sentence'][:left_index]))
                right_word_offset = len(tk.tokenize(example['sentence'][right_index:]))
                token_index = list(range(len(tokens)))
                token_length = float(len(token_index))
                for i in range(len(tokens)):
                    if i < left_word_offset:
                        token_index[i] = 1 - (left_word_offset - token_index[i]) / token_length
                    elif i >= right_word_offset:
                        token_index[i] = 1 - (token_index[i] - (len(tokens) - right_word_offset) + 1) / token_length
                    else:
                        token_index[i] = 0
                token_index += [-1.] * (max_offset_len - len(tokens))
                example['offset'].append((token_index, target, c.attrib['polarity']))
            yield example


def get_semeval(years, aspects, rest_lap='r', use_attribute=False, dedup=False):
    if rest_lap == 'r':
        semeval14_train = read_sentence14(ds_train[rest_lap][14])
        semeval14_train = list(semeval14_train)
        print("# SemEval 14 Train: {0}".format(len(semeval14_train)))
    else:
        semeval14_train = []

    print("# Train: {}".format(len(semeval14_train)))

    if rest_lap == 'r':
        semeval14_test = read_sentence14(ds_test[rest_lap][14])
        semeval14_test = list(semeval14_test)
        print("# SemEval 14 Test: {0}".format(len(semeval14_test)))
    else:
        semeval14_test = []
    print("# Test: {}".format(len(semeval14_test)))
    if rest_lap == 'r':
        semeval14_dev = read_sentence14(rest_dev[14])
        semeval14_dev = list(semeval14_dev)
        print("# SemEval 14 Dev: {0}".format(len(semeval14_dev)))
    else:
        semeval14_dev = []

    print("# Dev: {}".format(len(semeval14_dev)))
    return semeval14_train, semeval14_test, semeval14_dev


def get_semeval_target(years, rest_lap='rest', dedup=False):
    semeval14_train = list(read_sentence14_target(ds_train[rest_lap][14]))
    print("# SemEval 14 Train: {0}".format(len(semeval14_train)))

    print("# Train: {0}".format(len(semeval14_train)))

    semeval14_test = list(read_sentence14_target(ds_test[rest_lap][14]))
    print("# SemEval 14 Test: {0}".format(len(semeval14_test)))

    print("# Test: {0}".format(len(semeval14_test)))
    return semeval14_train, semeval14_test

