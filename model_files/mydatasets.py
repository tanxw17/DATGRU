import re
import os
import random
import tarfile
from six.moves import urllib
from torchtext import data
from torch.utils.data import Dataset


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()

class SemEval(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_field, as_field, sm_field, input_data, **kwargs):
        """Create an SemEval dataset instance given a path and fields.

        Arguments:
            text_field: The field that will be used for text data.
            as_field: The field that will be used for aspect data.
            sm_field: The field that will be used for sentiment data.
            input_data: The examples contain all the data.
            Remaining keyword arguments: Passed to the constructor of data.Dataset.
        """

        text_field.preprocessing = data.Pipeline(clean_str)
        fields = [('text', text_field), ('aspect', as_field), ('sentiment', sm_field)]

        examples = []
        for e in input_data:
            if 'pp.' in e['sentence']:
                continue
            examples.append(data.Example.fromlist([e['sentence'], e['aspect'], e['sentiment']], fields))
        super(SemEval, self).__init__(examples, fields, **kwargs)

    @staticmethod
    def unroll(input_data):
        import json
        # return unrolled sentences and sentences which have multiple aspects and different sentiments
        unrolled = []
        mixed = []
        from collections import defaultdict
        total_counter = defaultdict(int)
        mixed_counter = defaultdict(int)
        sen_set = set()
        for e in input_data:
            for aspect, sentiment in e['aspect_sentiment']:
                unrolled.append({'sentence': e['sentence'], 'aspect': aspect, 'sentiment': sentiment})
                if len(e['aspect_sentiment']) and len(set(map(lambda x: x[1], e['aspect_sentiment']))) > 1:
                    sen_set.add(e['sentence'])
                    mixed.append(
                        {'sentence': e['sentence'], 'aspect': aspect, 'sentiment': sentiment})
                    mixed_counter[sentiment] += 1
                total_counter[sentiment] += 1
        print("total")
        print(total_counter)
        print("hard")
        print(mixed_counter)
        print('hard sentence count: '+str(len(sen_set)))
        return unrolled, mixed

    @classmethod
    def splits_train_test(cls, text_field, as_field, sm_field, semeval_train, semeval_test, semeval_dev, **kwargs):
        unrolled_train, mixed_train = SemEval.unroll(semeval_train)
        print("# Unrolled Train: {}    # Mixed Train: {}".format(len(unrolled_train), len(mixed_train)))

        unrolled_test, mixed_test = SemEval.unroll(semeval_test)
        print("# Unrolled Test: {}    # Mixed Test: {}".format(len(unrolled_test), len(mixed_test)))

        unrolled_dev, mixed_dev = SemEval.unroll(semeval_dev)
        print("# Unrolled Dev: {}    # Mixed Dev: {}".format(len(unrolled_dev), len(mixed_dev)))
        
        return (cls(text_field, as_field, sm_field, unrolled_train),
                cls(text_field, as_field, sm_field, unrolled_test),
                cls(text_field, as_field, sm_field, unrolled_dev))


class SemEval_TD(data.Dataset):
    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_field, left_text_field, right_text_field, sm_field, input_data, **kwargs):

        text_field.preprocessing = data.Pipeline(clean_str)

        left_text_field.preprocessing = data.Pipeline(clean_str)
        left_text_field.init_token = '<beg>'

        right_text_field.preprocessing = data.Pipeline(clean_str)
        right_text_field.init_token = '<end>'

        fields = [('text', text_field), ('left_text', left_text_field), ('right_text', right_text_field),
                  ('sentiment', sm_field)]

        # unroll
        examples = []
        for e in input_data:
            examples.append(data.Example.fromlist([e['sentence'], e['left'], e['right'], e['sentiment']], fields))
        super(SemEval_TD, self).__init__(examples, fields, **kwargs)

    @staticmethod
    def unroll(input_data):
        unrolled = []
        mixed = []
        for e in input_data:
            all_sentiments = set()
            for l, r, s in e['left_right']:
                unrolled.append({'sentence': e['sentence'], 'left': l, 'right': r, 'sentiment': s})
                all_sentiments.add(s)
            if len(all_sentiments) > 1:
                for l, r, s in e['left_right']:
                    mixed.append({'sentence': e['sentence'], 'left': l, 'right': r, 'sentiment': s})
        return unrolled, mixed

    @classmethod
    def splits(cls, text_field, left_text_field, right_text_field, sm_field, semeval_train, semeval_test, **kwargs):
        unrolled_train, mixed_train = SemEval_TD.unroll(semeval_train)
        print("# Unrolled Train: {}    # Mixed Train: {}".format(len(unrolled_train), len(mixed_train)))

        unrolled_test, mixed_test = SemEval_TD.unroll(semeval_test)
        print("# Unrolled Test: {}    # Mixed Test: {}".format(len(unrolled_test), len(mixed_test)))
        return (cls(text_field, left_text_field, right_text_field, sm_field, unrolled_train),
                cls(text_field, left_text_field, right_text_field, sm_field, unrolled_test),
                cls(text_field, left_text_field, right_text_field, sm_field, mixed_test))


