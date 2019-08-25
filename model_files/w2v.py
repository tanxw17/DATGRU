import gensim
import numpy as np
import linecache
from sklearn.externals import joblib


def load_glove_embedding(word_list, uniform_scale, dimension_size):
    glove_words = []
    with open('../embedding/glove_words.txt', 'r') as fopen:
        for line in fopen:
            glove_words.append(line.strip())
    word2offset = {w: i for i, w in enumerate(glove_words)}
    word_vectors = []
    for word in word_list:

        if word in word2offset:
            line = linecache.getline('../embedding/glove.840B.300d.txt', word2offset[word]+1)
            assert(word == line[:line.find(' ')].strip())
            word_vectors.append(np.fromstring(line[line.find(' '):].strip(), sep=' ', dtype=np.float32))
        elif word == '<pad>':
            word_vectors.append(np.zeros(dimension_size, dtype=np.float32))
        else:
            word_vectors.append(np.random.uniform(-uniform_scale, uniform_scale, dimension_size))

    return word_vectors


def save_embedding(word_list, word_embedding,
                   word_list_file='embedding/yelp_words.txt',
                   word_embedding_file='embedding/yelp_embedding.txt'):
    with open(word_list_file, 'w') as fopen:
        for w in word_list:
            fopen.write(w + '\n')

    with open(word_embedding_file, 'w') as fopen:
        for i in range(len(word_list)):
            w = word_list[i]
            fopen.write(w)
            for n in word_embedding[i]:
                fopen.write(' {:.5f}'.format(n))
            fopen.write('\n')


def load_aspect_embedding_from_w2v(aspect_list, word_stoi, w2v):
    aspect_vectors = []
    for w in aspect_list:
        temp = w2v[word_stoi[w.split()[0]]]
        aspect_vectors.append(np.random.rand(300)*0.02-0.01)
    return aspect_vectors
