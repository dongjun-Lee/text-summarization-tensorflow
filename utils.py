from nltk.tokenize import word_tokenize
import re
import collections
import pickle
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec


train_article_path = "sumdata/train/train.article.txt"
train_title_path = "sumdata/train/train.title.txt"
valid_article_path = "sumdata/train/valid.article.filter.txt"
valid_title_path = "sumdata/train/valid.title.filter.txt"


def clean_str(sentence):
    sentence = re.sub("[#.]+", "#", sentence)
    return sentence


def get_text_list(data_path, toy):
    with open (data_path, "r", encoding='utf-8') as f:
        if not toy:
            return list(map(lambda x: clean_str(x.strip()), f.readlines()))
        else:
            return list(map(lambda x: clean_str(x.strip()), f.readlines()))[:50000]


def build_dict(step, toy=False):
    if step == "train":
        with open('word_dict.pickle', 'rb') as handle:
            print('Loading pre file')
            word_dict = pickle.load(handle)
            reversed_dict = dict(zip(word_dict.values(), word_dict.keys()))
            article_max_len = 50
            summary_max_len = 15
            return word_dict, reversed_dict, article_max_len, summary_max_len
        train_article_list = get_text_list(train_article_path, toy)
        train_title_list = get_text_list(train_title_path, toy)

        words = list()
        for sentence in train_article_list + train_title_list:
            for word in word_tokenize(sentence):
                words.append(word)

        word_counter = collections.Counter(words).most_common()
        word_dict = dict()
        word_dict["<padding>"] = 0
        word_dict["<unk>"] = 1
        word_dict["<s>"] = 2
        word_dict["</s>"] = 3
        for word, _ in word_counter:
            word_dict[word] = len(word_dict)

        with open("word_dict.pickle", "wb") as f:
            pickle.dump(word_dict, f)

    elif step == "valid":
        with open("word_dict.pickle", "rb") as f:
            word_dict = pickle.load(f)

    reversed_dict = dict(zip(word_dict.values(), word_dict.keys()))

    article_max_len = 50
    summary_max_len = 15

    return word_dict, reversed_dict, article_max_len, summary_max_len


def build_dataset(step, word_dict, article_max_len, summary_max_len, toy=False):
    if step == "train":
        article_list = get_text_list(train_article_path, toy)
        title_list = get_text_list(train_title_path, toy)
    elif step == "valid":
        article_list = get_text_list(valid_article_path, toy)
        title_list = get_text_list(valid_title_path, toy)
    else:
        raise NotImplementedError

    x = list(map(lambda d: word_tokenize(d), article_list))
    x = list(map(lambda d: list(map(lambda w: word_dict.get(w, word_dict["<unk>"]), d)), x))
    x = list(map(lambda d: d[:article_max_len], x))
    x = list(map(lambda d: d + (article_max_len - len(d)) * [word_dict["<padding>"]], x))

    y = list(map(lambda d: word_tokenize(d), title_list))
    y = list(map(lambda d: list(map(lambda w: word_dict.get(w, word_dict["<unk>"]), d)), y))
    y = list(map(lambda d: d[:(summary_max_len-1)], y))

    return x, y


def batch_iter(inputs, outputs, batch_size, num_epochs):
    inputs = np.array(inputs)
    outputs = np.array(outputs)

    num_batches_per_epoch = (len(inputs) - 1) // batch_size + 1
    for epoch in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(inputs))
            yield inputs[start_index:end_index], outputs[start_index:end_index]


def get_init_embedding(reversed_dict, embedding_size):
    glove_file = "glove/glove.42B.300d.txt"
    word2vec_file = get_tmpfile("word2vec_format.vec")
    glove2word2vec(glove_file, word2vec_file)
    print("Loading Glove vectors...")
    word_vectors = KeyedVectors.load_word2vec_format(word2vec_file)

    word_vec_list = list()
    for _, word in sorted(reversed_dict.items()):
        try:
            word_vec = word_vectors.word_vec(word)
        except KeyError:
            word_vec = np.zeros([embedding_size], dtype=np.float32)

        word_vec_list.append(word_vec)

    # Assign random vector to <s>, </s> token
    word_vec_list[2] = np.random.normal(0, 1, embedding_size)
    word_vec_list[3] = np.random.normal(0, 1, embedding_size)

    return np.array(word_vec_list)
