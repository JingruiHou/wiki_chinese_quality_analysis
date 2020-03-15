#! /usr/bin/env python
from __future__ import print_function
import numpy as np
import pandas as pd
import os
import time
import pickle
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import StratifiedKFold


def add_ngram(sequences, token_indice, ngram_range=2):
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for ngram_value in range(2, ngram_range + 1):
            for i in range(len(new_list) - ngram_value + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)
    return new_sequences


def create_ngram_dic(X, ngram_range):
    gram_dic = {}
    for ngram_value in range(2, ngram_range + 1):
        for input_list in X:
            for index in range(len(input_list) - ngram_value + 1):
                gram_key = tuple(input_list[index: index + ngram_value])
                if gram_key in gram_dic:
                    gram_dic[gram_key] += 1
                else:
                    gram_dic[gram_key] = 1
        print('gram_dic', len(gram_dic))
    return gram_dic


def build_fast_text_input(X, ngram_range, max_uni_features, maxlen, gram_min_count):
    max_features = max_uni_features
    if ngram_range > 1:
        print('Adding {}-gram features'.format(ngram_range))
        X = [x[: MAX_LEN // ngram_range] for x in X]
        start_index = max_features + 1
        gram_dic = create_ngram_dic(X, ngram_range)
        for gram_key in list(gram_dic.keys()):
            if gram_dic[gram_key] < gram_min_count[len(gram_key)]:
                gram_dic.pop(gram_key)
        ngram_set = gram_dic.keys()
        token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
        indice_token = {token_indice[k]: k for k in token_indice}
        max_features = np.max(list(indice_token.keys())) + 1
        X = add_ngram(X, token_indice, ngram_range)
    X = pad_sequences(X, maxlen=maxlen)
    return X, max_features


def get_input_token(inputs, max_word):
    txt_tokenizer = Tokenizer(nb_words=max_word)
    txt_tokenizer.fit_on_texts(inputs)
    txt_seq = txt_tokenizer.texts_to_sequences(inputs)
    return txt_seq


def build_model(args, output):
    if len(args) > 0:
        inputs = []
        poolings = []
        for arg in args:
            embedding_layer = layers.Embedding(arg['max_words'], arg['embedding_dim'], input_length=arg['maxlen'])
            sentence_input = layers.Input(shape=(arg['maxlen'],), dtype='int32')
            embedded_sequences = embedding_layer(sentence_input)
            pooling_1d = layers.GlobalAveragePooling1D()(embedded_sequences)
            inputs.append(sentence_input)
            poolings.append(pooling_1d)
    if len(args) > 1:
        pooling_layer = layers.Concatenate()(poolings)
    if len(args) == 1:
        pooling_layer = poolings[0]
        inputs = inputs[0]
    preds = layers.Dense(output, activation='softmax')(pooling_layer)
    model = models.Model(input=inputs, output=preds)
    return model


df_char = pd.read_csv(open('../cleanup/corpus_character.csv', 'r', encoding='UTF-8'))
df_word = pd.read_csv(open('../cleanup/corpus_words.csv', 'r', encoding='UTF-8'))


labels = df_char['label']
unique_label = list(labels.unique())
output_dim = len(unique_label)
Y = [unique_label.index(label) for label in labels]


MAX_CHAR = 6000
MAX_WORD = 25000
MAX_RADICAL = 300
MAX_PINYIN1 = 500
MAX_PINYIN2 = 1500

EMBEDDING_DIM = 100
MAX_LEN = 1000
GRAM_RANGE = 3
SEED = 9

os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'

X_char = [' '.join(list(x)) for x in df_char['text']]

X_char = get_input_token(X_char, MAX_CHAR)
X_word = get_input_token(df_word['content'], MAX_WORD)

X_char, MAX_CHAR = build_fast_text_input(X_char, GRAM_RANGE, MAX_CHAR, MAX_LEN, {2: 5, 3: 3})
X_word, MAX_WORD = build_fast_text_input(X_word, GRAM_RANGE, MAX_WORD, MAX_LEN, {2: 4, 3: 3})

print(MAX_WORD)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

index = 0
for train, test in kfold.split(X_char, Y):
    train_char = X_char[train]
    train_word = X_word[train]
    y_train = to_categorical(np.asarray(Y))[train]

    test_char = X_char[test]
    test_word = X_word[test]
    y_test = to_categorical(np.asarray(Y))[test]

    # 第一种模型：纯字符
    file_name = 'CHAR_MODEL_BI'
    args = [{'max_words': MAX_CHAR, 'embedding_dim': EMBEDDING_DIM, 'maxlen': MAX_LEN}]
    model = build_model(args, output_dim)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
    model.summary()
    start = time.clock()
    history = model.fit(train_char, y_train, validation_data=(test_char, y_test),
                        nb_epoch=20, batch_size=100)
    total_training_time = time.clock() - start
    index += 1
    with open("../saves/model_fitting_log", 'a') as f:
        f.write('Total training time of {}_{} is {}\n'.format(file_name, index, total_training_time))
    model.save(filepath="../saves/{}_{}_{}.h5".format(file_name, index, time.time()))
    history_dict = history.history
    with open("../saves/{}_{}_{}.dic".format(file_name, index, time.time()), "wb") as f:
        pickle.dump(history_dict, f)
    del model

    # 第二种模型：纯词
    file_name = 'WORD_MODEL_BI'
    args = [{'max_words': MAX_WORD, 'embedding_dim': EMBEDDING_DIM, 'maxlen': MAX_LEN}]
    model = build_model(args, output_dim)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
    model.summary()
    start = time.clock()
    history = model.fit(train_word, y_train, validation_data=(test_word, y_test),
                        nb_epoch=20, batch_size=100)
    total_training_time = time.clock() - start
    index += 1
    with open("../saves/model_fitting_log", 'a') as f:
        f.write('Total training time of {}_{} is {}\n'.format(file_name, index, total_training_time))
    model.save(filepath="../saves/{}_{}_{}.h5".format(file_name, index, time.time()))
    history_dict = history.history
    with open("../saves/{}_{}_{}.dic".format(file_name, index, time.time()), "wb") as f:
        pickle.dump(history_dict, f)
    del model

    # 第三种：混合输入，字、词、拼音1、拼音2、笔画
    file_name = 'COMBINED_MODEL_BI'
    args = [{'max_words': MAX_CHAR, 'embedding_dim': EMBEDDING_DIM, 'maxlen': MAX_LEN},
            {'max_words': MAX_WORD, 'embedding_dim': EMBEDDING_DIM, 'maxlen': MAX_LEN},
            {'max_words': MAX_PINYIN1, 'embedding_dim': EMBEDDING_DIM, 'maxlen': MAX_LEN},
            {'max_words': MAX_PINYIN2, 'embedding_dim': EMBEDDING_DIM, 'maxlen': MAX_LEN},
            {'max_words': MAX_RADICAL, 'embedding_dim': EMBEDDING_DIM, 'maxlen': MAX_LEN}]
    model = build_model(args, output_dim)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
    model.summary()
    start = time.clock()
    history = model.fit([train_char, train_word, train_pinyin1, train_pinyin2, train_radical], y_train,
                        validation_data=([test_char, test_word, test_pinyin1, test_pinyin2, test_radical], y_test),
                        nb_epoch=20, batch_size=100)
    total_training_time = time.clock() - start
    index += 1
    with open("../saves/model_fitting_log", 'a') as f:
        f.write('Total training time of {}_{} is {}\n'.format(file_name, index, total_training_time))
    model.save(filepath="../saves/{}_{}_{}.h5".format(file_name, index, time.time()))
    history_dict = history.history
    with open("../saves/{}_{}_{}.dic".format(file_name, index, time.time()), "wb") as f:
        pickle.dump(history_dict, f)
    del model