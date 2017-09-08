from enum import Enum
import os
import sys
from os import listdir
import numpy as np
import tensorflow as tf
from gensim.models.keyedvectors import KeyedVectors

from config import DefaultConfig
from model import Model, Phase
from numberer import Numberer

'''global variable data, needed in order to only iterate once'''
data = []

'''iterate over one file and save wordembedding an corresponding label as onehot representation'''
def read_data(filename, embeddings, labeldic):

    with open(filename, "r") as f:
        sentence = []
        for line in f:
            if (line != "\n"):
                parts = line.split("\t")
                word = parts[1]
                tag = parts[5]
                wordvec, tagvec = convert_data(word, tag, embeddings, labeldic)
                sentence.append((wordvec, tagvec))
            else:
                data.append(sentence)
                sentence = []



'''get number representations for all labels'''
def get_labels(files):
    tags = []
    for file in files:
        with open(file, "r") as f:
            for line in f:
                if (line != "\n"):
                    parts = line.split("\t")
                    tag = parts[5]
                    tags.append(tag)
    return tags



def convert_label_to_number(labels):
    unique_labels = list(set(labels))
    label_to_number = dict()
    number_to_label = dict()
    l = Numberer()
    for label in unique_labels:
        n = l.number(label)
        label_to_number[label]= n
        number_to_label[n] = label
    return (label_to_number, number_to_label)



'''get all files from corpus'''
def read_files(mypath):
    files = listdir(mypath)
    filenames = []
    for f in files:
        filenames.append(mypath+"/"+f)
    return filenames

'''read word embeddings from file'''
def read_wordEmbeddings(f):

    word_vectors = KeyedVectors.load_word2vec_format(f, binary=True)
    return word_vectors

'''convert word and label into wordembedding and label vector '''
def convert_data(word, tag, embeds, tagdic):
    if(word.lower() in embeds.wv.vocab):
        wordvec = embeds.wv[(word.lower())]
    else:
        wordvec = np.zeros(100)

    return (wordvec, tagdic[tag])



def generate_instances(data, n_labels, max_timesteps,  batch_size=DefaultConfig.batch_size):
    n_batches = len(data) // batch_size
    print("start")
    print(n_labels, n_batches, batch_size)
    # We are discarding the last batch for now, for simplicity.
    labels = np.zeros(
        shape=(
            n_batches,
            batch_size,
            max_timesteps),
        dtype=np.float32)
    lengths = np.zeros(
        shape=(
            n_batches,
            batch_size),
        dtype=np.int32)
    words = np.zeros(
        shape=(
            n_batches,
            batch_size,
            max_timesteps,
            100),
        dtype=np.int32)

    for batch in range(n_batches):
        for idx in range(batch_size):
            sen = data[(batch*batch_size)+idx]
            w = [i[0] for i in sen]
            l = [i[1] for i in sen]

            # Sequence
            timesteps = min(max_timesteps, len(sen))
            #labels
            labels[batch, idx, :timesteps] = l[:timesteps]

            # Sequence length (time steps)
            lengths[batch, idx] = timesteps

            #words

            words[batch, idx,:timesteps] = w[:timesteps]

    return (words, lengths, labels)


def train_model(config, train_batches, train_lens, train_labels,validation_batches, validation_lens, validation_labels):



    with tf.Session() as sess:
        with tf.variable_scope("model", reuse=False):
            train_model = Model(
                config,
                train_batches,
                train_lens,
                train_labels,
                label_to_number,
                number_to_label,

                phase=Phase.Train)

        with tf.variable_scope("model", reuse=True):
            validation_model = Model(
                config,
                validation_batches,
                validation_lens,
                validation_labels,
                label_to_number,
                number_to_label,
                phase=Phase.Validation)

        sess.run(tf.global_variables_initializer())

        for epoch in range(config.n_epochs):
            train_loss = 0.0
            validation_loss = 0.0
            accuracy = 0.0

            # Train on all batches.
            for batch in range(train_batches.shape[0]):
                loss, _ = sess.run([train_model.loss, train_model.train_op], {
                    train_model.x: train_batches[batch], train_model.lens: train_lens[batch], train_model.y: train_labels[batch]})
                train_loss += loss

            # validation on all batches.
            for batch in range(validation_batches.shape[0]):
                loss, acc = sess.run([validation_model.loss, validation_model.accuracy], {
                    validation_model.x: validation_batches[batch], validation_model.lens: validation_lens[batch], validation_model.y: validation_labels[batch]})
                validation_loss += loss
                accuracy += acc

            train_loss /= train_batches.shape[0]
            validation_loss /= validation_batches.shape[0]
            accuracy /= validation_batches.shape[0]

            print(
                "epoch %d - train loss: %.2f, validation loss: %.2f, validation acc: %.2f" %
                (epoch, train_loss, validation_loss, accuracy * 100))


if __name__ == "__main__":


    #(words, tags) = read_data("/home/neele/Dokumente/DeepLearningPY3/dl4nlp17-ner/part1.conll")
    #print(len(words))
    #print(len(tags))

    filenames = read_files("/home/neele/Dokumente/DeepLearningPY3/dl4nlp17-ner")
    tags = get_labels([filenames[8], filenames[6], filenames[7]])
    (label_to_number, number_to_label) = convert_label_to_number(tags)
    wordEmbeddings = read_wordEmbeddings("/home/neele/Downloads/wikipedia-100-mincount-20-window-5-cbow.bin")

    print("embeddings have been read")
    for f in filenames:
        read_data(f, wordEmbeddings, label_to_number)
    training = data[0:372418]
    test = data[372419:]
    print("data has been read")


    (train_sentences, train_lengths, train_labels) = generate_instances(
        training,
        len(label_to_number.keys())+1,
        DefaultConfig.max_timesteps,
        batch_size=DefaultConfig.batch_size)
    print("done")

    (valid_sentences, valid_lengths, valid_labels) = generate_instances(
        test,
        len(label_to_number.keys())+1,
        DefaultConfig.max_timesteps,
        batch_size=DefaultConfig.batch_size)

    print("done")


    # Train the model
    train_model(DefaultConfig, train_sentences, train_lengths, train_labels, valid_sentences, valid_lengths, valid_labels)