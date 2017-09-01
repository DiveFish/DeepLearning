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
        for line in f:
            if (line != "\n"):
                parts = line.split("\t")
                word = parts[1]
                tag = parts[5]
                wordvec, tagvec = convert_data(word, tag, embeddings, labeldic)
                data.append((wordvec, tagvec))

'''get onehot representations for all labels'''
def get_label_dic(files):
    tags = []
    for file in files:
        with open(file, "r") as f:
            for line in f:
                if (line != "\n"):
                    parts = line.split("\t")
                    tag = parts[5]
                    tags.append(tag)
    dic = get_one_hot_labels(tags)
    return dic

'''get one hot representation for one label'''
def get_one_hot_labels(labels):
    unique_labels = list(set(labels))
    one_hot_dic = dict()
    for i in range(len(unique_labels)):
        labelvec = np.zeros(len(unique_labels))
        labelvec[i]=1
        one_hot_dic[unique_labels[i]]= labelvec
    return one_hot_dic

'''get all files from corpus'''
def read_files(mypath):
    files = listdir(mypath)
    filenames = []
    for f in files:
        filenames.append(mypath+"/"+f)
    return filenames

'''read word embeddings from file'''
def read_wordEmbeddings(f):

    word_vectors = KeyedVectors.load_word2vec_format(f, binary=False)
    return word_vectors

'''convert word and label into wordembedding and label vector '''
def convert_data(word, tag, embeds, tagdic):
    if(word.lower() in embeds.wv.vocab):
        wordvec = embeds.wv[(word.lower())]
    else:
        wordvec = np.zeros(50)

    return (wordvec, tagdic[tag])

def read_lexicon(filename):
    with open(filename, "r") as f:
        lex = {}

        for line in f:
            parts = line.split()
            tag_parts = parts[1:]

            tags = {}

            for i in range(0, len(tag_parts), 2):
                tag = tag_parts[i]
                prob = float(tag_parts[i + 1])
                tags[tag] = prob

            lex[parts[0]] = tags

        return lex


def recode_lexicon(lexicon, chars, labels, train=False):
    int_lex = []

    for (word, tags) in lexicon.items():
        int_word = []
        for char in word:
            int_word.append(chars.number(char, train))

        int_tags = {}
        for (tag, p) in tags.items():
            int_tags[labels.number(tag, train)] = p

        int_lex.append((int_word, int_tags))

    return int_lex


def generate_instances(
        data,
        labelsize,
        max_timesteps,
        batch_size=DefaultConfig.batch_size):
    n_batches = len(data) // batch_size

    # We are discarding the last batch for now, for simplicity.
    labels = np.zeros(
        shape=(
            n_batches,
            batch_size,
            labelsize),
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
            50),
        dtype=np.int32)

    for batch in range(n_batches):
        for idx in range(batch_size):
            (word, l) = data[(batch * batch_size) + idx]
            # Add label distribution
            labels[batch, idx] = l

            # Sequence
            timesteps = min(max_timesteps, len(word))

            # Sequence length (time steps)
            lengths[batch, idx] = timesteps

            # Word characters
            words[batch, idx] = word
            print(words[batch, idx])

    return (words, lengths, labels)


def train_model(config, train_batches, validation_batches):
    train_batches, train_lens, train_labels = train_batches
    validation_batches, validation_lens, validation_labels = validation_batches

    n_chars = max(np.amax(validation_batches), np.amax(train_batches)) + 1

    with tf.Session() as sess:
        with tf.variable_scope("model", reuse=False):
            train_model = Model(
                config,
                train_batches,
                train_lens,
                train_labels,
                n_chars,

                phase=Phase.Train)

        with tf.variable_scope("model", reuse=True):
            validation_model = Model(
                config,
                validation_batches,
                validation_lens,
                validation_labels,
                n_chars,
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
    labeldic = get_label_dic([filenames[8], filenames[6], filenames[7]])
    wordEmbeddings = read_wordEmbeddings("/home/neele/Dokumente/InformationRetrieval/glove.6B/glove_model50d.txt")
    print("embeddings have been read")
    for f in filenames:
        print(f)
        read_data(f, wordEmbeddings, labeldic)
    training = data[0:372418]
    test = data[372419:]
    print("data has been read")

    train_batches = generate_instances(
        training,
        len(labeldic.keys()),
        DefaultConfig.max_timesteps,
        batch_size=DefaultConfig.batch_size)
    train_batches, train_lens, train_labels = train_batches
    #print(train_batches)

    """
    if len(sys.argv) != 3:
        sys.stderr.write("Usage: %s TRAIN_SET DEV_SET\n" % sys.argv[0])
        sys.exit(1)

    config = DefaultConfig()

    # Read training and validation data.
    train_lexicon = read_lexicon(sys.argv[1])
    validation_lexicon = read_lexicon(sys.argv[2])

    # Convert word characters and part-of-speech labels to numeral
    # representation.
    chars = Numberer()
    labels = Numberer()
    train_lexicon = recode_lexicon(train_lexicon, chars, labels, train=True)
    validation_lexicon = recode_lexicon(validation_lexicon, chars, labels)

    # Generate batches
    train_batches = generate_instances(
        train_lexicon,
        labels.max_number(),
        config.max_timesteps,
        batch_size=config.batch_size)
    validation_batches = generate_instances(
        validation_lexicon,
        labels.max_number(),
        config.max_timesteps,
        batch_size=config.batch_size)

    # Train the model
    train_model(config, train_batches, validation_batches)
    """