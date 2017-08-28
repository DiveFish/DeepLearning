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

def read_data(filename):
    words = []
    tags = []
    with open(filename, "r") as f:
        for line in f:
            if (line != "\n"):
                parts = line.split("\t")
                word = parts[1]
                tag = parts[5]
                words.append(word)
                tags.append(tag)
    return words, tags

def get_one_hot_labels(labels):
    print(labels)
    unique_labels = list(set(labels))
    zeros = np.zeros(len(unique_labels))
    print(zeros)
    one_hot_dic = dict()

    for i in range(len(unique_labels)):
        labelvec = np.zeros(len(unique_labels))
        labelvec[i]=1
        one_hot_dic[unique_labels[i]]= labelvec

    return one_hot_dic

def read_files(mypath):
    files = listdir(mypath)
    filenames = []
    for f in files:
        filenames.append(mypath+"/"+f)
    allwords = []
    alltags = []
    for file in filenames:
        (words, tags) = read_data(file)
        allwords.append(words)
        alltags.append(tags)
    return (allwords, alltags)

def read_wordEmbeddings(f):

    word_vectors = KeyedVectors.load_word2vec_format(f, binary=False)
    return word_vectors


def get_train_test_parts(allwords, alltags):
    training_words = [item for sublist in allwords[0:7] for item in sublist]
    test_words = [item for sublist in allwords[8:9] for item in sublist]
    training_tags = [item for sublist in alltags[0:7] for item in sublist]
    test_tags = [item for sublist in alltags[8:9] for item in sublist]
    return[training_words, test_words, training_tags, test_tags]

def convert_data(words, tags, embeds, tagdic):
    data = []
    for i in range(len(words)):
        if(words[i].lower() in embeds.wv.vocab):
            wordvec = embeds.wv[(words[i].lower())]
        else:
            wordvec = np.zeros(50)
        data.append(wordvec)
    return data

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
        max_label,
        max_timesteps,
        batch_size=128):
    n_batches = len(data) // batch_size

    # We are discarding the last batch for now, for simplicity.
    labels = np.zeros(
        shape=(
            n_batches,
            batch_size,
            max_label),
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
            max_timesteps),
        dtype=np.int32)

    for batch in range(n_batches):
        for idx in range(batch_size):
            (word, l) = data[(batch * batch_size) + idx]

            # Add label distribution
            for label, prob in l.items():
                labels[batch, idx, label] = prob

            # Sequence
            timesteps = min(max_timesteps, len(word))

            # Sequence length (time steps)
            lengths[batch, idx] = timesteps

            # Word characters
            words[batch, idx, :timesteps] = word[:timesteps]

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
    (allwords, alltags) = read_files("/home/neele/Dokumente/DeepLearningPY3/dl4nlp17-ner")
    labeldic = get_one_hot_labels(alltags[0])
    wordEmbeddings = read_wordEmbeddings("/home/neele/Dokumente/InformationRetrieval/glove.6B/glove_model50d.txt")
    print("embeddings have been read")
    [trainingwords, testwords, trainingtags, testtags] = get_train_test_parts(allwords, alltags)
    x = convert_data(testwords, testtags, wordEmbeddings, labeldic)
    print(x)

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