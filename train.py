#Authors: Neele Witte, 4067845; Patricia Fischer, 3928367
#Honor Code:  We pledge that this program represents our own work.

from enum import Enum
import os
import sys
from os import listdir
import math
import numpy as np
import tensorflow as tf
from gensim.models.keyedvectors import KeyedVectors

from config import DefaultConfig
from model import Model, Phase
from numberer import Numberer
from viterbi_decoder import Viterbi_Decoder
from scorer import Scorer


# Global variable data, needed in order to only iterate once
data = []


def word2index(model):
    """Extract word2index mapping from Word2Vec model.
    """
    counter = 0
    word2index = {}
    for key in word_embeddings.wv.index2word:
        word2index[key] = counter
        counter += 1
    return word2index

# Iterate over one file, save word embedding and corresponding label as one-hot representation
def read_data(filename, embeddings, label_dict, word2index):
    with open(filename, "r") as f:
        sentence = []
        for line in f:
            if (line != "\n"):
                parts = line.split("\t")
                word = parts[1]
                tag = parts[5]
                wordindex, tagvec = convert_data(word, tag, embeddings, label_dict, word2index)
                sentence.append((wordindex, tagvec))
            else:
                data.append(sentence)
                sentence = []


# Number representations for all labels
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
        label_to_number[label] = n
        number_to_label[n] = label
    return (label_to_number, number_to_label)


def convert_word_embeddings(embeds):
    embeds.wv.init_sims(True)
    complete_embeddings = np.zeros(
        shape=(
            len(embeds.syn0norm) + 2,
            len(embeds.syn0norm[0])),
        dtype=np.float32)
    for line_idx in range(len(embeds.syn0norm)):
        complete_embeddings[line_idx] = embeds.syn0norm[line_idx]
    unknown = np.random.sample(100)
    none_word = np.random.sample(100)   # needed for
    complete_embeddings[len(embeds.syn0norm)] = unknown
    complete_embeddings[len(embeds.syn0norm) + 1] = none_word
    return complete_embeddings


# Get all files from corpus
def read_files(mypath):
    files = listdir(mypath)
    filenames = []
    for f in files:
        filenames.append(mypath+"/"+f)
    return filenames


# Read word embeddings from file
def read_word_embeddings(f):
    word_vectors = KeyedVectors.load_word2vec_format(f, binary=True)
    return word_vectors


# Convert word and label into word embedding and label vector
def convert_data(word, tag, embeds, tag_dict, index2word):
    if (word in embeds.wv.vocab):
        wordindex = index2word[word]
    else:
        #hier sollte eine id für unknown word angegeben werden
        wordindex = 0

    return (wordindex, tag_dict[tag])


def generate_instances(data, max_timesteps, batch_size=DefaultConfig.batch_size):
    n_batches = len(data) // batch_size

    # We are discarding the last batch for now, for simplicity.
    labels = np.zeros(
        shape=(
            n_batches,
            batch_size,
            max_timesteps),
        dtype=np.int32)
    lengths = np.zeros(
        shape=(
            n_batches,
            batch_size),
        dtype=np.int32)
    sentences = np.zeros(
        shape=(
            n_batches,
            batch_size,
            max_timesteps),
        dtype=np.int32)

    for batch in range(n_batches):
        for idx in range(batch_size):
            sent = data[(batch*batch_size)+idx]
            s = [i[0] for i in sent]
            l = [i[1] for i in sent]

            # Sequence
            timesteps = min(max_timesteps, len(sent))

            # Sequence length (time steps)
            lengths[batch, idx] = timesteps

            # Labels
            labels[batch, idx, :timesteps] = l[:timesteps]

            # Words
            sentences[batch, idx,:timesteps] = s[:timesteps]

    return (sentences, lengths, labels)


def train_model(config, train_batches, train_lens, train_labels,validation_batches, validation_lens, validation_labels,
                word_embeddings, num_of_labels, number_to_label):

    with tf.Session() as sess:
        with tf.variable_scope("model", reuse=False):
            train_model = Model(
                config,
                train_batches,
                train_lens,
                train_labels,
                num_of_labels,
                word_embeddings,
                phase=Phase.Train)

        with tf.variable_scope("model", reuse=True):
            validation_model = Model(
                config,
                validation_batches,
                validation_lens,
                validation_labels,
                num_of_labels,
                word_embeddings,
                phase=Phase.Validation)

        sess.run(tf.global_variables_initializer())

        for epoch in range(config.n_epochs):
            train_loss = 0.0
            validation_loss = 0.0
            precision = 0.0
            recall = 0.0
            f1_score = 0.0

            # Train on all batches.
            for batch in range(train_batches.shape[0]):
                loss, _ = sess.run([train_model.loss,
                                    train_model.train_op], {
                                       train_model.x: train_batches[batch],
                                       train_model.lens: train_lens[batch],
                                       train_model.y: train_labels[batch]})
                train_loss += loss

                print("Trained on batch "+str(batch))

            print("Training done")
            decoder = Viterbi_Decoder()
            scorer = Scorer()
            # Validate on all batches.
            for batch in range(validation_batches.shape[0]):
                loss, logits, transition_params = sess.run([validation_model.loss,
                                                            validation_model.logits,
                                                            validation_model.transition_params], {
                                                               validation_model.x: validation_batches[batch],
                                                               validation_model.lens: validation_lens[batch],
                                                               validation_model.y: validation_labels[batch]})
                validation_loss += loss
                print()
                viterbi_sequences = decoder.decode(logits, transition_params, validation_lens[batch])
                print("Scores for batch "+str(batch))
                prec, rec, f1 = scorer.scores(viterbi_sequences, number_to_label, validation_labels, batch)
                # Get prec, rec and f1 for current batch
                precision += prec
                recall += rec
                f1_score += f1

            train_loss /= train_batches.shape[0]
            validation_loss /= validation_batches.shape[0]
            precision /= validation_batches.shape[0]
            recall /= validation_batches.shape[0]
            f1_score /= validation_batches.shape[0]

            print(
                "epoch %d - train loss: %.2f, validation loss: %.2f,"
                "validation precision: %.2f, validation recall: %.2f, validation f1-score: %.2f" %
                (epoch, train_loss, validation_loss, precision, recall, f1_score))


# TODO: use different embedding file
if __name__ == "__main__":

    filenames = read_files(sys.argv[1])
    tags = get_labels(filenames)
    (label_to_number, number_to_label) = convert_label_to_number(tags)
    embedding_file = sys.argv[2]
    word_embeddings = read_word_embeddings(embedding_file)
    print("Embeddings have been read")
    word_to_index = word2index(word_embeddings)
    for f in filenames:
        read_data(f, word_embeddings, label_to_number, word_to_index)
    complete_embeddings = convert_word_embeddings(word_embeddings)

    split = math.ceil((len(data)/5)*4)  # TODO: change back to /5) *4
    training = data[0:split]
    test = data[split+1:]
    print("Data has been read")

    num_of_labels = len(label_to_number)
    print("Number of labels: " + str(num_of_labels))

    (train_sentences, train_lengths, train_labels) = generate_instances(
        training,
        DefaultConfig.max_timesteps,
        batch_size=DefaultConfig.batch_size)

    (validation_sentences, validation_lengths, validation_labels) = generate_instances(
        test,
        DefaultConfig.max_timesteps,
        batch_size=DefaultConfig.batch_size)

    # Train the model
    train_model(DefaultConfig, train_sentences, train_lengths, train_labels,
                validation_sentences, validation_lengths, validation_labels,
                complete_embeddings, num_of_labels, number_to_label)
