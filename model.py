#Authors: Neele Witte, 4067845; Patricia Fischer, 3928367
#Honor Code:  We pledge that this program represents our own work.

from enum import Enum

import tensorflow as tf
from tensorflow.contrib import rnn

from chunker import Chunker


class Phase(Enum):
    Train = 0
    Validation = 1
    Predict = 2


class Model:
    def __init__(
            self,
            config,
            batch,
            lens_batch,
            label_batch,
            n_chars,
            label_dict,  # dictionary of labels and their ID
            phase=Phase.Predict):
        num_of_batches = len(lens_batch)
        batch_size = batch.shape[1]
        input_size = batch.shape[2]
        label_size = label_batch.shape[2]
        hidden_layers = 200
        embedding_size = 50

        # The integer-encoded words. Input_size is the (maximum) number of time steps,
        # here the longest sentence.
        self._x = tf.placeholder(tf.int32, shape=[batch_size, input_size])
        # TODO: replace by Daniel's embeddings
        self._embeddings = embeddings = tf.get_variable("embeddings", shape=[n_chars, embedding_size])
        embeddings = tf.nn.embedding_lookup(embeddings, self._x)
        if phase == Phase.Train:
            embeddings = tf.nn.dropout(embeddings, keep_prob=config.input_dropout)

        # This tensor provides the actual number of time steps for each instance,
        # here the actual sentence lengths.
        self._lens = tf.placeholder(tf.int32, shape=[batch_size])

        # The label distribution.
        #TODO: self._y needs to be of same shape as input/self._x, i.e. a list of list of labels/ label vectors
        if phase != Phase.Predict:
            self._y = tf.placeholder(tf.float32, shape=[batch_size, len(label_dict)])

        forward_cell = tf.contrib.rnn.BasicLSTMCell(hidden_layers)
        backward_cell = tf.contrib.rnn.BasicLSTMCell(hidden_layers)
        if phase == Phase.Train:
            forward_cell = rnn.DropoutWrapper(forward_cell, state_keep_prob=config.hidden_dropout,
                                              output_keep_prob=config.hidden_dropout)
            backward_cell = rnn.DropoutWrapper(backward_cell, state_keep_prob=config.hidden_dropout,
                                               output_keep_prob=config.hidden_dropout)

        _, hidden = tf.nn.bidirectional_dynamic_rnn(forward_cell, backward_cell, embeddings,
                                                    sequence_length=self._lens, dtype=tf.float32)
        hidden_1, hidden_2 = hidden
        hidden = tf.concat([hidden_1, hidden_2], 1)

        w = tf.get_variable("w", shape=[hidden.shape[1], label_size])
        b = tf.get_variable("b", shape=[1])
        logits = tf.matmul(hidden, w) + b

        # CRF layer
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
                                                                    logits, labels=self._y, sequence_length=self._lens)
        if phase == Phase.Train or Phase.Validation:
            self.loss = tf.reduce_mean(-log_likelihood)

        if phase == Phase.Train:
            global_step = tf.Variable(0, trainable=False)
            start_lr = 0.01
            # Compute current learning rate
            learning_rate = tf.train.exponential_decay(start_lr, global_step, num_of_batches, 0.90)
            self._train_op = tf.train.AdamOptimizer(learning_rate=learning_rate)\
                                     .minimize(self.loss, global_step=global_step)
            self._probs = probs = tf.nn.softmax(logits)

        # TODO:  Where to put the following code?
        #<<<
        viterbi_sequences = []
        # iterate over the sentences
        for logit, sequence_length in zip(logits, self._lens):
            logit = logit[:sequence_length]
            viterbi_sequence, viterbi_score = tf.contrib.crf.viterbi_decode(logit, transition_params)
            viterbi_sequences += [viterbi_sequence]  # predicted labels

        if phase == Phase.Validation:
            foundGuessed = 0
            foundCorrect = 0
            correctChunks = 0
            chunker = Chunker()

            for i in range(len(viterbi_sequences)):
                sentence = viterbi_sequences[i]
                for j in range(len(sentence)):
                    guessed_tag = label_dict.get(sentence[j])
                    y_tag = label_dict.get(self._y[j])
                    if chunker.chunk_start(guessed_tag):
                        foundGuessed += 1
                    if chunker.chunk_start(y_tag):
                        foundCorrect += 1
                    if chunker.chunk_end(guessed_tag) and chunker.chunk_end(y_tag) and guessed_tag == y_tag:
                        correctChunks += 1

            self._precision = prec = 0
            if foundGuessed > 0:
                prec = 100 * correctChunks / foundGuessed

            self._recall = rec = 0
            if foundCorrect > 0:
                rec = 100 * correctChunks / foundCorrect

            self._f1_score = f1 = 0
            if prec > 0 and rec > 0:
                f1 = 2.0*prec*rec/(prec+rec)

    @property
    def embeddings(self):
        return self._embeddings

    @property
    def f1_score(self):
        return self._f1_score

    @property
    def lens(self):
        return self._lens

    @property
    def loss(self):
        return self._loss

    @property
    def precision(self):
        return self._precision

    @property
    def probs(self):
        return self._probs

    @property
    def recall(self):
        return self._recall

    @property
    def train_op(self):
        return self._train_op

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y