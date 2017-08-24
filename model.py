#Authors: Neele Witte, 4067845; Patricia Fischer, 3928367
#Honor Code:  We pledge that this program represents our own work.

from enum import Enum

import tensorflow as tf
from tensorflow.contrib import rnn


class Phase(Enum):
    Train = 0
    Validation = 1
    Predict = 2


class Model:
    def __init__(
            self,
            config,
            label_vectors, # dictionary of labels and their one-hot vector representation
            batch,
            lens_batch,
            label_batch,
            n_chars,
            phase=Phase.Predict):
        num_of_batches = len(lens_batch)
        batch_size = batch.shape[1]
        input_size = batch.shape[2]
        label_size = label_batch.shape[2]
        hidden_layers = 200
        embedding_size = 50
        num_of_lstms = 2    # for stacked lstm

        # The integer-encoded words. input_size is the (maximum) number of
        # time steps.
        self._x = tf.placeholder(tf.int32, shape=[batch_size, input_size])
        self._embeddings = embeddings = tf.get_variable("embeddings", shape=[n_chars, embedding_size])
        embeddings = tf.nn.embedding_lookup(embeddings, self._x)
        if phase == Phase.Train:
            embeddings = tf.nn.dropout(embeddings, keep_prob=config.input_dropout)

        # This tensor provides the actual number of time steps for each
        # instance.
        self._lens = tf.placeholder(tf.int32, shape=[batch_size])

        # The label distribution.
        if phase != Phase.Predict:
            self._y = tf.placeholder(tf.float32, shape=[batch_size])

        forward_cell = tf.contrib.rnn.BasicLSTMCell(hidden_layers)  # instead use rnn.BasicLSTMCell
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

        if phase == Phase.Train or Phase.Validation:
            losses = tf.nn.softmax_cross_entropy_with_logits(labels=self._y, logits=logits)
            self._loss = loss = tf.reduce_sum(losses)

        if phase == Phase.Train:
            global_step = tf.Variable(0, trainable=False)
            start_lr = 0.01
            # Compute current learning rate
            learning_rate = tf.train.exponential_decay(start_lr, global_step, num_of_batches, 0.90)
            self._train_op = tf.train.AdamOptimizer(learning_rate=learning_rate) \
                .minimize(losses, global_step=global_step)
            self._probs = probs = tf.nn.softmax(logits)

        if phase == Phase.Validation:
            # Highest probability labels of the gold data: self.y

            # Predicted labels: logits

            # TODO: include label dict
            # accuracy; for precision, consider only the named-entity labels, not O (outisde of named entity)
            correct = tf.equal(self._y, logits)
            correct = tf.cast(correct, tf.float32)
            self._precision = tf.reduce_mean(correct)

            # TODO: include label dict
            # recall: correctly labeled named entities vs. correctly and erroneously labeled named entities
            correct = tf.equal(self._y, logits)
            correct = tf.cast(correct, tf.float32)
            incorrect = tf.not_equal(self._y, logits)
            incorrect = tf.cast(incorrect, tf.float32)
            self._recall = correct / (correct+incorrect)

            # f1 score: 2∗precision∗recall/(precision+recall)
            self._f1_score = 2*self._precision*self._recall/(self._precision+self._recall)

    @property
    def embeddings(self):
        return self._embeddings

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