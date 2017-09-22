
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
            batch,
            lens_batch,
            label_batch,
            num_of_labels,
            embedding_matrix,
            phase=Phase.Predict):
        num_of_batches = len(lens_batch)
        batch_size = batch.shape[1]
        input_size = batch.shape[2]
        embedding_size = len(embedding_matrix[0])
        label_size = label_batch.shape[2]
        hidden_layers = 150  # 100
        # The integer-encoded words. Input_size is the (maximum) number of time steps,
        # here the longest sentence.
        self._x = tf.placeholder(tf.int32, shape=[batch_size, input_size])
        self._embeddings = tf.get_variable("embeddings", shape=[input_size, embedding_size])
        self._embeddings = embeddings = tf.nn.embedding_lookup(embedding_matrix, self._x)
        if phase == Phase.Train:
            embeddings = tf.contrib.layers.dropout(embeddings, keep_prob=config.input_dropout)

        # This tensor provides the actual number of time steps for each instance,
        # here the actual sentence lengths.
        self._lens = tf.placeholder(tf.int32, shape=[batch_size])

        # The label distribution.
        if phase != Phase.Predict:
            self._y = tf.placeholder(tf.int32, shape=[batch_size, label_size])

        forward_cell = tf.contrib.rnn.BasicLSTMCell(hidden_layers, reuse=tf.get_variable_scope().reuse)
        backward_cell = tf.contrib.rnn.BasicLSTMCell(hidden_layers, reuse=tf.get_variable_scope().reuse)
        if phase == Phase.Train:
            forward_cell = rnn.DropoutWrapper(forward_cell, state_keep_prob=config.hidden_dropout,
                                              output_keep_prob=config.hidden_dropout)
            backward_cell = rnn.DropoutWrapper(backward_cell, state_keep_prob=config.hidden_dropout,
                                               output_keep_prob=config.hidden_dropout)

        (hidden_1, hidden_2), _ = tf.nn.bidirectional_dynamic_rnn(forward_cell, backward_cell, embeddings,
                                                                  sequence_length=self._lens, dtype=tf.float32)
        hidden = tf.concat([hidden_1, hidden_2], axis=1)
        hidden = tf.nn.dropout(hidden, 0.9)

        w = tf.get_variable("w", shape=[2*hidden_layers, num_of_labels])
        b = tf.get_variable("b", shape=[1])

        hidden_flattened = tf.reshape(hidden, [-1, 2*hidden_layers])
        logits = tf.matmul(hidden_flattened, w) + b
        self._logits = logits = tf.reshape(logits, [batch_size, config.max_timesteps, num_of_labels])

        # CRF layer.
        if phase == Phase.Train or Phase.Validation:
            log_likelihood, self._transition_params = tf.contrib.crf.crf_log_likelihood(logits, self._y, self._lens)
            self._loss = tf.reduce_mean(-log_likelihood)
        if phase == Phase.Train:
            global_step = tf.Variable(0, trainable=False)
            start_lr = 0.01
            # Compute current learning rate
            learning_rate = tf.train.exponential_decay(start_lr, global_step, num_of_batches, 0.90)
            # TODO: compare different optimizers (originally: AdamOptimizer)
            self._train_op = tf.train.AdamOptimizer(learning_rate=learning_rate) \
                .minimize(self.loss, global_step=global_step)

    @property
    def embeddings(self):
        return self._embeddings

    @property
    def lens(self):
        return self._lens

    @property
    def logits(self):
        return self._logits

    @property
    def loss(self):
        return self._loss

    @property
    def train_op(self):
        return self._train_op

    @property
    def transition_params(self):
        return self._transition_params

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y