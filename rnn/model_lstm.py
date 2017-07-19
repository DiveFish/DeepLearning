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
            n_chars,
            use_LSTM = True,
            multi_LSTM = False,
            phase=Phase.Predict):
        batch_size = batch.shape[1]
        input_size = batch.shape[2]
        label_size = label_batch.shape[2]

        # The integer-encoded words. input_size is the (maximum) number of
        # time steps.
        self._x = tf.placeholder(tf.int32, shape=[batch_size, input_size])

        embedding_dims = 50
        self._embeddings = embeddings = tf.get_variable("embeddings", shape = [n_chars, embedding_dims])
        embeddings = tf.nn.embedding_lookup(embeddings, self._x)


        # This tensor provides the actual number of time steps for each
        # instance.
        self._lens = tf.placeholder(tf.int32, shape=[batch_size])


# The label distribution.
        if phase != Phase.Predict:
            self._y = tf.placeholder(
                tf.float32, shape=[batch_size, label_size])

        # Add your implementation of the RNN here. The code below
        # expects a final layer/node called 'logits', that one can
        # run a softmax over to get the label probability distribution.

        if (use_LSTM == True):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(100, reuse=tf.get_variable_scope().reuse)
            _, (c,h) = tf.nn.dynamic_rnn(lstm_cell, embeddings, sequence_length=self._lens, dtype=tf.float32)
            w = tf.get_variable("w", shape = [h.shape[1], label_size])
            b = tf.get_variable("b", shape = [1])

            logits = tf.matmul(h, w) + b

        elif (multi_LSTM == True):
            #cell = tf.contrib.rnn.LSTMCell(100, reuse=tf.get_variable_scope().reuse, state_is_tuple=True)

            #cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(3)], state_is_tuple=True)
            #init_state = cell.zero_state(batch_size, tf.float32)
            def lstm_cell():
                return tf.contrib.rnn.BasicLSTMCell(100)
            stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(3)])

            rnn_outputs, final_state = tf.nn.dynamic_rnn(stacked_lstm, embeddings, sequence_length=self._lens, dtype=tf.float32)

            with tf.variable_scope('softmax'):
                w = tf.get_variable("w", shape = [rnn_outputs.shape[1], label_size])
                b = tf.get_variable("b", shape = [1])

                #reshape rnn_outputs and y so we can get the logits in a single matmul
                rnn_outputs = tf.reshape(rnn_outputs, [-1, 100])
                self._y = tf.reshape(self._y, [-1])
                logits = [tf.matmul(rnn_output, w) + b for rnn_output in rnn_outputs]
        else:
            gru_cell = rnn.GRUCell(100)
            _, hidden = tf.nn.dynamic_rnn(gru_cell, embeddings, sequence_length=self._lens, dtype=tf.float32)
            w = tf.get_variable("w", shape = [hidden.shape[1], label_size])
            b = tf.get_variable("b", shape = [1])

            logits = tf.matmul(hidden, w) + b

        if phase == Phase.Train or Phase.Validation:
            losses = tf.nn.softmax_cross_entropy_with_logits(
                labels=self._y, logits=logits)
            self._loss = loss = tf.reduce_sum(losses)

        if phase == Phase.Train:
            start_lr = 0.01
            self._train_op = tf.train.AdamOptimizer(start_lr) \
                .minimize(losses)
            self._probs = probs = tf.nn.softmax(logits)

        if phase == Phase.Validation:
            # Highest probability labels of the gold data.
            hp_labels = tf.argmax(self.y, axis=1)

            # Predicted labels
            labels = tf.argmax(logits, axis=1)

            correct = tf.equal(hp_labels, labels)
            correct = tf.cast(correct, tf.float32)
            self._accuracy = tf.reduce_mean(correct)

    @property
    def accuracy(self):
        return self._accuracy

    @property
    def lens(self):
        return self._lens

    @property
    def loss(self):
        return self._loss

    @property
    def probs(self):
        return self._probs

    @property
    def train_op(self):
        return self._train_op

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def embeddings(self):
        return self._embeddings
