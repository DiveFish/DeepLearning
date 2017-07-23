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
            phase=Phase.Predict,
            use_lstm = False,
            use_stacked = True,
            use_bidir = False):
        num_of_batches = len(lens_batch)
        batch_size = batch.shape[1]
        input_size = batch.shape[2]
        label_size = label_batch.shape[2]
        hidden_layers = 10
        embedding_size = 50

        print("CONFIG: bidirectional")
        print("input dropout: "+str(config.input_dropout))
        print("hidden dropout: "+str(config.hidden_dropout))
        print("hidden layers: "+str(hidden_layers))
        print("embedding size: "+str(embedding_size))
        print("learning rate: 0.01")
        print("optimizer: AdamOptimizer")

        # The integer-encoded words. input_size is the (maximum) number of
        # time steps.
        self._x = tf.placeholder(tf.int32, shape=[batch_size, input_size])
        self._embeddings = embeddings = tf.get_variable("embeddings", shape=[n_chars, embedding_size])
        embeddings = tf.nn.embedding_lookup(embeddings, self._x)

        # This tensor provides the actual number of time steps for each
        # instance.
        self._lens = tf.placeholder(tf.int32, shape=[batch_size])

        # The label distribution.
        if phase != Phase.Predict:
            self._y = tf.placeholder(
                tf.float32, shape=[batch_size, label_size])

        if use_bidir:
            forward_cell = rnn.GRUCell(hidden_layers)
            reg_forward_cell = rnn.DropoutWrapper(forward_cell, input_keep_prob=config.input_dropout,
                                                  output_keep_prob=config.hidden_dropout)
            backward_cell = rnn.GRUCell(hidden_layers)
            reg_backward_cell = rnn.DropoutWrapper(backward_cell, input_keep_prob=config.input_dropout,
                                                   output_keep_prob=config.hidden_dropout)
            _, hidden = tf.nn.bidirectional_dynamic_rnn(reg_forward_cell, reg_backward_cell, embeddings,
                                                        sequence_length=self._lens, dtype=tf.float32)
            hidden_1, hidden_2 = hidden
            hidden = tf.concat([hidden_1, hidden_2], 1)

        elif use_lstm:
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_layers, reuse=tf.get_variable_scope().reuse)
            # TODO: check if dropout works!
            reg_lstm_cell = rnn.DropoutWrapper(lstm_cell, input_keep_prob=config.input_dropout,
                                               output_keep_prob=config.hidden_dropout)
            _, (_, hidden) = tf.nn.dynamic_rnn(reg_lstm_cell, embeddings, sequence_length=self._lens, dtype=tf.float32)

        elif use_stacked:
            num_of_lstms = 5
            # TODO: check if dropout works!

            def stack_input_cell():
                basic_stack_cell = tf.contrib.rnn.BasicLSTMCell(hidden_layers, reuse=tf.get_variable_scope().reuse)
                return rnn.DropoutWrapper(basic_stack_cell, input_keep_prob=config.input_dropout,
                                          output_keep_prob=config.hidden_dropout)
            stacked_cell = tf.contrib.rnn.MultiRNNCell([stack_input_cell() for _ in range(num_of_lstms)], state_is_tuple=True)
            _, hiddens = tf.nn.dynamic_rnn(stacked_cell, embeddings, sequence_length=self._lens, dtype=tf.float32)
            # hiddens = hidden1, hidden2, hidden3,....

            _, hidden_orig = hiddens[0]
            for _, hidden in hiddens[1:]:
                hidden_orig = tf.concat([hidden_orig, hidden], 1)

        else:
            gru_cell = rnn.GRUCell(hidden_layers)
            # TODO: check state_keep_prob dropout!
            regularized_cell = rnn.DropoutWrapper(gru_cell, input_keep_prob=config.input_dropout,
                                                  output_keep_prob=config.hidden_dropout)
            _, hidden = tf.nn.dynamic_rnn(regularized_cell, embeddings, sequence_length=self._lens, dtype=tf.float32)

        w = tf.get_variable("w", shape=[hidden.shape[1], label_size])
        b = tf.get_variable("b", shape=[1])
        logits = tf.matmul(hidden, w) + b

        if phase == Phase.Train or Phase.Validation:
            losses = tf.nn.softmax_cross_entropy_with_logits(
                labels=self._y, logits=logits)
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
    def embeddings(self):
        return self._embeddings

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
