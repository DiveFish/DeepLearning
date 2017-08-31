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
            self._y = tf.placeholder(tf.float32, shape=[batch_size, len(label_vectors)])

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
            # keep only the valid time steps
            logit = logit[:sequence_length]
            viterbi_sequence, viterbi_score = tf.contrib.crf.viterbi_decode(logit, transition_params)
            viterbi_sequences += [viterbi_sequence]
        #>>>

        # TODO: Does "outside of named entity" tag count as named-entity tag?
        if phase == Phase.Validation:
            # Highest probability labels of the gold data: self.y
            # Predicted labels: pred

            not_named_entity_val = label_vectors.get("O")
            # A tensor of same shape as y where each element is equal to the outside-named-entity label vector
            not_named_entity = tf.constant(not_named_entity_val, tf.float32, shape=[batch_size])

            named_entities_y = tf.not_equal(self._y, not_named_entity)
            named_entities_y = tf.cast(named_entities_y, tf.float32)
            not_named_entities_y = tf.equal(self._y, not_named_entity)
            not_named_entities_y = tf.cast(not_named_entities_y, tf.float32)

            # TODO: switch from logits to
            named_entities_logits = tf.not_equal(logits, not_named_entity)
            named_entities_logits = tf.cast(named_entities_logits, tf.float32)
            not_named_entities_logits = tf.equal(logits, not_named_entity)
            not_named_entities_logits = tf.cast(not_named_entities_logits, tf.float32)

            # matmul returns 1 for true=true and 0 for all other combinations of true/false
            true_positives = tf.matmul(named_entities_y, named_entities_logits)
            true_positives = tf.equal(true_positives, True)
            true_positives = tf.reduce_sum(true_positives)

            false_positives = tf.matmul(named_entities_logits, not_named_entities_y)
            false_positives = tf.equal(false_positives, True)
            false_positives = tf.reduce_sum(false_positives)

            false_negatives = tf.matmul(not_named_entities_logits, named_entities_y)
            false_negatives = tf.equal(false_negatives, True)
            false_negatives = tf.reduce_sum(false_negatives)

            self._precision = prec = true_positives / (true_positives + false_positives)
            self._recall = rec = true_positives / (true_positives + false_negatives)
            self._f1_score = f1 = 2.0*prec*rec/(prec+rec)

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