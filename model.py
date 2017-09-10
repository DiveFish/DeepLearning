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
            label_id_dict, # dictionary of IDs with their label
            phase=Phase.Predict):
        num_of_batches = len(lens_batch)
        batch_size = batch.shape[1]
        input_size = batch.shape[2]
        embedding_size = batch.shape[3]
        label_size = label_batch.shape[2]
        hidden_layers = 100

        '''
        NER code:
        a batch is a list of tuples, a tuple = (sentence, tags);
        minibatches() method returns x_batch, y_batch = [], []
        x_batch = [[word1, word2], [...], ...]
        y_batch = [[label1, label2], [...], ...]

        '''

        # The integer-encoded words. Input_size is the (maximum) number of time steps,
        # here the longest sentence.
        self._x = tf.placeholder(tf.float32, shape=[batch_size, input_size, embedding_size])
        if phase == Phase.Train:
            self._x = tf.nn.dropout(self._x, keep_prob=config.input_dropout)

        # This tensor provides the actual number of time steps for each instance,
        # here the actual sentence lengths.
        self._lens = tf.placeholder(tf.int32, shape=[batch_size])

        # The label distribution.
        if phase != Phase.Predict:
            self._y = tf.placeholder(tf.int32, shape=[batch_size, label_size])

        forward_cell = tf.contrib.rnn.BasicLSTMCell(hidden_layers)
        backward_cell = tf.contrib.rnn.BasicLSTMCell(hidden_layers)
        if phase == Phase.Train:
            forward_cell = rnn.DropoutWrapper(forward_cell, state_keep_prob=config.hidden_dropout,
                                              output_keep_prob=config.hidden_dropout)
            backward_cell = rnn.DropoutWrapper(backward_cell, state_keep_prob=config.hidden_dropout,
                                               output_keep_prob=config.hidden_dropout)

        (hidden_1, hidden_2), _ = tf.nn.bidirectional_dynamic_rnn(forward_cell, backward_cell, self._x,
                                                    sequence_length=self._lens, dtype=tf.float32)
        hidden = tf.concat([hidden_1, hidden_2], -1)
        hidden = tf.reshape(hidden, [-1, 2*hidden_layers])
        # TODO: test
        hidden = tf.nn.dropout(hidden, 0.9)

        w = tf.get_variable("w", shape=[2*hidden_layers, label_size])
        b = tf.get_variable("b", shape=[1])
        logits = tf.matmul(hidden, w) + b
        logits = tf.reshape(logits, [-1, tf.shape(hidden)[1], label_size])

        # CRF layer.
        if phase == Phase.Train or Phase.Validation:
            log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(logits, self._y, self._lens)
            self._loss = tf.reduce_mean(-log_likelihood)

        if phase == Phase.Train:
            global_step = tf.Variable(0, trainable=False)
            start_lr = 0.01
            # Compute current learning rate
            learning_rate = tf.train.exponential_decay(start_lr, global_step, num_of_batches, 0.90)
            # TODO: compare different optimizers
            self._train_op = tf.train.AdamOptimizer(learning_rate=learning_rate)\
                                     .minimize(self.loss, global_step=global_step)

        # Predicted labels.
        viterbi_sequences = []
        # Iterate over sentences
        for logit, sentence_length in zip(logits, self._lens):
            logit = logit[:sentence_length]
            viterbi_sequence, viterbi_score = tf.contrib.crf.viterbi_decode(logit, transition_params)
            viterbi_sequences += [viterbi_sequence]

        # Get performance of model: precision, recall, f1-score
        if phase == Phase.Validation:
            found_guessed = 0
            found_correct = 0
            correct_chunks = 0
            chunker = Chunker()

            for seq in range(len(viterbi_sequences)):
                tags = viterbi_sequences[seq]
                inside_chunk = False
                for tag_idx in range(len(tags)):
                    guessed_tag = label_id_dict.get(tags[tag_idx])
                    y_tag = label_id_dict.get(self._y[tag_idx])
                    if chunker.chunk_start(guessed_tag):
                        found_guessed += 1
                        inside_chunk = True
                        if guessed_tag == y_tag:
                            in_correct = True
                    if chunker.chunk_start(y_tag):
                        found_correct += 1
                    # For each tag check whether it matches the gold standard tag
                    # in_correct true if all tags in chunk processed so far matched
                    if inside_chunk and (guessed_tag == y_tag) and in_correct:
                        continue
                    else:
                        in_correct = False
                    if chunker.chunk_end(guessed_tag) and chunker.chunk_end(y_tag) and in_correct:
                        correct_chunks += 1
                        # Reset values for next chunk
                        inside_chunk = False
                        in_correct = False

            if found_guessed > 0:
                self._precision = prec = 100 * correct_chunks / found_guessed

            if found_correct > 0:
                self._recall = rec = 100 * correct_chunks / found_correct

            if (prec > 0) and (rec > 0):
                self._f1_score = 2.0 * prec*rec / (prec + rec)

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