from enum import Enum

import tensorflow as tf

class Phase(Enum):
    Train = 0
    Validation = 1
    Predict = 2

class Model:
    def __init__(
            self,
            batch,
            label_batch,
            n_chars,
            hidden_sizes=[],
            phase=Phase.Predict,
            use_char_embeds=True):
        batch_size = batch.shape[1]
        input_size = batch.shape[2]
        label_size = label_batch.shape[2]

        self._x = tf.placeholder(tf.int32, shape=[batch_size, input_size])

        if phase != Phase.Predict:
            self._y = tf.placeholder(tf.float32, shape=[batch_size, label_size])

        # Fixme: Your code goes here!
        raise NotImplementedError

    @property
    def accuracy(self):
        return self._accuracy

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
