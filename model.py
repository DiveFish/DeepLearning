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

        if use_char_embeds:
            embedding_dims = 200
            self._embeddings = embeddings = tf.get_variable("embeddings", shape = [n_chars, embedding_dims])
            embeddings = tf.nn.embedding_lookup(embeddings, self._x)
            embeddings = tf.reshape(embeddings, [batch_size, (input_size * embedding_dims)])
        else:
            x_hot = tf.one_hot(self._x, n_chars)
            x_hot = tf.reshape(x_hot, [batch_size, (input_size * n_chars)])

        if phase != Phase.Predict:
            self._y = tf.placeholder(tf.float32, shape=[batch_size, label_size])

        if (use_char_embeds):
            hidden = self.hidden_layers(phase, embeddings, hidden_sizes)
        else:
            hidden = self.hidden_layers(phase, x_hot, hidden_sizes)  # output of last hidden layer

        w = tf.get_variable("w", shape = [hidden.shape[1], label_size])
        b = tf.get_variable("b", shape = [1])

        logits = tf.matmul(hidden, w) + b

        # loss in train and validation phase
        if phase != Phase.Predict:
            losses = tf.nn.softmax_cross_entropy_with_logits(labels = self._y, logits = logits)
            self._loss = loss = tf.reduce_sum(losses)

        if phase == Phase.Train:
            self._train_op = tf.train.AdagradOptimizer(0.005).minimize(loss)
        else:
            self._probs = probs = tf.nn.softmax(logits) # train on softmax -> use softmax here too; prob dist over tags
            #self._labels = tf.cast(tf.round(probs), tf.float32) # rounds probs <0.5 to 0 and >0.5 to 1

        if phase != Phase.Train:
            correct = tf.equal(tf.argmax(self.y, axis=1), tf.argmax(probs, axis=1))
            self._accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    def hidden_layers(self, phase, input_layer, hidden_sizes):
        for (i, hidden_size) in enumerate(hidden_sizes):
            if phase == Phase.Train:
                input_layer = tf.nn.dropout(input_layer, 0.90)
            W = tf.get_variable("W_hidden_%d" % i, shape = [input_layer.shape[1], hidden_size])
            b = tf.get_variable("b_hidden_%d" % i, shape = [hidden_size])
            hidden_outputs = tf.nn.relu(tf.matmul(input_layer, W) + b)
            input_layer = hidden_outputs
        return input_layer

    @property
    def accuracy(self):
        return self._accuracy

    @property
    def embeddings(self):
        return self._embeddings

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
