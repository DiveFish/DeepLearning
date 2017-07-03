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

        'placeholder for input data'
        self._x = tf.placeholder(tf.int32, shape=[batch_size, input_size])

        'if char_embeds is true a variable for char embeddings is created. Embedding dimensions are set to 200' \
        'The embedding matrix contains one embedding per character of prefix and suffic for every instance of input'
        if use_char_embeds:
            embedding_dims = 200
            self._embeddings = embeddings = tf.get_variable("embeddings", shape = [n_chars, embedding_dims])
            embeddings = tf.nn.embedding_lookup(embeddings, self._x)
            embeddings = tf.reshape(embeddings, [batch_size, (input_size * embedding_dims)])

            'If char_embeds is false a onehot representation of every character of prefix and suffix of the input is created' \
            'The onehot vectors for all characters are concatenated in order to retrieve the feature vector '
        else:
            x_hot = tf.one_hot(self._x, n_chars)
            x_hot = tf.reshape(x_hot, [batch_size, (input_size * n_chars)])

        if phase != Phase.Predict:
            'a placeholder for the label of the training data is created'
            self._y = tf.placeholder(tf.float32, shape=[batch_size, label_size])

        'data is send through the hidden layers.'
        if (use_char_embeds):
            hidden = self.hidden_layers(phase, embeddings, hidden_sizes)
        else:
            hidden = self.hidden_layers(phase, x_hot, hidden_sizes)  # output of last hidden layer

        w = tf.get_variable("w", shape = [hidden.shape[1], label_size])
        b = tf.get_variable("b", shape = [1])

        logits = tf.matmul(hidden, w) + b

        'loss is calculated in train and validation phase. loss function is softmax_cross_entropy'
        if phase != Phase.Predict:
            losses = tf.nn.softmax_cross_entropy_with_logits(labels = self._y, logits = logits)
            self._loss = loss = tf.reduce_sum(losses)

        if phase == Phase.Train:
            'best optimizer: AdagradOptimizer with a learning rate of 0.005'
            self._train_op = tf.train.AdagradOptimizer(0.005).minimize(loss)
        else:
            'probability distribution over tags is calculated with softmax function'
            self._probs = probs = tf.nn.softmax(logits)
        if phase != Phase.Train:
            'accuracy is calculated in validation and prediction phase'
            correct = tf.equal(tf.argmax(self.y, axis=1), tf.argmax(probs, axis=1))
            self._accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    'takes input, applies nonlinear function and multiplication with weighvector, adding bias' \
    'returns the result after applieng all hidden layers'
    def hidden_layers(self, phase, input_layer, hidden_sizes):
        for (i, hidden_size) in enumerate(hidden_sizes):
            if phase == Phase.Train:
                'using dropout for regularization, 0.7 works best'
                input_layer = tf.nn.dropout(input_layer, 0.70)
            W = tf.get_variable("W_hidden_%d" % i, shape = [input_layer.shape[1], hidden_size])
            b = tf.get_variable("b_hidden_%d" % i, shape = [hidden_size])
            'best activation function compared to others: ReLU'
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