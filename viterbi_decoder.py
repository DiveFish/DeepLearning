# Authors: Neele Witte, 4067845; Patricia Fischer, 3928367
# Honor Code:  We pledge that this program represents our own work.

import tensorflow as tf


# Decode the best tag sequence from all possible tag sequences
class Viterbi_Decoder:

    def decode(self, logits, transition_params, sentence_lengths):
        # Predicted labels.
        viterbi_sequences = []
        # Iterate over sentences
        for logit, sentence_length in zip(logits, sentence_lengths):
            logit = logit[:sentence_length]
            viterbi_sequence, viterbi_score = tf.contrib.crf.viterbi_decode(logit, transition_params)
            viterbi_sequences += [viterbi_sequence]

        return viterbi_sequences