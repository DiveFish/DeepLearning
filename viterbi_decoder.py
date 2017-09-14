import tensorflow as tf

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