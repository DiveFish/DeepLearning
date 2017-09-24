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

    # Extract all named entities from a sentence and store them with their
    # start and end index to match them against the gold standard named entities.
    def extract_named_entities(sequence):
        chunks = []
        chunk = []
        for tag_index in range(len(sequence)):
            tag = sequence[tag_index]
            sequence_end = tag_index == len(sequence)-1
            # Beginning of chunk
            if tag.startswith('B') and (len(chunk) == 0):
                chunk.append(tag_index)
                chunk.append(tag)
            # (Model makes wrong prediction that chunk starts with 'I')
            elif tag.startswith('I') and (len(chunk) == 0):
                chunk.append(tag_index)
                chunk.append(tag)
            # Inside of chunk: for 'I' continue, for 'O' end chunk, for 'B' end chunk and begin new chunk
            elif tag.startswith('I'):
                chunk.append(tag)
            elif tag.startswith('O') and (len(chunk) > 0):
                chunk.append(tag_index)
                chunks.append(chunk)
                chunk = []
            elif tag.startswith('B') and (len(chunk) > 0):
                chunk.append(tag_index)
                chunks.append(chunk)
                chunk = []
                chunk.append(tag_index)
                chunk.append(tag)
            # A chunk at the end of the sequence
            if sequence_end and (len(chunk) > 0):
                chunk.append(tag_index)
                chunks.append(chunk)
        return chunks
