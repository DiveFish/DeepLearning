import numpy as np

class Scorer:

    def __init__(self):
        self.precision = 0.0
        self.recall = 0.0
        self.f1_score = 0.0

    def scores(self, viterbi_sequences, id_to_tag_dict, validation_labels, batch):
        # Get performance of model: precision, recall, f1-score
        found_viterbi = 0
        correct_viterbi = 0
        found_y = 0  # i.e. correct_y
        for seq in range(len(viterbi_sequences)):
            viterbi_tag_idxs = viterbi_sequences[seq]
            viterbi_tags = [id_to_tag_dict[i] for i in np.asarray(viterbi_tag_idxs)]
            viterbi_entities = extract_named_entities(viterbi_tags)

            y_tag_idxs = validation_labels[batch][seq][:len(viterbi_tag_idxs)]
            y_tags = [id_to_tag_dict[j] for j in y_tag_idxs]
            y_entities = extract_named_entities(y_tags)

            found_viterbi += len(viterbi_entities)
            found_y += len(y_entities)
            for ner in viterbi_entities:
                if ner in y_entities:
                    correct_viterbi += 1

        if found_viterbi > 0:
            print("Correct chunks "+str(correct_viterbi))
            print("Found guessed "+str(found_viterbi))
            self.precision = 100.00 * (correct_viterbi / found_viterbi)
            print("Precision: " + str(self.precision))

        if found_y > 0:
            print("Correct chunks "+str(correct_viterbi))
            print("Found correct "+str(found_y))
            self.recall = 100.00 * (correct_viterbi / found_y)
            print("Recall: " + str(self.recall))

        if (self.precision > 0) and (self.recall > 0):
            self.f1_score = 2.0 * self.precision * self.recall / (self.precision + self.recall)
            print("F1 score: "+str(self.f1_score))


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
        elif tag.startswith('I') and (len(chunk) == 0):
            chunk.append(tag_index)
            chunk.append(tag)
        # Inside of chunk: for 'I' continue, for 'O' end chunk, for 'B' begin new chunk
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

