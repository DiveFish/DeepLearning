# Authors: Neele Witte, 4067845; Patricia Fischer, 3928367
# Honor Code:  We pledge that this program represents our own work.

import numpy as np
from chunker import Chunker

# Class to calculate precision, recall and f1 score
# for the best tag sequence determined by the viterbi decoder.
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
            chunker = Chunker()
            viterbi_entities = chunker.extract_named_entities(viterbi_tags)

            y_tag_idxs = validation_labels[batch][seq][:len(viterbi_tag_idxs)]
            y_tags = [id_to_tag_dict[j] for j in y_tag_idxs]
            y_entities = chunker.extract_named_entities(y_tags)

            found_viterbi += len(viterbi_entities)
            found_y += len(y_entities)
            for ner in viterbi_entities:
                if ner in y_entities:
                    correct_viterbi += 1

        if found_viterbi > 0:
            print("Found (correct) "+str(correct_viterbi))
            print("Found (guessed) "+str(found_viterbi))
            self.precision = 100.00 * (correct_viterbi / found_viterbi)
            print("Precision: " + str(self.precision))

        if found_y > 0:
            print("Found (correct) "+str(correct_viterbi))
            print("Actual correct "+str(found_y))
            self.recall = 100.00 * (correct_viterbi / found_y)
            print("Recall: " + str(self.recall))

        if (self.precision > 0) and (self.recall > 0):
            self.f1_score = 2.0 * self.precision * self.recall / (self.precision + self.recall)
            print("F1 score: "+str(self.f1_score))


