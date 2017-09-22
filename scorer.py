from chunker import Chunker


class Scorer:

    def scores(self, viterbi_sequences, id_to_tag_dict, validation_labels, batch):
        # Get performance of model: precision, recall, f1-score
        found_viterbi = 0
        correct_viterbi = 0
        found_y = 0  # i.e. correct_y
        chunker = Chunker()
        for seq in range(len(viterbi_sequences)):
            viterbi_tag_idxs = viterbi_sequences[seq]
            viterbi_tags = [id_to_tag_dict[i] for i in viterbi_tag_idxs]
            viterbi_entities = chunker.extract_named_entities(viterbi_tags)

            y_tag_idxs = validation_labels[batch][seq][0:len(viterbi_tag_idxs)]
            y_tags = [id_to_tag_dict[j] for j in y_tag_idxs]
            y_entities = chunker.extract_named_entities(y_tags)

            found_viterbi += len(viterbi_entities)
            found_y += len(y_entities)
            for ner in viterbi_entities:
                if ner in y_entities:
                    correct_viterbi += 1

        precision = 0.0
        if found_viterbi > 0:
            print("Correct chunks "+str(correct_viterbi))
            print("Found guessed "+str(found_viterbi))
            precision = 100.00 * (correct_viterbi / found_viterbi)
            print("Precision: " + str(precision))

        if found_y > 0:
            print("Correct chunks "+str(correct_viterbi))
            print("Found correct "+str(found_y))
            recall = 100.00 * (correct_viterbi / found_y)
            print("Recall: " + str(recall))

        if (precision > 0) and (recall > 0):
            f1_score = 2.0 * precision * recall / (precision + recall)
        else:
            f1_score = 0.0
            print("F1 score: "+str(f1_score))

        return precision, recall, f1_score
