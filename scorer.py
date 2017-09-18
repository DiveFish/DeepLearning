from chunker import Chunker


class Scorer:

    def scores(self, viterbi_sequences, id_to_tag_dict, validation_labels, batch):
        # Get performance of model: precision, recall, f1-score
        found_guessed = 0
        found_correct = 0
        correct_chunks = 0
        chunker = Chunker()
        for seq in range(len(viterbi_sequences)):
            sequence = viterbi_sequences[seq]
            in_correct = False
            sequence_end = False
            for tag_idx in range(len(sequence)):
                guessed_tag = id_to_tag_dict.get(sequence[tag_idx])
                y_tag = id_to_tag_dict.get(validation_labels[batch][seq][tag_idx])
                if tag_idx > 0:  # no previous tag available
                    previous_guessed_tag = id_to_tag_dict.get(sequence[tag_idx-1])
                    previous_y_tag = id_to_tag_dict.get(validation_labels[batch][seq][tag_idx-1])
                else:
                    previous_guessed_tag = "O"
                    previous_y_tag = "O"

                if chunker.chunk_start(guessed_tag):
                    found_guessed += 1
                    if guessed_tag == y_tag:
                        in_correct = True
                if chunker.chunk_start(y_tag):
                    found_correct += 1

                # For each tag check whether it matches the gold standard tag
                # in_correct true if all tags in chunk processed so far matched
                if (guessed_tag == y_tag) and in_correct:
                    continue
                else:
                    in_correct = False

                if tag_idx == (len(sequence)-1):
                    sequence_end = True
                if chunker.chunk_end(previous_guessed_tag, guessed_tag, sequence_end)\
                        and chunker.chunk_end(previous_y_tag, y_tag, sequence_end) and in_correct:
                    correct_chunks += 1
                    # Reset values for next chunk
                    in_correct = False
        precision = 0.0
        if found_guessed > 0:
            print(correct_chunks)
            print(found_guessed)
            precision = 100.00 * (float(correct_chunks) / float(found_guessed))
            print("Precision: " + str(precision))

        if found_correct > 0:
            recall = 100.00 * (float(correct_chunks) / float(found_correct))

        if (precision > 0) and (recall > 0):
            f1_score = 2.0 * precision * recall / (precision + recall)
        else:
            f1_score = 0.0
            print(f1_score)

        return precision, recall, f1_score
