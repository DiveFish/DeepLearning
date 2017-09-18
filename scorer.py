from chunker import Chunker


class Scorer:

    def scores(self, viterbi_sequences, id_to_tag_dict, validation_labels, batch):
        # Get performance of model: precision, recall, f1-score
        found_guessed = 0
        found_correct = 0
        correct_chunks = 0
        chunker = Chunker()
        for seq in range(len(viterbi_sequences)):
            tags = viterbi_sequences[seq]
            print("predicted")
            print(tags)
            print('gold')
            print(validation_labels[batch][seq])

            inside_chunk = False
            for tag_idx in range(len(tags)):
                guessed_tag = id_to_tag_dict.get(tags[tag_idx])
                y_tag = id_to_tag_dict.get(validation_labels[batch][seq][tag_idx])
                sequence_end = False
                if  tag_idx == len(viterbi_sequences):
                    sequence_end = True
                    previous_guessed = id_to_tag_dict.get(tags[tag_idx-1])
                    previous_gold = id_to_tag_dict.get(validation_labels[batch][seq][tag_idx-1])
                else:
                    previous_guessed = id_to_tag_dict.get(tags[tag_idx])
                    previous_gold = id_to_tag_dict.get(validation_labels[batch][seq][tag_idx])
                if chunker.chunk_start(guessed_tag):
                    found_guessed += 1
                    inside_chunk = True
                    if guessed_tag == y_tag:
                        in_correct = True
                if chunker.chunk_start(y_tag):
                    found_correct += 1
                # For each tag check whether it matches the gold standard tag
                # in_correct true if all tags in chunk processed so far matched
                if inside_chunk and (guessed_tag == y_tag) and in_correct:
                    continue
                else:
                    in_correct = False

                if chunker.chunk_end(previous_guessed,guessed_tag, sequence_end) and chunker.chunk_end(previous_gold,y_tag, sequence_end) and in_correct:
                    correct_chunks += 1
                    # Reset values for next chunk
                    inside_chunk = False
                    in_correct = False
        print('correct')
        print(correct_chunks)
        f1_score = 0
        precision = 0
        if found_guessed > 0:
            print('prescision')
            print(correct_chunks)
            print(found_guessed)
            precision = 100.00 * (float(correct_chunks) / float(found_guessed))
            print(precision)

        if found_correct > 0:
            recall = 100.00 * (float(correct_chunks) / float(found_correct))

        if (precision > 0) and (recall > 0):
            f1_score = 2.0 * precision * recall / (precision + recall)
            print(f1_score)

        return precision, recall, f1_score