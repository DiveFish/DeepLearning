from chunker import Chunker


class Scorer:

    def scores(self, viterbi_sequences, id_to_tag_dict, validation_labels, batch):
        # Get performance of model: precision, recall, f1-score
        found_guessed = 0
        found_correct = 0
        correct_chunks = 0
        TP = 0
        FP = 0
        chunker = Chunker()
        counter =0
        for seq in range(len(viterbi_sequences)):
            tags = viterbi_sequences[seq]
            guessedtags = validation_labels[batch][seq]
            golden = guessedtags[0:len(tags)]
            goldentags = [id_to_tag_dict[i] for i in golden]
            viterbittags = [id_to_tag_dict[k] for k in tags]
            golden_entities = extract_namedEntites(goldentags)
            viterbi_entities = extract_namedEntites(viterbittags)
            found_guessed+=len(viterbi_entities)
            found_correct+=len(golden_entities)
            for ner in viterbi_entities:
                if (ner in golden_entities):
                    correct_chunks+=1

        f1_score = 0
        precision = 0
        print('found guessed')
        print(found_guessed)

        print('golden chunks')
        print(found_correct)
        print('correct')
        print(correct_chunks)
        if found_guessed > 0:
            precision = float(correct_chunks) / float(found_guessed)

        if found_correct > 0:

            recall = float(correct_chunks) / float(found_correct)

        if (precision > 0) and (recall > 0):
            f1_score = 2.0 * precision * recall / (precision + recall)


        return precision, recall, f1_score

def extract_namedEntites(sequence):
    chunks = []
    chunk = []
    for tagindex in range(len(sequence)):
        if sequence[tagindex].startswith('B') and len(chunk)==0:
            chunk.append(tagindex)
            chunk.append(sequence[tagindex])
        elif sequence[tagindex].startswith('I') and len(chunk)==0:
            chunk.append(tagindex)
            chunk.append(sequence[tagindex])
        elif sequence[tagindex].startswith('I'):
            chunk.append(sequence[tagindex])
        elif sequence[tagindex].startswith('O') and len(chunk)!=0:
            chunk.append(tagindex)
            chunks.append(chunk)
            chunk = []
        elif sequence[tagindex].startswith('B') and len(chunk)!=0:
            chunk.append(tagindex)
            chunks.append(chunk)
            chunk = []
            chunk.append(tagindex)
            chunk.append(sequence[tagindex])
    return chunks
