
class Chunker:

    def chunk_start(self, current_tag):
        if current_tag.startswith("B"):
            return True
        else:
            return False

    def chunk_end(self, previous_tag, current_tag, sequence_end):
        if sequence_end and (current_tag.startswith("B") or current_tag.startswith("I")):
            return True
        else:
            if (previous_tag.startswith("B") or previous_tag.startswith("I")) and (current_tag.startswith("B") or current_tag.startswith('O')):
                return True
            else:
                return False

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

    if __name__ == '__main__':
        seq = ['B-loc', 'B-loc', 'I-o', 'O', 'O', 'B-I', 'I-o', 'I-o', 'B-o', 'O']
        seq2 = ['B-loc', 'B-loc', 'B-loc', 'I-o', 'O', 'O', 'B-I', 'I-o', 'I-o', 'B-o', 'O']


        k = extract_namedEntites(seq2)
        k2 = extract_namedEntites(seq)
        for ner in k:
            if (ner in k2):
                print(ner)

        print(k)
        print(k2)

