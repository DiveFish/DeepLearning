
class Chunker:

    """
    Extract all named entities from a sentence and store them with their
    start and end index to match them against the gold standard named entities.
    """
    def extract_named_entities(self, sequence):
        chunks = []
        chunk = []
        for tag_index in range(len(sequence)):
            # Beginning of chunk
            if sequence[tag_index].startswith("B") and len(chunk) == 0:
                chunk.append(tag_index)
                chunk.append(sequence[tag_index])
            elif sequence[tag_index].startswith("I") and len(chunk) == 0:
                chunk.append(tag_index)
                chunk.append(sequence[tag_index])
            # Inside of chunk: for "I" continue, for "O" end chunk, for "B" begin new chunk
            elif sequence[tag_index].startswith("I"):
                chunk.append(sequence[tag_index])
            elif sequence[tag_index].startswith("O") and len(chunk) > 0:
                chunk.append(tag_index)
                chunks.append(chunk)
                chunk = []
            elif sequence[tag_index].startswith("B") and len(chunk) > 0:
                chunk.append(tag_index)
                chunks.append(chunk)
                chunk = []
                chunk.append(tag_index)
                chunk.append(sequence[tag_index])
        return chunks

    if __name__ == "__main__":
        seq = ["B-loc", "B-loc", "I-o", "O", "O", "B-I", "I-o", "I-o", "B-o", "O"]
        seq2 = ["B-loc", "B-loc", "B-loc", "I-o", "O", "O", "B-I", "I-o", "I-o", "B-o", "O"]

        k = extract_named_entities(seq)
        k2 = extract_named_entities(seq2)
        for ner in k:
            if ner in k2:
                print(ner)

        print(k)
        print(k2)
