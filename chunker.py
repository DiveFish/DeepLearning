
class Chunker:

    def chunk_start(self, current_tag):
        if current_tag.startsWith("B"):
            return True
        else:
            return False

    def chunk_end(self, previous_tag, current_tag, sequence_end):
        if sequence_end and (current_tag.startsWith("B") or current_tag.startsWith("I")):
            return True
        else:
            if ((previous_tag.startsWith("B") or previous_tag.startsWith("I"))
                and (current_tag.startsWith("B") or current_tag.startsWith('O'))):
                return True
            else:
                return False