
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