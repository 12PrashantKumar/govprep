def recursive_chunk(text,chunk_size=500, overlap=50):
    """
    Recursively splits text using a hierarchy of natural boundaries 
    (paragraphs -> lines -> sentences -> words) to keep semantic thoughts intact.
    """
    seperators = ["\n\n" , "\n", "."," "]

    def split(text_to_split,seps):
        # Base cases: if text fits context window or we ran out of seperators
        if len(text_to_split) <= chunk_size or not seps:
            return [text_to_split]
        
        sep = seps[0]
        parts = text_to_split.split(sep)
        chunks, current = [], ""

        for part in parts:
            # Re-attach the separator to preserve document format structure
            piece = part + sep

            if len(current) + len(piece) <= chunk_size:
                current += piece
            else:
                if current:
                    chunks.append(current)

                # If a single piece is larger than chunk_size, recurse with a finer separator
                if len(piece) > chunk_size:
                    chunks.extend(split(piece, seps[1:]))
                    current = ""

                else:
                    current = piece

        if current:
            chunks.append(current)
        return chunks
    return split(text, seperators)