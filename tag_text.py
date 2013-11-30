import nltk
def tag_text(text): 
    """ Tag text as entity or non entity. This could chnage to use word embeddings if needed."""
    tagged_text = []
    for sent in nltk.sent_tokenize(text):
       for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent)), binary=True):
            if hasattr(chunk, 'node'):
                tagged_text.append((' '.join(c[0] for c in chunk.leaves()), chunk.node))
            else:
                tagged_text.append((chunk[0], 'O'))
    return tagged_text
