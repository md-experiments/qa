from spacy.lang.en.stop_words import STOP_WORDS

def drop_words(txt,ls_words):
    """
    Removes a set list of words from text.
    >>> drop_words("I'd like to say some",['say'])
    "I'd like to some"
    """
    txt_split=txt.split()
    txt_res=[t for t in txt_split if t not in ls_words]
    txt_res=' '.join(txt_res)
    return txt_res

def drop_stopwords(txt):
    """
    Removes all stopwords from text. Stopwords based on SpaCy
    >>> drop_words("I'd like to say some",list(STOP_WORDS))
    "I'd like"
    """
    return drop_words(txt,list(STOP_WORDS))
    
