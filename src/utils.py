import json
import pandas as pd
import spacy
import copy
import itertools

nlp = spacy.load("en_core_web_sm")

def read_data(file_name='simplified-nq-train.jsonl', length=10000):
    lines=[]

    with open(file_name,'r') as f:
        for i in range(length):
            l=json.loads(f.readline())
            lines.append(l)

    df=pd.DataFrame(lines)
    return df

def rebuild_text(ls, 
                 punc_leading="% . , : ' '' ; )",punc_lagging='`` (',
                 html='<P> </P> <H3> <Li> </Li> </Tr> <Td> <Ul>'):
    html=html.split()
    ls=[l for l in ls if l not in html]
    txt=' '.join(ls)
    punc_leading=punc_leading.split()
    for p in punc_leading:
        txt=txt.replace(' '+p,p)
    punc_lagging=punc_lagging.split()
    for p in punc_lagging:
        txt=txt.replace(p+' ',p)    
    return txt

def get_entities(txt):
    """
    Extracts spacy entities from txt

    Args:
        txt (string or list): string or list of strings to extract entities from
    Returns:
        list of dictionaries

    >>> get_entities(['Malia Obama','Michelle Obama']) # Didn't work with Barack or Sasha for some reason, TODO talk to Matthew about it
    [{'text': 'Malia Obama', 'label': 'PERSON'},{'text': 'Michelle Obama', 'label': 'PERSON'}]
    >>> get_entities('Michelle Obama')
    [{'text': 'Michelle Obama', 'label': 'PERSON'}]
    >>> get_entities(['Barack Obama'])
    []
    """

    if isinstance(txt,str):
        txt=[txt]
    res=[]
        
    for t in txt:
        doc=nlp(t)

        for e in doc.ents:
            res.append({'text':e.text, 'label':e.label_})
    return res

def flatten_list(res):
    res=copy.deepcopy(res)
    return list(itertools.chain.from_iterable(res))