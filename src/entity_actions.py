import pandas as pd
import matplotlib.pyplot as plt
from src.utils import TimeClass

def find_ents(quora):
    quora=quora.copy()
    quora['q1_ent']=quora.question1.apply(lambda x: get_entities(x))
    quora['q2_ent']=quora.question2.apply(lambda x: get_entities(x))
    return quora

def replace_list_ents(txt,ent_ls,stand_in_ls=None, lower=True):
    """
    Replace a list of entities mentioned in a string with a list of stand-in entities
    
    Args:
        txt (string): the string to replace
        ent_ls (list): the list of entities to replace, list of strings
        stand_in_ls (list, optional): list of stand in values to replace entities with, defaults to None which leaves txt unchanged
        lower (bool, optional): whether to lowercase everything, defaults to True

    >>> replace_list_ents('Bob met Sally',['Bob','Sally'],['man','woman'])
    'man met woman'
    >>> replace_list_ents('Bob met Sally',['Bob','Sally'],['man'])
    'man met man'
    >>> replace_list_ents('Bob met Sally',['Bob','Sally'])
    'Bob met Sally'
    """
    if not isinstance(txt,str):
        txt=str(txt)

    if lower:
        txt=txt.lower()
    if not stand_in_ls==None:
        assert(isinstance(stand_in_ls,list))
        if len(stand_in_ls)<len(ent_ls):
            stand_in_ls = stand_in_ls*(int(len(ent_ls)/len(stand_in_ls))+1)

        for (e,s) in zip(ent_ls,stand_in_ls):
            if lower:
                txt=txt.replace(e.lower(),s)
            else:
                txt=txt.replace(e,s)
    return txt

def get_ent_by_type(ent_ls, ent_type):
    """
    Return a list of matching entities of a certain type from list of entities
    
    >>> ent=[{'text': 'Quora', 'label': 'PERSON'}, {'text': 'Google', 'label': 'ORG'}]
    >>> get_ent_by_type(ent, 'ORG')
    ['Google']
    """
    ent_match_ls=[ent['text'] for ent in ent_ls if ent['label']==ent_type]
    return ent_match_ls

def overlap_entity_by_type(ent_ls1, ent_ls2, ent_type):
    """
    Return list of exactly matching entities of a certain type between a pair of entities
    
    >>> ent=[{'text': 'Quora', 'label': 'PERSON'}, {'text': 'Google', 'label': 'ORG'}]
    >>> overlap_entity_by_type(ent, ent, 'ORG')
    ['Google']
    """
    ent_match1=get_ent_by_type(ent_ls1, ent_type)
    ent_match2=get_ent_by_type(ent_ls2, ent_type)
    ent_match_ls=list(set(ent_match1) & set(ent_match2))
    return ent_match_ls

def reduce_to_matching_ent_by_type(df, ent_type, stand_in_ls, columns=['question1','question2','q1_ent','q2_ent'], mask_col_ending='_mask'):
    """
    For a pair of sentences and their entities, find all exactly matching entities in both 
    and replace them with stand-ins to create a new pair of 'masked' sentences
    
    >>> df=pd.DataFrame([['bob one','bob two', [{'text': 'bob', 'label': 'PERSON'}], [{'text': 'bob', 'label': 'PERSON'}]]],columns=['s1','s2','e1','e2'])
    >>> reduce_to_matching_ent_by_type(df, 'PERSON', ['mask1','mask2'], columns=['s1','s2','e1','e2'])
            s1       s2                                    e1                                    e2 ent_overlap    s1_mask    s2_mask
    0  bob one  bob two  [{'text': 'bob', 'label': 'PERSON'}]  [{'text': 'bob', 'label': 'PERSON'}]       [bob]  mask1 one  mask1 two
    """

    df=df.copy()
    df['ent_overlap']=df.apply(lambda x: overlap_entity_by_type(x[columns[2]], x[columns[3]], ent_type),axis=1)
    df[f'{columns[0]}{mask_col_ending}']=df.apply(lambda x: replace_list_ents(x[columns[0]], x['ent_overlap'], stand_in_ls),axis=1)
    df[f'{columns[1]}{mask_col_ending}']=df.apply(lambda x: replace_list_ents(x[columns[1]], x['ent_overlap'], stand_in_ls),axis=1)
    df_res=df[df.ent_overlap.apply(lambda x: len(x)>0)]
    return df_res
