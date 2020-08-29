import json
import pandas as pd

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
