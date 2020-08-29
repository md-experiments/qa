import json
import pandas as pd

from src.utils import rebuild_text, get_entities

class NaturalQuestions():
    def __init__(self, df_qa):


        # Questions
        df_qa['first']=df_qa.question_text.apply(lambda x: x.split()[0])
        df_qa['second']=df_qa.question_text.apply(lambda x: x.split()[1])

        # Answers
        df_qa['ANS_YES_NO']=df_qa.annotations.apply(lambda x: x[0]['yes_no_answer'])

        df_qa['ANS_SHORT']=df_qa.apply(lambda x: self.extract_answer(x, long_answer=False), axis=1)
        df_qa['HAS_SHORT']=df_qa.ANS_SHORT.apply(lambda x: len(x)>0)
        df_qa['ANS_LONG']=df_qa.apply(lambda x: self.extract_answer(x, long_answer=True), axis=1)

        self.df_qa=df_qa
        self.len_df=len(self.df_qa)

    def summary(self):
        
        # len of annotations
        print('Average number of annotations',self.df_qa['annotations'].apply(lambda x: len(x)).mean())
        # len of short answers
        short_ans=self.df_qa['annotations'].apply(lambda x: \
                            len(x[0]['short_answers']) if len(x[0]['short_answers'])>0 else 0).sum()
        print(f'Short answers {short_ans}, {round(100*short_ans/self.len_df,3)}%')
        mult_short=self.df_qa['annotations'].apply(lambda x: \
                            len(x[0]['short_answers']) if len(x[0]['short_answers'])>1 else 0).sum()
        print(f'Multiple short answers {mult_short}, {round(100*mult_short/self.len_df)}%')
        # long answers
        long_answers=self.df_qa['annotations'].apply(lambda x: \
                            1 if len(x[0]['long_answer'])>0 else 0).sum()
        print(f'Long answers {long_answers}, {round(100*long_answers/self.len_df,3)}%')
        # bool answers
        print('% bool answers',self.df_qa['annotations'].apply(lambda x: \
                            1 if x[0]['yes_no_answer']!='NONE' else 0).sum()/self.len_df)

    def extract_answer(self,line, long_answer=True):
        doc=line['document_text'].split()
        answers=[]
        if long_answer:
            a_start=line['annotations'][0]['long_answer']['start_token']
            a_end=line['annotations'][0]['long_answer']['end_token']
            answer=rebuild_text(doc[a_start:a_end])
            answers.append(answer)
            answers=answers[0]
        else:
            for ans in line['annotations'][0]['short_answers']:
                a_start=ans['start_token']
                a_end=ans['end_token']
                answer=' '.join(doc[a_start:a_end])
                answers.append(answer)
        return answers

    def get_entities(self):
        self.df_qa['ENTS_SHORT']=self.df_qa.ANS_SHORT.apply(lambda x: get_entities(x) if x!='NONE' else [])