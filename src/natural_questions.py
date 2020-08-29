import json
import pandas as pd

from src.utils import rebuild_text

class NaturalQuestions():
    def __init__(self, df_qa):
        self.df_qa=df_qa

        # Questions
        self.df_qa['first']=self.df_qa.question_text.apply(lambda x: x.split()[0])
        self.df_qa['second']=self.df_qa.question_text.apply(lambda x: x.split()[1])

        # Answers
        

    def summary(self):
        # len of annotations
        print('Average number of annotations',self.df_qa['annotations'].apply(lambda x: len(x)).mean())
        # len of short answers
        print('% short answers',self.df_qa['annotations'].apply(lambda x: \
                            len(x[0]['short_answers']) if len(x[0]['short_answers'])>0 else 0).sum()/10000)
        print('% multiple short answers',self.df_qa['annotations'].apply(lambda x: \
                            len(x[0]['short_answers']) if len(x[0]['short_answers'])>1 else 0).sum()/10000)
        # long answers
        print('% long answers',self.df_qa['annotations'].apply(lambda x: \
                            1 if len(x[0]['long_answer'])>0 else 0).sum()/10000)
        # bool answers
        print('% bool answers',self.df_qa['annotations'].apply(lambda x: \
                            1 if x[0]['yes_no_answer']!='NONE' else 0).sum()/10000)

    def extract_answer(line, long_answer=True):
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