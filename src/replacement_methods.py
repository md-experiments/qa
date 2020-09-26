from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
import pandas as pd
import matplotlib.pyplot as plt

from src.utils import TimeClass, simple_time_stamp
from src.entity_actions import reduce_to_matching_ent_by_type
from src.statistical_distributions import beta_pdf_vector

def get_similarities(df, model, columns=['question1','question2']):
    embeddings1 = model.encode(list(df[columns[0]].values), batch_size=16, 
                                       show_progress_bar=False, convert_to_numpy=True)
    embeddings2 = model.encode(list(df[columns[1]].values), batch_size=16, 
                                       show_progress_bar=False, convert_to_numpy=True)
    return 1 - (paired_cosine_distances(embeddings1, embeddings2))

def compare_replacement_methods(df,
                                model,
                                experiments={
                                    'no change':{'stand_in':None,'col_name':''},
                                    'remove':{'stand_in':[''],'col_name':'_remove'},
                                },
                                ent_type='ORG'):
    """Creates plots for the distribution of cosine similarities 
    between embeddings using different replacement strategies for a certain entity type
    
    
    """
    df=df.copy()
    t=TimeClass()
    ls_stats=[]
    col_stats=['Time (secs)','Nr Samples','Nr Duplicates','Avg Duplicates','Variance Duplicates','Nr Negative','Avg Negative','Variance Negative']
    for e in experiments:
        df=reduce_to_matching_ent_by_type(df,
                                          ent_type,
                                          experiments[e]['stand_in'],
                                          mask_col_ending=experiments[e]['col_name'])
        similarity_col=f'similarity{experiments[e]["col_name"]}'
        df[similarity_col] = get_similarities(df, 
                  model, 
                  columns=[f"question1{experiments[e]['col_name']}",
                           f"question2{experiments[e]['col_name']}"])


        secs,mins = t.take()
        nr_samples=len(df)
        df_pos=df[df.is_duplicate==1][similarity_col]
        df_neg=df[df.is_duplicate==0][similarity_col]
        
        ls_stats.append([secs,nr_samples,len(df_pos),df_pos.mean(),df_pos.var(),len(df_neg),df_neg.mean(),df_neg.var(),])

        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(16,2))
        fig.suptitle(f'Comparing {e}, took {mins} minutes, {nr_samples} samples',size=12)
        df_pos.plot.hist(bins=100, alpha=0.7, ax=axes, color='blue')
        pd.DataFrame(beta_pdf_vector(df_pos.mean(),df_pos.var(),len(df_pos)),
                     index=[x/100 for x in range(101)],
                     columns=['Duplicate']).plot(ax=axes, color='midnightblue')
        df_neg.plot.hist(bins=100, alpha=0.7, ax=axes, color='red')
        pd.DataFrame(beta_pdf_vector(df_neg.mean(),df_neg.var(),len(df_neg)), 
                     index=[x/100 for x in range(101)], 
                     columns=['Non-Duplicate']).plot(ax=axes, color='red')
        
    df_stats=pd.DataFrame(ls_stats,columns=col_stats)
    dt=simple_time_stamp()
    df_stats.to_csv(f'exp_{ent_type}_{dt}.csv')
    return df, df_stats