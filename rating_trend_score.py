import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from  scipy import stats
import string
from sklearn.preprocessing import KBinsDiscretizer




def rating_labels(n):
    
    alphabet = list(string.ascii_uppercase)
    
    if n == 5:
        
        labels = ['A','B','C','D','E']
        
    elif (n==10 or n==11):
        labels  = ['A1','A2',
                   'B1','B2',
                   'C1','C2',
                   'D1','D2',
                   'E1','E2']
    elif n == 15:

        labels  = ['A1','A2','A3',
                   'B1','B2','B3',
                   'C1','C2','C3',
                   'D1','D2','D3',
                   'E1','E2','E3']

    elif n == 20:

        labels  = ['A1','A2','A3','A4',
                   'B1','B2','B3','B4',
                   'C1','C2','C3','C4',
                   'D1','D2','D3','D4',
                   'E1','E2','E3','E4']

    else:
       
        labels = alphabet[:n+1]
    
    
    return labels
        

    
def number_of_rating(df,bin_,stat,prob,strategy):
    
    probs = df[[prob]].astype(float)
  
    discretizer21 = KBinsDiscretizer(n_bins=bin_, encode='ordinal', strategy=strategy)
    discretizer21.fit(probs)
    discretizer21_transf = discretizer21.transform(probs)
    
    probs01 = pd.DataFrame(probs)
    probs01 = probs01.reset_index()
    discretizer21_transf = pd.DataFrame(discretizer21_transf)
    discretizer21_transf = pd.concat([probs01, discretizer21_transf], axis=1)

    df_corte = discretizer21_transf.groupby(0, as_index = False).agg({prob:['size','min', 'mean', 'median', 'max']})
    
    
    bins_ = df_corte[prob][stat].values
    bins_ = np.append(0,bins_)    
    bins_[-1] = 1
    
    return bins_








def ajusted_rating(rating_group,rating):
    adj_rating = rating

    if (rating_group == 'B') and (rating in ['A1','A2']):
        adj_rating = 'B1'

    elif (rating_group == 'C') and (rating in ['B1','B2']):
        adj_rating = 'C1'

    elif (rating_group == 'D') and (rating in ['C1','C2']):
        adj_rating = 'D1'

    elif (rating_group == 'E') and (rating in ['D1','D2']):
        adj_rating = 'E1'

    return adj_rating












def plot_estabilidade_target(df, target, title,var,sharey=True, figsize=(30,10)): 
    df_func = df.copy()
    if (df_func[var].nunique() < 40):
        fig, axes = plt.subplots(nrows=1, ncols=2, sharey=sharey)
        fig.suptitle(title, fontsize=24)

        br_plot = df_func.groupby(by=['safra', var])[target].mean().unstack()
        vol_plot = df_func.groupby(by=['safra'])[var].value_counts(normalize=True).unstack()

        ax1 = br_plot.plot(ax=axes[0], figsize=figsize)
        ax1.legend(fontsize = 14)
        ax2 = vol_plot.plot.area(ax=axes[1], figsize=figsize)
        ax2.legend(fontsize = 14)
        
        ax1.set_facecolor("white")
        ax2.set_facecolor("white")






def trend_features(score_name,df):
    
    score_list =   []



    for name in score_list:
                
            temp_array_score1 = []
        
            temp_array_score2 = []
        
            for i in range(0,6):
            
                score_name  = name   + '_{}m'.format(str(i))
            
                temp_array_score1.append(score_name)
            
            
            agv1_score_6m = df[temp_array_score1].mean(axis = 1)
            
            
        for i in range(6,12):
            
            score_name  = name   + '_{}m'.format(str(i))
            
            temp_array_score1.append(score_name)
            
        agv1_score_12m = df[temp_array_score2].mean(axis = 1)
                    
                    
        trend_feature = 'trend_' + score +'_6m_12m             
        
        df[trend_feature] =  = get_angle_trend(agv1_score_12m,agv1_score_6m)
    
    return df

        
        
       



