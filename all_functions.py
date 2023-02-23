# import pyathena

import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from  scipy import stats
import time
from sklearn.preprocessing import LabelEncoder
import os
from dotenv import load_dotenv
from datetime import date
from sqlalchemy import create_engine
import tempfile

# import pandasql as ps

import boto3
import awswrangler as wr
import logging, os
import warnings


# from utils.data_prep.data_prep_open2 import data_prep

import numpy as np
import pandas as pd
import warnings


def get_threshold(max_corte,add):
    
    corte = []

    i = 0
    while i <= max_corte:   
        i = i+add
    
        corte.append(round(i,2))
    return corte
    
def info_thresholds(corte,df,prob,target):
    
    corte = np.float_(corte)
    df_temp = df
    
    df_temp['apr'] = np.where(df_temp[prob] <= corte,1,0)
    
    
    sum_apr   = np.sum(df_temp['apr'])
    vol_total = len(df_temp)
    mean_fpd  = df_temp[target].mean()
    
    _target = df_temp[df_temp['apr'] ==  1][target].mean()
    
    sum_tv_fpd = df_temp[(df_temp[target] ==  1) & (df_temp['apr'] ==  1)]['tv'].sum()
    
    perda_total_fpd = df_temp[df_temp[target] ==  1]['tv'].sum()
    
    perc_fpd_tv = - (1 - (sum_tv_fpd/perda_total_fpd))
        
    reducao_loanvalueinternal =  df_temp[df_temp['apr'] ==  0]['loanvalueinternal'].sum()
    
    li = -reducao_loanvalueinternal /  df_temp['loanvalueinternal'].sum() 
       
    
    reducao_base = (sum_apr / vol_total) - 1
    
    
    valor_fpd30_monetario = df_temp[df_temp['apr'] ==  1]['tv'].sum()
    
    perc_fpd30_monetario = sum_tv_fpd / valor_fpd30_monetario
    
    
    
    
    
    if mean_fpd == 0:
        reducao_fpd = 0
    else:
        reducao_fpd = ( _target - mean_fpd) / mean_fpd
    
    target
    
    df_ = pd.DataFrame({'corte':[corte],
           'aprovados':[sum_apr],
           'total_base':[vol_total],
           'reducao_base':[reducao_base],
            target:[_target],
           'reducao_fpd':reducao_fpd,
           'perc_fpd30_monetario':[perc_fpd30_monetario],           
           'sum_tv_fpd':[sum_tv_fpd],
           'perc_fpd_tv':[perc_fpd_tv],
           'reducao_loanvalueinternal':[reducao_loanvalueinternal],
           'li':[li]})
    
    df_ = df_.fillna(0)
    
    return df_



def simulator_thresholds(df,target,prob,corte):
    
    
    df_all_cortes = pd.DataFrame([],columns = ['corte','aprovados','total_base','reducao_base',target,'reducao_fpd','perc_fpd30_monetario',
                                          'sum_tv_fpd',	'perc_fpd_tv','reducao_loanvalueinternal',
                                           'li'])
    
    
    


    for corte_ in corte:

        df_temp = info_thresholds(corte = corte_,df = df,prob = prob,target = target)
        df_all_cortes = df_all_cortes.append(df_temp)

    
    return df_all_cortes.reset_index(drop = True)



def info_thresholds_amount(corte,df,prob):
    
    corte = np.float_(corte)
    df_temp = df
    
    df_temp['apr'] = np.where(df_temp[prob] <= corte,1,0)
    
    
    sum_apr   = np.sum(df_temp['apr'])
    vol_total = len(df_temp)
    
    reducao_base = (sum_apr / vol_total) - 1
        

    
    df_ = pd.DataFrame({'corte':[corte],
                        'aprovados':[sum_apr],
                        'total_base':[vol_total],
                        'reducao_base':[reducao_base]})
                       
    df_ = df_.fillna(0)
    
    return df_



def simulator_thresholds_time(df,corte,target,prob,safra):
    
    prb = []
    vol_reducao_base = []
    target_append = []
    reducao_fpd = []
    safra_final = []
    sum_tv_fpd = []
    perc_fpd_tv = []
    reducao_loanvalueinternal = []
    li = []
    perc_fpd30_monetario = []
    
    df[safra] = pd.to_datetime(df[safra])
    safras = df[safra].unique()
    
    
    for safra_ in safras:
        
        df_temp = df[df[safra] == safra_].reset_index(drop = True)
        
        df_temp_threshold = info_thresholds(df = df_temp,target = target,prob = prob,corte = corte)
        
        df_temp_threshold = df_temp_threshold.reset_index(drop = True)
        
        
        
        
        prb.append(df_temp_threshold['reducao_base'][0])
        
        
        vol_reducao_base.append(df_temp_threshold['total_base'][0] - df_temp_threshold['aprovados'][0])
        
        
        target_append.append(df_temp_threshold[target][0])
        reducao_fpd.append(df_temp_threshold['reducao_fpd'][0])
        safra_final.append(safra_)
        sum_tv_fpd.append(df_temp_threshold['sum_tv_fpd'][0])
        perc_fpd_tv.append(df_temp_threshold['perc_fpd_tv'][0])
        reducao_loanvalueinternal.append(df_temp_threshold['reducao_loanvalueinternal'][0])
        li.append(df_temp_threshold['li'][0])
        perc_fpd30_monetario.append(df_temp_threshold['perc_fpd30_monetario'][0])        
        
        
        
        
        
        
    df_safra_final = pd.DataFrame({'safra':safra_final,
                                 'vol_reducao_base':vol_reducao_base,
                                 'prb':prb,
                                  target:target_append,
                                 'reducao_fpd':reducao_fpd,
                                 'perc_target_monetario':perc_fpd30_monetario,
                                 'sum_tv_fpd':sum_tv_fpd,
                                 'perc_fpd_tv':perc_fpd_tv,
                                 'reducao_loanvalueinternal':reducao_loanvalueinternal,
                                 'li':li})
    
    
    
    return df_safra_final   





def plot_line_time_money(df_time,cortes_principais,brand):
    
    fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(5,3))
    fig.set_size_inches(18,15)
    title_a = ' Monetário em R$'
    title_b = 'Redução FPD30 Monetário em %'
    title_c = 'Redução de Principal em R$'
    title_d = 'Redução de Principal em %'
    sns.set(font_scale=1.2) 
    df_time = df_time[df_time['safra'] > '2020-01']
    df_time['perc_fpd_tv'] = np.abs(df_time['perc_fpd_tv'])
    df_time['li'] = np.abs(df_time['li'])
    
    df_time['perc_target_monetario'] = df_time['perc_target_monetario'].astype(float)
     
    corte_leg = []
    for corte in cortes_principais:
        
        
        df_aux = df_time[df_time['corte'] == corte]        
        

        ax1 = df_aux.plot(ax = axes[0,0],x = 'safra', y = 'perc_fpd30_monetario',title = title_a)
        ax2 = df_aux.plot(ax = axes[0,1],x = 'safra', y = 'perc_fpd_tv',title = title_b)

        ax3 = df_aux.plot(ax = axes[1,0],x = 'safra', y = 'reducao_loanvalueinternal',title = title_c)
        ax4 = df_aux.plot(ax = axes[1,1],x = 'safra', y = 'li',title = title_d)
        
        corte_leg.append('corte: ' + str(corte))

  
    ax1.legend(corte_leg)
    ax1.set_xlabel('safra', fontsize=14)
    ax1.tick_params(axis="x", labelsize=8)
    
    ax2.legend(corte_leg)
    ax2.set_xlabel('safra', fontsize=14)
    ax2.tick_params(axis="x", labelsize=8)
    
    
    #ax3.set(ylim=(0,0.3))
    ax3.legend(corte_leg)
    ax3.set_xlabel('safra', fontsize=8)
    ax3.tick_params(axis="x", labelsize=8)
    
    ax4.legend(corte_leg)
    ax4.set_xlabel('safra', fontsize=14)
    ax4.tick_params(axis="x", labelsize=8)

    
    fig.suptitle(brand, fontsize=16)

    
    
    
    
def get_df_time_all_thresholds(df_fpd30,cortes_principais,target,prob):
    
    
    df_time_final = pd.DataFrame([],columns = ['safra', 'vol_reducao_base', 'prb', target,'perc_target_monetario',
       'reducao_target', 'sum_tv_target', 'perc_target_tv',
       'reducao_li', 'li','corte'])
    
    for corte in cortes_principais: 
        
        df_time = simulator_thresholds_time(df = df_fpd30,target = target,prob = prob,corte = corte, safra = 'safra')
        
        df_time['corte'] = corte
        df_time_final = df_time_final.append(df_time)
        
        del df_time
        
        
        
    return df_time_final





def plot_line_time_all_corte(df_time,cortes_principais,target,brand):
    
    fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(5,3))
    fig.set_size_inches(18,15)
    title_a = 'Redução da base em volume Absoluto'
    title_b = 'Redução da base em %'
    title_c = target + ' em %'
    title_d = 'Reducão do '+ target + 'em %'
    sns.set(font_scale=1.2) 
   
    
    df_time['prb'] = np.abs(df_time['prb'])
    df_time['reducao_fpd'] = np.abs(df_time['reducao_fpd'])
    
    corte_leg = []
    for corte in cortes_principais:
        
        df_aux = df_time[df_time['corte'] == corte]
        
        ax1 = df_aux.plot(ax = axes[0,0],x = 'safra', y = 'vrb',title = title_a)
        ax2 = df_aux.plot(ax = axes[0,1],x = 'safra', y = 'prb',title = title_b)
    
        ax3 = df_aux.plot(ax = axes[1,0],x = 'safra', y = target,title = title_c)
        ax4 = df_aux.plot(ax = axes[1,1],x = 'safra', y = 'rf',title = title_d)
        corte_leg.append('corte: ' + str(corte))
    
    ax1.legend(corte_leg)
    ax1.set(ylim=(0,500))
    ax1.set_xlabel('safra', fontsize=14)
    ax1.tick_params(axis="x", labelsize=8)
    
    ax2.legend(corte_leg)
    ax2.set(ylim=(0,0.5))
    ax2.set_xlabel('safra', fontsize=14)
    ax2.tick_params(axis="x", labelsize=8)
    
    ax3.legend(corte_leg)
    ax3.set(ylim=(0,0.5))
    ax3.set_xlabel('safra', fontsize=8)
    ax3.tick_params(axis="x", labelsize=8)
    
    ax4.legend(corte_leg)
    ax4.set(ylim=(0,0.5))
    ax4.set_xlabel('safra', fontsize=14)
    ax4.tick_params(axis="x", labelsize=8)
    
    fig.suptitle(brand, fontsize=16)