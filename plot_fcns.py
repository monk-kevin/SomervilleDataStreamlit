# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 15:15:26 2025

@author: kevjm
"""
import pandas as pd
import numpy as np
import scipy
import sklearn
import os
import matplotlib.pyplot as plt
from xgboost import plot_importance

def plot_happiness_dist(df,groupby_var):
    """
    This function takes the dataframe that houses all happiness survey answers and 
    creates stacked bar plots based on the provided groupby_var.
    
    If one variable is provided, each stack of bar represent the proportion of 
    respondents who scored their happiness as 1, 2, 3, 4, or 5 for that given 
    variable.
    
    If, however, two variables are provided (e.g., [Ward, Year]), then a series of 
    subplots are made for all of the options for the first variable and then split 
    by the second wihtin stacked bars
    """
    my_colors = ['#b0b0b0','#758a9b','#ffd366','#f6a437','#ec7014']
    grouped_scores = (df.groupby(groupby_var)
                      ['Happiness.5pt.num']
                      .value_counts()
                     )
    
    if len(groupby_var) > 2:
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(1,1,1)
        grouped_scores_norm = (grouped_scores.unstack()
                               .iloc[:,0:5]
                               .div(grouped_scores
                                    .unstack()
                                    .sum(axis=1),
                                    axis = 0)
                              )
        grouped_scores_norm.plot(ax = ax,
                                 kind = 'barh',
                                 stacked = True,
                                 color = my_colors,
                                 legend = True)
        ax.legend(loc='center left',bbox_to_anchor = (1.05,0.5),title='Happiness Score')
        ax.set_ylabel(f'{groupby_var}')
        ax.set_xlabel('Prop Respondents')
        ax.set_title(f'Happiness Score Across {groupby_var}')
    else:
        lvl1_vals = sorted(list(set(grouped_scores.index.get_level_values(0))))
        fig,axes = plt.subplots(1,len(lvl1_vals))
        c = 0
        for val in lvl1_vals:
            grouped_scores_sub = grouped_scores.xs(val)
            grouped_scores_sub_norm = (grouped_scores_sub.unstack()
                                       .iloc[:,0:5]
                                       .div(grouped_scores_sub
                                            .unstack()
                                            .sum(axis=1),
                                            axis = 0)
                                      )
            
            # plt.subplot(1,len(lvl1_vals),c+1)
            grouped_scores_sub_norm.plot(ax = axes[c], 
                                         kind = 'barh',
                                         stacked = True, 
                                         color = my_colors,
                                         legend = False)
            axes[c].set_xlabel(groupby_var[1])
            if c == 0:
                axes[c].set_ylabel('Year')
            else:
                axes[c].set_ylabel('')
            
            axes[c].set_title(f'{groupby_var[0]}: {val}')
            c += 1
        fig.set_figwidth(30)
    
    return fig
    
def plot_mn_happiness(df,col_feature,col_score = 'Happiness.5pt.num',sem_flag = False):
    df_grp = (df[[col_feature,col_score]]
          .groupby([col_feature])
          .agg(['mean','median','std','count'])
          .reset_index()
         )
    fig, ax = plt.subplots()
    
    if sem_flag:
        plt.errorbar(df_grp[col_feature],df_grp[col_score,'mean'],
             df_grp[col_score,'std'] / (df_grp[col_score,'count'])**0.5,
             color = 'k')
    else:
        plt.errorbar(df_grp[col_feature],df_grp[col_score,'mean'],
             df_grp[col_score,'std'],
             color = 'k')

    ax.set_ylim(0,5.5)
    ax.set_xlabel(col_feature)

    if col_score == 'Happiness.5pt.num':
        ax.set_ylabel('Mean Happiness Score')
    else:
        ax.set_ylabel(f'Mean ({col_score})')
        
    return fig


def plot_feature_importance_column(xgb_mdl,importance_types,max_features = 5):
    '''
    plots the feature importance for an XGBoost model (xgb_mdl) and creates a set of vertically stacked
    feature importance plots for each importance_type. By default will plot the top 10 features
    '''
    fig, axs = plt.subplots(len(importance_types),figsize = (10,15))
    xlbl=''
    for idx, imp_type in enumerate(importance_types):
        if idx == 2:
            xlbl = 'Importance Score'
        plot_importance(xgb_mdl,
                        ax = axs[idx],
                        importance_type=imp_type,
                        max_num_features = max_features,
                        title = 'Importance Type: ' + imp_type,
                        show_values = False,
                        xlabel = xlbl
                       )    
        
    return fig