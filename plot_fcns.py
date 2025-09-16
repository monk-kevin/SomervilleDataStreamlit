# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 15:15:26 2025

@author: kevjm
"""
import streamlit as st
import pandas as pd
import numpy as np
import scipy
import sklearn
import os
import matplotlib.pyplot as plt
from xgboost import plot_importance
import altair as alt


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


def plot_happiness_dist_altair(df, groupby_var):
    """
    Creates stacked bar plots using Altair based on the provided groupby_var.
    If one variable is provided, a single stacked bar chart is returned.
    If two variables are provided, a faceted chart is returned.
    """
    # Prepare data: count and normalize
    df_plot = (
        df.groupby(groupby_var + ['Happiness.5pt.num'])
          .size()
          .reset_index(name='count')
    )

    # Normalize within each group
    df_plot['total'] = df_plot.groupby(groupby_var)['count'].transform('sum')
    df_plot['proportion'] = df_plot['count'] / df_plot['total']

    # Convert happiness score to string for categorical axis
    df_plot['Happiness.5pt.num'] = df_plot['Happiness.5pt.num'].astype(str)

    # Define color scale
    happiness_colors = ['#b0b0b0','#758a9b','#ffd366','#f6a437','#ec7014']
    color_scale = alt.Scale(domain=['1','2','3','4','5'], range=happiness_colors)

    # Base chart
    base = alt.Chart(df_plot).mark_bar().encode(
        x=alt.X('proportion:Q', title='Proportion of Respondents'),
        y=alt.Y(f'{groupby_var[-1]}:N', title=groupby_var[-1]),
        color=alt.Color('Happiness.5pt.num:N', title='Happiness Score', scale=color_scale),
        tooltip=['Happiness.5pt.num', 'proportion', 'count']
    )

    # Facet if two grouping variables
    if len(groupby_var) == 2:
        chart = base.facet(
            row=alt.Row(f'{groupby_var[0]}:N', title=groupby_var[0])
        ).properties(
            title=f'Happiness Score Across {groupby_var[0]} and {groupby_var[1]}'
        )
    else:
        chart = base.properties(
            title=f'Happiness Score Across {groupby_var[0]}'
        )
    st.dataframe(df_plot)
    return chart.configure_axis(labelFontSize=12, titleFontSize=14).configure_title(fontSize=16)
    
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

    ax.set_ylim(0.5,5.5)
    ax.set_xlabel(col_feature)
    if col_feature == 'Year':
        ax.set_xlim(2010,2024)
        ax.set_xticks(range(2011,2024,2))
    else:
        ax.set_xlim(df[col_feature].min()-.5,df[col_feature].max()+.5)
        ax.set_xticks(np.arange(df[col_feature].min(),
                                df[col_feature].max()+1,
                                step = 1.0
                                )
                      )
    

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

def get_combined_importance(model, top_n = 5, weights = None):
    """
    Returns a DataFrame with normalized weight, gain, cover, and a combined importance score.
    `weights` is a dict like {'weight': 0.2, 'gain': 0.5, 'cover': 0.3}
    """
    booster = model.get_booster()
    importance_types = ['weight', 'gain', 'cover']

    # Extract raw importance scores
    raw_scores = {
        imp_type: booster.get_score(importance_type=imp_type)
        for imp_type in importance_types
    }

    # Create unified DataFrame
    df = pd.DataFrame.from_dict(raw_scores).fillna(0)
    df.index.name = 'Feature'
    df.reset_index(inplace=True)

    # Normalize each column
    for imp_type in importance_types:
        max_val = df[imp_type].max()
        df[f"{imp_type}_norm"] = df[imp_type] / max_val if max_val > 0 else 0

    # Default weights if none provided
    if weights is None:
        weights = {'weight': 0.15, 'gain': 0.6, 'cover': 0.25}

    # Compute combined score
    df['combined_score'] = (
        weights['weight'] * df['weight_norm'] +
        weights['gain']   * df['gain_norm'] +
        weights['cover']  * df['cover_norm']
    )

    df.sort_values(by='combined_score', ascending=False, inplace=True)

    if top_n:
        df = df.head(top_n)

    return df[['Feature', 'weight', 'gain', 'cover', 'combined_score']]

def plot_feature_importance_scores(df):
    #reverse order for top-down plotting
    df = df[::-1]
    # plotting
    fig, ax = plt.subplots(figsize=(8,4))
    ax.barh(df['Feature'],
            df['combined_score'],
            color='skyblue')

    # Labels and title
    ax.set_xlabel('Combined Importance Score')
    ax.set_xlim(0,1)
    ax.set_title('Top 5 Feature Importances')
    plt.tight_layout()

    return fig