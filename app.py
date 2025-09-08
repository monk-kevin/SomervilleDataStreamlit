# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 11:38:01 2025

Python code to create Streamlit app for Somerville Open Datasets. Created
with agentic coding alongside Microsoft Co-Pilot

@author: Kevin Monk
"""

import streamlit as st
import pandas as pd
import os
import plot_fcns
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import pickle

def app():
    """
    All code for the app goes here

    """
    if os.path.exists('data/df_happiness_prepped.pkl'):
        df = pd.read_pickle('data/df_happiness_prepped.pkl')
    else:
        st.error('Cannot find processed data -- check file location')
        
    if os.path.exists('data/best_params_full.pkl'):
        with open("data/best_params_full.pkl","rb") as f:
            best_params_full = pickle.load(f)
    else:
        st.error('Cannot find saved parameters -- check file location')

    st.title('How happy are Somerville Residents?')
    st.markdown(f"""
                #### Insights from the Binannual Somerville Happiness Survey
                Every two years since 2011, Somerville has asked a random
                selection of residents to complete a Happiness survey. In this 
                survey, respondents are asked to score their happiness with life
                on a 1 to 5 scale (1 being very unhappy and 5 being very happy).
                In addition, respondents are also asked a number of different questions
                related to satisfaction with life in Somerville along with
                a set of demographic questions. Here, I have created a dashboard
                to visualize historical trends in Happiness data 
                (i.e., visualize happiness trends across the years) and identify
                key features that are important for predicting a Happiness score.
                At left, there are a number of different demographic features
                used to filter data from Somerville residents allowing you to interact
                with the data and understand how happiness scores vary across the years
                and across demographics.  
                **__________________________________________________________________________________**
                """)
    
    # visualize data across column features
    def apply_filter(df, column_name, sidebar_label):
        """
        Adds a selectbox to the sidebar with an 'All' option and filters the DataFrame accordingly.

        Parameters:
            - df: pandas DataFrame
            - column_name: str, name of the column to filter
            - sidebar_label: str, label to display in the sidebar
        
        Returns:
            - filtered_df: pandas DataFrame after applying the filter
            - selected_value: the value selected in the selectbox
            """
        options = ["All"] + sorted(df[column_name].unique().tolist())
        selected_value = st.sidebar.selectbox(sidebar_label, options)
        
        if selected_value != "All":
            df = df[df[column_name] == selected_value]
        
        return df, selected_value
    demographic_columns = ['Age.mid','Ward','Rent.Mortgage.mid','Household.Income.mid',
                          'Gender','Race.Ethnicity','Housing.Status']
    filtered_df = df.copy()
    selected_filters = {}
    for col in demographic_columns:
        sidebar_label = f"Select {col} value"
        filtered_df, selected_value = apply_filter(filtered_df,col,sidebar_label)
        selected_filters[col] = selected_value
    
    #convert to hashable key
    filter_key = tuple(selected_filters.items())
        
    # filtered_df, age_group = apply_filter(df,'Age.mid','Median Age Group')
    # filtered_df, ward_num = apply_filter(filtered_df,'Ward','Ward Number')
    # filtered_df, year_asked = apply_filter(filtered_df,'Year','Survey Year')
    
    col_score = 'Happiness.5pt.num'
    mean_val = filtered_df[col_score].mean()
    std_val = filtered_df[col_score].std()
    count_val = filtered_df[col_score].count()
    tot_resp = df[col_score].count()
    prop_val = count_val / tot_resp * 100
    
    st.markdown(f"""
                With these filters, there are {count_val} respondents, or {prop_val:.2f}% of the full dataset.  
                The average happiness score is {mean_val:.2f} +/- {std_val:.2f}
                """)
    
    fig1 = plot_fcns.plot_mn_happiness(filtered_df, 'Year')
    st.pyplot(fig1)
    
    fig2 = plot_fcns.plot_happiness_dist(filtered_df,'Year')
    st.pyplot(fig2)
    
    is_all_filters = all(val == "All" for val in selected_filters.values())
    # performing XGBoost for Feature Importance only upon button press
    if st.button('Identify Important Features for these data'):
        
        
        # classify columns as numeric or categorical
        col_numeric = filtered_df.select_dtypes(include=['float64','int64']).columns.tolist()
        col_categorical = filtered_df.select_dtypes(exclude=['float64','int64']).columns.tolist()

        # find categorical columns that are not repeated within the numerical columns
        col_categorical_unique = []
        for s_cat in col_categorical:
            if (
                    not any(s_cat in s_num for s_num in col_numeric) 
                    and 'label' not in s_cat
                    ):
                col_categorical_unique.append(s_cat)
                
                OHE_col = col_categorical_unique + ['Ward','Year']
                MMS_col =  ['CrashCount','CrimeCount',
                            'CrashCount_PerYearPerWard','CrimeCount_PerYearPerWard',
                            'bike_count','ped_count','ped2bike_foldinc',
                            'Income.Per.Number.In.Household','Rent.Mortgage.mid',
                            'Rent.Mortgage.Per.Bedroom','ACS.Somerville.Median.Income']
    
        #define a new column transformer that removes the categorical columns
        CT_dropcat = ColumnTransformer(
            transformers = [
                ('dropcat','drop',OHE_col),
                ('MinMax',MinMaxScaler(),MMS_col)
                ],
            remainder = 'passthrough'
            )   

        # create pipeline to pre-process data in X for model fit
        pipeline_xgb = Pipeline([
            ('encode_scale',CT_dropcat),
            ('imputer', SimpleImputer(strategy = 'most_frequent')),
            ('bst',XGBClassifier(objective = 'multi:softmax',
                                 num_class = 5,
                                 eval_metric = 'mlogloss'))
            ])
        
        if is_all_filters:
            best_params = best_params_full
            scores = {}
            scores['accuracy'] = 0.702
            scores['f1'] = 0.699
        else:
            # Train / Test Split with Stratification
            X_train, X_test, y_train, y_test = train_test_split(filtered_df.drop(['Happiness.5pt.num'],axis=1),
                                                                filtered_df['Happiness.5pt.num'],
                                                                test_size = 0.2,
                                                                random_state = 42,
                                                                stratify = filtered_df['Happiness.5pt.num'])
            y_train = y_train - 1
            y_test = y_test - 1
        
            # setting up for cross-validation
            params = {
                    'bst__eta': [0.01, 0.05, 0.1, 0.2, 0.3],
                    'bst__min_child_weight': [1, 2, 4, 8, 10],
                    'bst__gamma': [0, 0.75, 1.5, 3, 6],
                    'bst__max_depth': [2, 4, 6, 8, 10],
                    }
            skf = StratifiedKFold(n_splits = 5)
            
            XGB_CV = RandomizedSearchCV(pipeline_xgb,
                                        param_distributions=params,
                                        n_iter = 10,
                                        scoring = "f1_weighted",
                                        cv = skf.split(X_train,y_train),
                                        random_state = 42)
        
            #fitting and evaluating
            XGB_CV.fit(X_train,y_train);
            best_params = XGB_CV.best_params_
            
            y_pred = XGB_CV.predict(X_test)
            scores = {}
            scores['accuracy'] = round(metrics.accuracy_score(y_test, y_pred), 3)
            scores['f1'] = round(metrics.f1_score(y_test,y_pred,average = 'weighted'),3)
            
        st.markdown(f"""
                    An XGBClassifier was evaluated with this filtered data. 
                    Here are those results:  
                        - Accuracy: {scores['accuracy']}  
                        - F1 Score: {scores['f1']}
                        """)
        
        # running optimized XGB with full data
        XGB_optimized = XGBClassifier(**best_params)
        X = filtered_df.drop(['Happiness.5pt.num'],axis=1);
        y = filtered_df['Happiness.5pt.num'] - 1; #making base 0 for fitting
        
        # create pipeline to pre-process data in X for model fit
        pipeline_preprocess = Pipeline([
            ('encode_scale',CT_dropcat),
            ('imputer', SimpleImputer(strategy = 'most_frequent'))
            ])

        # perform pre-processing for dataset X
        X_processed = pipeline_preprocess.fit_transform(X)
        
        # fit full dataset
        XGB_optimized.fit(X_processed,y)
        
        #defining feature names based on the columns in X
        x_col = list(X.columns)
        ftr_names = [col_name for col_name in x_col if col_name not in OHE_col]
        XGB_optimized.get_booster().feature_names = ftr_names
        
        fig3 = plot_fcns.plot_feature_importance_column(XGB_optimized,['weight','gain','cover'])
        st.pyplot(fig3)
        
            
if __name__ == '__main__':
    app()

#%% - To-Do List - 
# add an error catcher
# figure out alternate feature importance strategies for filtered datasets too small for XGB
# Cache/load common demographics (e.g., all fields selected as All)
# List top features and show plots of how they interact with Happiness score
# Create a second page for historical demographics visualization