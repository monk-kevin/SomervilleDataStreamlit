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
import demographic_fcns
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
    # loading in data and saved model parameters
    if os.path.exists('data/df_happiness_prepped.pkl'):
        df = pd.read_pickle('data/df_happiness_prepped.pkl')
    else:
        st.error('Cannot find processed data -- check file location')
        
    if os.path.exists('data/best_params_full.pkl'):
        with open("data/best_params_full.pkl","rb") as f:
            best_params_full = pickle.load(f)
    else:
        st.error('Cannot find saved parameters -- check file location')

    # welcome text
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
    
    #sidebar filtering demographics
    # set up binning for demographics
    age_bins = [0, 20, 30, 40, 50, 60, 70, float("inf")]
    rent_mortgage_bins = [0,1500,2000,2500,3000,3500,4000,float("inf")]
    income_bins = [0, 30000, 60000, 90000, 120000, 150000, 175000, float("inf")]
    
    # creating dictionary for future binning
    binning_config = {
        "Age": {
            "col_name": "Age.mid",
            "bins": age_bins,
            "labels": demographic_fcns.generate_labels(age_bins)
            },
        "RentOrMortgage": {
            "col_name": "Rent.Mortgage.mid",
            "bins": rent_mortgage_bins,
            "labels": demographic_fcns.generate_labels(rent_mortgage_bins)
            },
        "Income": {
            "col_name": "Household.Income.mid",
            "bins": income_bins,
            "labels": demographic_fcns.generate_labels(income_bins)}
        }
    
    #iterating over demographics to bin and keep new categorical columns for future CT
    binned_demo_columns = []
    for col, config in binning_config.items():
        new_col_name = f"{col}Group"
        df[new_col_name] = demographic_fcns.bin_demographics(df, 
                                                             config["col_name"],
                                                             config["bins"],
                                                             config["labels"])
        binned_demo_columns.append(new_col_name)

    demographic_columns = ['AgeGroup','Ward','RentOrMortgageGroup','IncomeGroup',
                          'Gender','Race.Ethnicity','Housing.Status']
    
    # adding a reset button for all filters
    if st.sidebar.button('Reset Filters'):
        for col in demographic_columns:
            st.session_state[f"{col}_filter"] = "All"
    
    filtered_df = df.copy()
    selected_filters = {}
    for col in demographic_columns:
        filter_key = f"{col}_filter"
        sidebar_label = f"Select {col} value"
        
        # Initialize session state if not already set
        if filter_key not in st.session_state:
            st.session_state[filter_key] = "All"
        
        #Apply filter using session state
        filtered_df, selected_value = demographic_fcns.apply_filter(filtered_df,col,sidebar_label)
        selected_filters[col] = selected_value
    
    
    # organizing data for visualization
    col_score = 'Happiness.5pt.num'
    mean_val = filtered_df[col_score].mean()
    std_val = filtered_df[col_score].std()
    count_val = filtered_df[col_score].count()
    tot_resp = df[col_score].count()
    prop_val = count_val / tot_resp * 100
    
    # display results
    st.markdown(f"""
                With these filters, there are {count_val} respondents, or {prop_val:.2f}% of the full dataset.  
                The average happiness score is {mean_val:.2f} +/- {std_val:.2f}
                """)
    
    # visualize mean happiness scores across years for data
    fig1 = plot_fcns.plot_mn_happiness(filtered_df, 'Year')
    st.pyplot(fig1)
    
    # visualize distributions of happiness filters
    fig2 = plot_fcns.plot_happiness_dist(filtered_df,'Year')
    st.pyplot(fig2)
    
    # identify if filters are set for saved hyperparameters
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
                
                OHE_col = col_categorical_unique + binned_demo_columns + ['Ward','Year']
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
                        - **Accuracy**: {scores['accuracy']}, 
                        **F1 Score**: {scores['f1']}
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
        
        
        
        
        
        # plotting features for different importance types
        # fig3 = plot_fcns.plot_feature_importance_column(XGB_optimized,['weight','gain','cover'])
        # st.pyplot(fig3)
        
        # identifying important features with weighted avg of importance
        n_ftrs = 5
        df_top_features = plot_fcns.get_combined_importance(XGB_optimized, top_n = n_ftrs)
        
        # Store results in session state
        st.session_state['df_top_features'] = df_top_features
        st.session_state['XGB_optimized'] = XGB_optimized
        st.session_state['best_params'] = best_params
        st.session_state['model_trained'] = True
    
        
    if st.session_state.get('model_trained', False):
        df_top_features = st.session_state['df_top_features']
        XGB_optimized = st.session_state['XGB_optimized']

        st.subheader(f"Top {len(df_top_features)} Features Driving Happiness Score:")
        st.markdown("""
                    Using built-in methods, I have identified the top features that are important
                    for predicting Happiness Scores for this filtered dataset. Namely, I have created
                    a Combined Score through a weighted average of 'weight', 'gain', and 'cover' scores.
                    """)
        fig3 = plot_fcns.plot_feature_importance_scores(df_top_features)
        st.pyplot(fig3)

        st.markdown("""
                    Let's see how each feature relates to a respondent's Happiness Score.
                    Select a feature below to plot Happiness Score as a function of its values.
                    """)

        feature_options = [f"{ii+1}. {row.Feature}" for ii, row in enumerate(df_top_features.itertuples(index=False))]
        selected_label = st.selectbox("Choose a Feature to Explore", feature_options)
        selected_feature = selected_label.split(". ", 1)[-1]

        fig4 = plot_fcns.plot_mn_happiness(filtered_df, selected_feature)
        st.pyplot(fig4)
        
            
if __name__ == '__main__':
    app()

#%% - To-Do List - 
# add an error catcher
# figure out alternate feature importance strategies for filtered datasets too small for XGB
# Group demographics - done
# Cache/load common demographics (e.g., all fields selected as All)
#   loading all All - done
#   caching user input
# List top features and show plots of how they interact with Happiness score
# Create a second page for historical demographics visualization?