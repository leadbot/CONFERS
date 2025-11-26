# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 11:42:59 2025

@author: dl923 / leadbot
"""

import logging
import time
import sys
import joblib
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, label_binarize
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, auc
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.multiclass import OneVsRestClassifier
#%% Import data 
########### DEEP LEARNING ###########
###import and format data
df=pd.read_csv("Training_data/CONFERS_LPMO_training_n200_per_famil_alpha_out_singles_output.csv_2703_16_15_08.csv", low_memory=False, sep=',', encoding='utf-8')

df['classification']=df['Structure'].str.split('\\').str[1].str.split('_').str[0]
df['ID']=df['Structure'].str.split('\\').str[1].str.rsplit('.',n=2).str[0].str.rsplit('_',n=2).str[0]
single_class=True
if single_class:
    #df['classification']=df['Structure'].str.split('\\').str[1].str.split('_').str[0].str[0:2]
     df['classification']='LPMO'

###read in OTHER data     
df_CAZydb=pd.read_csv("Training_data/CONFERS_CAZydb_training_alpha_out_singles_output.csv_2703_17_26_33.csv", low_memory=False, sep=',', encoding='utf-8')
df_CAZydb['ID']=df_CAZydb['Structure'].str.split('\\').str[1].str.rsplit('.',n=2).str[0].str.rsplit('_',n=2).str[0]
df_CAZydb['classification']='OTHER'

#%% Misclassificaiton audit - Import data from ERROR ids in previous version
audit=True
if audit:
    error_df=pd.read_csv("Audit_error_ids/CONFERS_Error_ids_files_output.csv_2803_09_06_44.csv", low_memory=False, sep=',', encoding='utf-8')
    error_df['ID']=error_df['Structure'].str.split('\\').str[1].str.rsplit('.',n=2).str[0].str.rsplit('_',n=2).str[0]
    error_df['classification']='OTHER'
#%% Check for contamination in sequences
#Read in annotation dataset
filter_data=True
if filter_data:
    annot_df=pd.read_csv("Training_metadata/CAZyDB_dbCAN3_output.txt", index_col='Gene ID', sep='\t', encoding='utf-8')
    columns_to_check = ['HMMER', 'DIAMOND', 'dbCAN_sub']
    strings_to_search = ['AA10', 'AA11', 'AA13', 'AA14','AA15','AA16','AA17','AA9']
    # Create a boolean mask based on the conditions
    mask = annot_df[columns_to_check].apply(lambda col: col.str.contains('|'.join(strings_to_search), na=False)).any(axis=1)

    # Apply the mask to filter the DataFrame
    filtered_df = annot_df[mask]
    ##check for AA domains in non AA families (CAZydb)
    df_CAZydb_filt=df_CAZydb[(~df_CAZydb['ID'].isin(filtered_df.index))]
    print("CAZy proteins containing LPMO domains have been removed. n = ", str(len(df_CAZydb[df_CAZydb['ID'].isin(filtered_df.index)])))
    ##check for absent histidine brace containing proteins within LPMO dataset
    df_filt=df[df['NE2_NE2_dist']>0]
    print("Putative LPMOs lacking his-braces have been removed. n =", str(len(df[df['NE2_NE2_dist']==0])))

#%% Classify by individual families
family_level_classification=True
if family_level_classification:
    CAZy_classes=pd.read_csv("Training_metadata/CAZydb_ALL_LPMOs_n200.csv", sep=',', encoding='utf-8')
    mapping_dict = dict(zip(CAZy_classes['Accession'], CAZy_classes['Family']))
    if filter_data:
         df_filt['classification']=df_filt['ID'].map(mapping_dict)
    else: 
         df['classification']=df['ID'].map(mapping_dict)        
#%% Combine data to form merged classification datasets
concat=True
if concat:
    if filter_data:
         ## NB reset index otherwise recalling erroneous classifications will be difficult
         df=pd.concat([df_filt, df_CAZydb_filt]).fillna(0).reset_index()
         if audit:
             df=pd.concat([df_filt, df_CAZydb_filt, error_df]).fillna(0).reset_index()
    else: 
         if audit:
             df=pd.concat([df, df_CAZydb, error_df]).fillna(0).reset_index()
         else:    
              df=pd.concat([df, df_CAZydb]).reset_index()

#%% Factorise secondary structure
factorize=True
if factorize:#
     fact_columns=[col for col in df.columns if col.startswith("SS_")]
     unique = np.unique(df[fact_columns].astype(str))
     unique_dict_map={k:v for v, k in dict(enumerate(unique)).items()}
     df[fact_columns]=df[fact_columns].applymap(lambda x: unique_dict_map.get(x, x))

###Manually drop problematic proteins 
#drop_proteins=['AAF49624.2']
#df=df[~(df.ID.isin(drop_proteins))]
#%% Deep learning data preparation
# Assuming 'classification' is the target variable
drop_columns=False
df2=df
#columns_to_drop=['NE2overCA']
columns_to_drop=['Min_num_beta_strands', 'NE2overCA']
if drop_columns:
    df2=df.drop(columns=columns_to_drop)

metadata_columns=['index', 'Index', 'Structure', 'Brace_ID', 'classification', 'ID']
X = df2.drop(columns=metadata_columns).astype('float32')  # Features
y = df2['classification']  # Target variable
def make_weight_lists(multipliers):
    weightlist=[]
    sample_weights = np.ones(X_train.shape[1])
    sum_other_variables = len(X_train.iloc[:,8:]) #number of seconardy structure columns
    for x in multipliers:
          sample_weights[:8] = sum_other_variables*x/len(sample_weights[:8]) # Multiply weights of the first 8 variables by x times the mean of the sum of other variables
          weightlist.append(list(sample_weights))
          print(sum_other_variables*x/len(sample_weights[:8]))
    return weightlist

def weighting(dataframe):
     #dataframe[0:4]=dataframe[0:4]*5
     df=dataframe.copy(deep=True)
     #df['NE2overCA']=df['NE2overCA']*2
     df[0:5]=df[0:5]*4
     df[8:]=df[8:]/2
     return df
 
weighting_on_off=False    
if weighting_on_off==True:
    X=weighting(X)
    weights=make_weight_lists([0.01,0.02,0.03])

#NN_datalength_hardcoded_stop=1543
NN_datalength_hardcoded_stop=500
X=X.iloc[:,:NN_datalength_hardcoded_stop]

# Encode labels to numerical values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, y_encoded, df2.index, test_size=0.3, random_state=42)
