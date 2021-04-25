# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 15:03:08 2021

@author: Ozan
"""

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns



data = pd.read_excel('C:/Users/Ozan/Desktop/cancer patient data sets.xlsx',header=None)

print(data.isnull().sum())
print(data.isna().sum())

columns_name = list(data.iloc[0])
data.columns = columns_name


df = data.drop(index=0,axis=1, inplace=False)
Patient_Id = df['Patient Id']
df1 = df.drop(labels=['Patient Id'],axis=1,inplace=False)
columns = df1.columns

#Data Visualization
for i in columns:
    figure = plt.figure(figsize=(15,10))
    sns.countplot( x=i,hue='Gender',data=df1)
    plt.show() 
    
    
    
figure = plt.figure(figsize=(15,10))
sns.boxplot( x='Age',y='Level',data=df1)
plt.show()






