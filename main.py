# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 15:03:08 2021

@author: Ozan
"""


import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix



data = pd.read_excel('--------/cancer patient data sets.xlsx',header=None)

columns_name = list(data.iloc[0])
data.columns = columns_name

print(data.isnull().sum())
print(data.isna().sum())


data_1 = data.drop(index=0,axis=1, inplace=False)
data_1.to_csv('canser_patient.csv', encoding='utf-8', index=False)

df = pd.read_csv('-------------/canser_patient.csv')
df1 = df.drop(labels=['Patient Id'],axis=1,inplace=False)
Patient_Id = df['Patient Id']
col = df1.columns



corr_Pearson = df.corr(method='pearson')

figure = plt.figure(figsize=(25,15))
sns.heatmap(corr_Pearson,vmin=-1,vmax=+1,cmap='Blues',annot=True, 
            linewidths=1,linecolor = 'white')
plt.title('Pearson Correlation')
plt.savefig('/cor.png')
plt.show()

for i in col:
    figure = plt.figure(figsize=(15,10))
    sns.countplot( x=i,hue='Gender',data=df1)
    plt.savefig('/'+str(i)+'.png')
    plt.show() 

data_x = df1.drop(labels=['Level'], axis=1, inplace=False)
data_y = df1['Level']

x_train, x_test, y_train, y_test = train_test_split(data_x,data_y,
                                    test_size = 0.3, random_state = 44, 
                                    shuffle = True)









Logistic_R = LogisticRegression(solver="liblinear").fit(x_train,y_train)

model = Logistic_R.predict(x_test)
cf_matrix = confusion_matrix(y_test,model)
figure = plt.figure(figsize=(20,10))
sns.heatmap(cf_matrix, linecolor = "white", linewidth = 1,
            cmap = "Blues", annot = True)
plt.savefig('/conf.png')
plt.show()





