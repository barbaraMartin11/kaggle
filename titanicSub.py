# -*- coding: utf-8 -*-
"""
Created on Mon May  8 10:32:29 2017

@author: Babishula
"""
import numpy as np 
import pandas as pd

train_df = pd.read_csv('D:/Documents/ZZ2/STAGE_BIGML/kaggle/train.csv')
test_df = pd.read_csv('D:/Documents/ZZ2/STAGE_BIGML/kaggle/test.csv')
sub_df = pd.read_csv('D:/Documents/ZZ2/STAGE_BIGML/kaggle/gender_submission.csv')
bach_df = pd.read_csv('D:/Documents/ZZ2/STAGE_BIGML/kaggle/bach_prediction.csv')

def result(a):
    res=0
    if (a=="S"):
        res=1
    return res
type(sub_df)


print(sub_df[1,1])
print(train_df.shape)
print(test_df.shape)
print(sub_df[0:10])
print(bach_df[0:10])

print(bach_df[0:10])
data=[]
data=sub_df[:418]['PassengerId']
print(bach_df[1:25])



#that is important
sur=bach_df[['PassengerId', 'Embarked.1']]
#in order to see the shape of sur
print(sur[0:10])
#Change "Embarked.1" to  "Survived"
sur=sur.rename(columns={'Embarked.1': 'Survived'})
print(sur[0:10])
#Save in a csv file and do not save the index column
(sur).to_csv('D:/Documents/ZZ2/STAGE_BIGML/kaggle/sub.csv',index=False)
#(sur.rename(columns={'Embarked.1': 'Survived'})).to_csv('D:/Documents/ZZ2/STAGE_BIGML/kaggle/sub.csv')

