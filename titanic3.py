# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 11:44:39 2017

@author: Babishula
"""

#Titanicby myself

#exporting the packages 
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

#Exporting data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
combine = [train_df, test_df]