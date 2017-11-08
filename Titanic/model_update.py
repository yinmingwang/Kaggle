import pandas as pd
import seaborn as sns
import numpy as np
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model, discriminant_analysis, cross_validation
from sklearn.preprocessing import MinMaxScaler
train = pd.read_csv('./Data/train.csv')
test = pd.read_csv('./Data/test.csv')
print(train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print(train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print(train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print(train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False))
grid = sns.FacetGrid(train, 'Survived')
grid.map(plt.hist, 'Age', bins=20)
sns.set_style('darkgrid')#设置背景
plt.show()