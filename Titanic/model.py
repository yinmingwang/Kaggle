import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model, discriminant_analysis, cross_validation
from sklearn.preprocessing import MinMaxScaler
train = pd.read_csv('./Data/train.csv')
test = pd.read_csv('./Data/test.csv')
#print(train.info())
#print(test.info())
selectFeatures = ['Pclass','Sex','Age','Embarked','SibSp','Parch','Fare']
X_train = train[selectFeatures]
X_test = test[selectFeatures]
y_train = train['Survived']
X_train['Embarked'].fillna('S',inplace=True)
X_test['Embarked'].fillna('S',inplace=True)
X_train['Age'].fillna(X_train['Age'].mean(),inplace=True)
X_test['Age'].fillna(X_test['Age'].mean(),inplace=True)
X_test['Fare'].fillna(X_test['Fare'].mean(),inplace = True)
def normalization(x, name):
    xmin = min(x[name])
    xmax = max(x[name])
    xnum = len(x)
    for i in range(xnum):
        x[name][i] = (float(x[name][i]) - xmin) / float(xmax - xmin)
#normalization(X_test,'Age')
#normalization(X_test,'Fare')
#normalization(X_train,'Age')
#normalization(X_train,'Fare')
#X_train.info()
#X_test.info()
dict_vec = DictVectorizer(sparse=False)
X_train = dict_vec.fit_transform(X_train.to_dict(orient='record'))
X_test = dict_vec.fit_transform(X_test.to_dict(orient='record'))
#train_x = DataFrame(X_train,columns=list(dict_vec.feature_names_))
#train_x.to_csv('train_x.csv')
regr = linear_model.LogisticRegression()
regr.fit(X_train,y_train)
y_predict = regr.predict(X_test)
#rfc = RandomForestClassifier()
#rfc.fit(X_train,y_train)
#y_predict = rfc.predict(X_test)
regr_submission = DataFrame({'PassengerId':test['PassengerId'].as_matrix(),'Survived':y_predict.astype(np.int32)})
regr_submission.to_csv('Logisticpredict_norm.csv',index=False)