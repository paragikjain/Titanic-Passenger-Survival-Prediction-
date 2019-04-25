#classification problem 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#data insertion in programme
data_train=pd.read_csv('train.csv')
data_test=pd.read_csv('test.csv')

#count the missing values in each colume
count=data_train.count()  #this give you no of element in each colume
total_rows=data_train.shape[0]
missing_rows=total_rows-count

#finding the median of Embarked Colume
data_train.groupby('Embarked').count()
#by this we got c=168  q=77 and s=644 so we replace NaN by s

#fiiling value by mean
data_train.Age=data_train.fillna(data_train.Age.mean()).Age  
#filling the value of Embarked by Median
data_train.Embarked=data_train.Embarked.replace(np.nan,'S')
count=data_train.count()  #this give you no of element in each colume
total_rows=data_train.shape[0]
missing_rows=total_rows-count

train_x=data_train.iloc[:,[2,4,5,6,7,9,11]].values
train_y=data_train.iloc[:,1].values

from sklearn.preprocessing import LabelEncoder , OneHotEncoder
train_x[:,1]=LabelEncoder().fit_transform(train_x[:,1])
train_x[:,6]=LabelEncoder().fit_transform(train_x[:,6])

#create model

from sklearn.neighbors import KNeighborsClassifier
cls=KNeighborsClassifier(n_neighbors=3, metric='minkowski',p=1).fit(train_x,train_y)#,p=1 means some other distancep=2 for euclidian distance
pred=cls.predict(train_x)

#making the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(pred,train_y)

#more advance Evloution K flod cross validation  #model for different dataset 
from sklearn.model_selection import cross_val_score as cvs
accuracies=cvs(estimator=cls,X=train_x,y=train_y,cv=10)
accuracies.mean() 
accuracies.std()

#grid serach for find best algorithm and best parameter
from sklearn.model_selection import GridSearchCV as gsc
parameters=[{'n_neighbors':[3,4,5,6,7,8,9,10],'metric':['minkowski'],'p':[1]},
            {'n_neighbors':[3,4,5,6,7,8,9,10],'metric':['minkowski'],'p':[2]}]
grid_search=gsc(estimator=cls,param_grid=parameters,scoring='accuracy',cv=10) #cv is for cross validation
grid_search=grid_search.fit(train_x,train_y)
best_accuracy=grid_search.best_score_
best_parameter=grid_search.best_params_





 
#Analysing data 

