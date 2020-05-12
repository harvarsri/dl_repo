import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import time

Name = 'Churn_Modelling_{}'.format(int(time.time()))
tensorboard = TensorBoard(log_dir = 'logs\\{}'.format(Name))

df = pd.read_csv('dataset/Churn_Modelling.csv')
X = df.iloc[:,3:13].values
Y = df.iloc[:,13].values

from sklearn.preprocessing import LabelEncoder , OneHotEncoder
le1 = LabelEncoder()
X[:,1] = le1.fit_transform(X[:,1])
le2 = LabelEncoder()
X[:,2] = le2.fit_transform(X[:,2])

from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder',OneHotEncoder(),[1])],remainder = 'passthrough')
X = ct.fit_transform(X)
X = X[:,1:]

from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size = 0.2 , random_state = 0)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Dense


model = Sequential()

model.add(Dense(units = 6, activation = 'relu',kernel_initializer = 'uniform', input_dim = 11))
model.add(Dense(units = 6, activation = 'relu',kernel_initializer = 'uniform'))
model.add(Dense(units = 6, activation = 'relu',kernel_initializer = 'uniform'))
model.add(Dense(units = 1, activation = 'sigmoid',kernel_initializer = 'uniform'))

model.compile(optimizer = 'adam',loss = 'binary_crossentropy', metrics = [('accuracy')])

model.fit(X_train,Y_train,epochs = 200 , batch_size = 25,callbacks = [tensorboard])

y_pred = model.predict(X_test)

y_pred = y_pred > 0.5

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,y_pred)
accuracy  = (cm[0][0] + cm[1][1]) / 2000

print(accuracy)