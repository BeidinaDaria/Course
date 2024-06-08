import pandas as pd
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn import preprocessing
df1 = pd.read_csv('Train_Network Intrusion Detection.csv')
print(df1.info())
df2 = pd.read_csv('ddos_dataset.csv',float_precision='high')
print(df2.info())
print(df2['Label'].describe())
col_rank=df1.columns
ordinal_encoder = preprocessing.OrdinalEncoder(dtype=int)
df1 = pd.DataFrame(ordinal_encoder.fit_transform(df1), columns=col_rank )
columns_to_drop = ['Flow ID','Timestamp', 'Source IP', 'Destination IP']
df2.drop(columns=columns_to_drop, inplace=True)
df2=df2.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
df2.dropna(how = 'all')
df2['Label'].replace(['BENIGN','DDoS'],[0,1],inplace=True)
print(df2.info())
X1_train, X1_test, Y1_train, Y1_test = train_test_split(df1.drop('class',axis=1), df1['class'], test_size = 0.1, random_state=5)
X2_train, X2_test, Y2_train, Y2_test = train_test_split(df2.drop('Label',axis=1), df2['Label'], test_size = 0.2, random_state=42)

# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.svm import SVC
# from sklearn.ensemble import VotingClassifier
# from sklearn.ensemble import RandomForestClassifier
# KNeighbors = KNeighborsClassifier(n_neighbors=9000)
# KNeighbors.fit(X1_train,Y1_train)
# pred_1=KNeighbors.predict(X1_test)
# dtf = DecisionTreeClassifier(random_state=0)
# dtf.fit(X1_train,Y1_train)
# pred_2=dtf.predict(X1_test)
# svc = SVC(kernel='rbf', probability=False)
# svc.fit(X1_train,Y1_train)
# pred_3=svc.predict(X1_test)
# rfc = RandomForestClassifier(random_state=0)
# rfc.fit(X1_train,Y1_train)
# pred_4=rfc.predict(X1_test)
# KNeighbors = KNeighborsClassifier(n_neighbors=9000)
# KNeighbors.fit(X2_train,Y2_train)
# pred_5=KNeighbors.predict(X2_test)
# dtf = DecisionTreeClassifier(random_state=0)
# dtf.fit(X2_train,Y2_train)
# pred_6=dtf.predict(X2_test)
# svc = SVC(kernel='rbf',probability=False)
# svc.fit(X2_train,Y2_train)
# pred_7=svc.predict(X2_test)
# rfc = RandomForestClassifier(random_state=0)
# rfc.fit(X2_train,Y2_train)
# pred_8=rfc.predict(X2_test)

from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from keras.utils import to_categorical
model1=keras.Sequential([
                      layers.Dense(units=64,activation='relu', input_shape=(X1_train.shape[1],)),
                      layers.Dropout(rate=0.2),
                      layers.Dense(units=1, activation='sigmoid')
                        ])
model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history1=model1.fit(X1_train,Y1_train,batch_size=32,epochs=10,validation_split=0.2,verbose=2)
prediction1=model1.predict(X1_test)
print(prediction1)
model2=keras.Sequential([
   layers.LSTM(units=64, return_sequences=True),
   layers.Dropout(rate=0.2),
   layers.LSTM(units=64, return_sequences=False),
   layers.Dropout(rate=0.2),
   layers.Dense(units=1, activation='sigmoid'),
                        ])
model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
X = X1_train.to_numpy().reshape((X1_train.shape[0], X1_train.shape[1], 1))
history2=model2.fit(X,Y1_train,batch_size=32,epochs=10,validation_split=0.2,verbose=2)
prediction2=model2.predict(X1_test)
model1=keras.Sequential([layers.Dense(units=64,activation='relu', input_shape=(X2_train.shape[1],)),
                      layers.Dropout(rate=0.2),
                      layers.Dense(units=1, activation='sigmoid')
                        ]) 
model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history3=model1.fit(X2_train,Y2_train,batch_size=32,epochs=10,validation_split=0.2,verbose=2)
prediction3=model1.predict(X2_test)
X = X2_train.to_numpy().reshape((X2_train.shape[0], X2_train.shape[1], 1))
# history4=model2.fit(X,Y2_train,batch_size=32,epochs=10,validation_split=0.2,verbose=2)
# prediction4=model2.predict(X2_test)

plt.plot(history1.history['accuracy'])
plt.plot(history1.history['val_accuracy'])
plt.title('Dense net dataset 1 accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.plot(history1.history['loss'])
plt.plot(history1.history['val_loss'])
plt.title('Dense net dataset 1 loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
i=0
pred=[0]*Y1_test.shape[0]
for c in prediction1:
  if (c>0.5):
    pred[i]=1
  else:
    pred[i]=0
  i+=1
df1=pd.crosstab(Y1_test,pred)
print(df1)

plt.plot(history2.history['accuracy'])
plt.plot(history2.history['val_accuracy'])
plt.title('LSTM net dataset 1 accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.title('LSTM net dataset 1 loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
i=0
pred=[0]*Y1_test.shape[0]
for c in prediction2:
  if (c>0.5):
    pred[i]=1
  else:
    pred[i]=0
  i+=1
df1=pd.crosstab(Y1_test,pred)
print(df1)

plt.plot(history3.history['accuracy'])
plt.plot(history3.history['val_accuracy'])
plt.title('Dense net dataset 2 accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.plot(history3.history['loss'])
plt.plot(history3.history['val_loss'])
plt.title('Dense net dataset 2 loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
i=0
pred=[0]*Y2_test.shape[0]
for c in prediction3:
  if (c>0.5):
    pred[i]=1
  else:
    pred[i]=0
  i+=1
df1=pd.crosstab(Y2_test,pred)
print(df1)

# plt.plot(history4.history['accuracy'])
# plt.plot(history4.history['val_accuracy'])
# plt.title('LSTM net dataset 2 accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()
# plt.plot(history4.history['loss'])
# plt.plot(history4.history['val_loss'])
# plt.title('LSTM-net dataset 2 loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()
# i=0
# pred=[0]*Y2_test.shape[0]
# for c in prediction4:
#   if (c>0.5):
#     pred[i]=1
#   else:
#     pred[i]=0
#   i+=1
# df1=pd.crosstab(Y2_test,pred)
# print(df1)