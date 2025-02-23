import pandas as pd
import numpy as np
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn
from zipfile import ZipFile
import keras
from keras.models import Sequential
from keras.regularizers import l2
from keras.layers import Dense, Dropout, Activation, BatchNormalization


datas = '/content/CovidData.csv.zip'
with ZipFile(datas,'r') as zip:
  zip.extractall()
  print('The dataset is extracted')


dataset = r'/content/diabetes.csv'
data  = pd.read_csv(dataset)
data.head(3)

data.shape

data.isnull().sum()

x = data.drop(columns=['Outcome']).iloc[::-1].reset_index(drop=True)
y = data['Outcome'].iloc[::-1].reset_index(drop=True)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)

x_test.dtypes

ann = Sequential()


ann.add(Dense(6, input_dim=8, activation=None, kernel_regularizer=l2(0.01)))  # No activation here
ann.add(BatchNormalization())  # Apply batch normalization
ann.add(Activation('relu'))  # Apply ReLU activation after batch normalization
ann.add(Dropout(rate=0.2))  # Apply dropout after activation

# Second hidden layer
ann.add(Dense(8, activation=None, kernel_regularizer=l2(0.01)))
ann.add(BatchNormalization())
ann.add(Activation('relu'))  # ReLU after batch norm
ann.add(Dropout(rate=0.2))

# Third hidden layer
ann.add(Dense(10, activation=None, kernel_regularizer=l2(0.01)))
ann.add(BatchNormalization())
ann.add(Activation('relu'))
ann.add(Dropout(rate=0.2))

# Fourth hidden layer
ann.add(Dense(12, activation=None, kernel_regularizer=l2(0.01)))
ann.add(BatchNormalization())
ann.add(Activation('relu'))
ann.add(Dropout(rate=0.2))

# Fifth hidden layer
ann.add(Dense(14, activation=None, kernel_regularizer=l2(0.01)))
ann.add(BatchNormalization())
ann.add(Activation('relu'))
ann.add(Dropout(rate=0.2))

# Output layer
ann.add(Dense(1, activation='sigmoid'))  # Sigmoid activation for binary classification

# Compile the model
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with a smaller batch size
ann.fit(x_train, y_train, batch_size=32, epochs=10)

prd = ann.predict(x_test)

prd_data = []
for i in prd:
  if i > 0.5:
    prd_data.append(1)
  else:
    prd_data.append(0)

prd_data[0:5]
prd_data[:-5]


cm = confusion_matrix(y_test, prd_data)
print(cm)
accuracy_score(y_test, prd_data)

# [[100   0]
#  [ 52   2]]
# 0.6623376623376623

pri = np.array([1, 96, 122, 0, 0, 22.4, 0.207, 27]).reshape(1, -1)  # reshape to (1, 8)
out = ann.predict(pri)
out_obj = out.astype(object)

# Print the result
print(out_obj)

