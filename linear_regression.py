import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style 
style.use('ggplot')
import pandas as pd
import math , datetime

from sklearn import preprocessing
from sklearn.linear_model import LinearRegression

#Importing the dataset
dataset = pd.read_csv('covid prediction.csv')
dataset = dataset[['Day','COVID-19','new_cases']]

forecast_col = 'new_cases'

forecast_out = int(math.ceil(0.05*len(dataset)))


dataset['label'] = dataset[forecast_col].shift(-forecast_out)


X = np.array(dataset['COVID-19'])
X = preprocessing.scale(X)
X = X[:-forecast_out]
X_lately = X[-forecast_out:]

dataset.dropna(inplace= True)

y = np.array(dataset['label'])
y = np.array(dataset['label'])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

clf = LinearRegression(n_jobs=-1)

clf.fit(X_train.reshape(-1,1), y_train.reshape(-1,1))

accuracy = clf.score(X_test.reshape(-1,1), y_test.reshape(-1,1))
forecast_set = clf.predict(X_lately.reshape(-1,1))
print(forecast_set, accuracy, forecast_out)



y_pred = clf.predict(X_test.reshape(-1,1))

df = pd.DataFrame({'Real Values':y_test.reshape(-1), 'Predicted Values':y_pred.reshape(-1)})
df


#performance measure
from sklearn.metrics import explained_variance_score
explained_variance_score(y_test, y_pred)
