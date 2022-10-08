import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as MSE

%matplotlib inline

#data that its entered is being read
data = pd.read_csv("Real_estate.csv")
data.head()

#the data is describing
data.describe()

#data from the information is being analyzed
data.info()

#vizualizing plot
sns.pairplot(data)


#Data scale
scaler = StandardScaler()

X = data.drop(['price','house_No'],axis = 1)
Y = data['price']

cols = X.columns

X = scaler.fit_transform(X)


#testing and training for data splitting 
X_train,X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.4, random_state = 101)


#predicting linear regression model

lr = LinearRegression()
lr.fit(X_train, Y_train)

predict = lr.predict(X_test)

MSE_score = MSE(Y_test,predict)

#prediction visualizing
sns.scatterplot(x=y_test, y = predict)


#the residuals of the model plotting
sns.histplot((y_test-predict),bins= 50, kde = True)
plt.legend(loc = "upper right")
plt.title("Residual errors)
plt.show()

#coefficients observing
 cdf = pd.DataFrame(lr.coef_,cols,['coefficients']).sort_values('coefficients',ascending = False)cdf

#print

print("Mean Squared Error: ",MSE_score.mean())
print("coefficients: ",lr.coef_)
