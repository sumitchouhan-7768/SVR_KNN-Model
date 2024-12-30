import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
#importing dataset
df = pd.read_csv (r"C:\Users\HP\OneDrive\Desktop\NIT All-Projects\SVR & Knn Model\emp_sal.csv")
X = df.iloc[:,1:2].values
y = df.iloc[:,2].values

#SVR model
from sklearn.svm import SVR
svr_regressor = SVR(kernel='poly',degree=4,gamma='auto')
svr_regressor.fit(X,y)

svr_model_pred = svr_regressor.predict([[6.5]])
print(f'Svr Predction:{svr_model_pred}')

# Visualising the SVR results
plt.scatter(X, y, color = 'red')
plt.plot(X, svr_regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#Knn model
from sklearn.neighbors import KNeighborsRegressor
knn_regressor = KNeighborsRegressor(n_neighbors=4,weights = 'uniform',p = 2)
knn_regressor.fit(X,y)

knn_reg_pred = knn_regressor.predict([[6.5]])
print(f'Knn Predction:{knn_reg_pred}')

# Visualising the Knn results
plt.scatter(X, y, color = 'red')
plt.plot(X, knn_regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (KNN)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()