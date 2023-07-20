from sklearn import datasets,linear_model
import numpy as np
import matplotlib.pyplot as plt
import sklearn

wine = datasets.load_wine()
wine_x = wine.data
# wine_x = wine.data[:,np.newaxis,2]

wine_x_train = wine_x[:-30]
wine_x_test = wine_x[-30:]

wine_y_train  = wine.target[:-30]
wine_y_test = wine.target[-30:]

model = linear_model.LinearRegression()
model.fit(wine_x_train, wine_y_train)
wine_y_predicted = model.predict(wine_x_test)

mse = sklearn.metrics.mean_squared_error(wine_y_test,wine_y_predicted)

print("Mean squared error is:  ",mse)
print("Weights:  ",  model.coef_)
print("Intercept:  ",  model.intercept_)

# plt.scatter(wine_x_test, wine_y_test)
# plt.plot(wine_x_test, wine_y_predicted)
# plt.show()