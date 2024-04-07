import knn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
iris = pd.read_csv("iris.csv")
print(iris.head())
print(iris.describe())
print("Vegetables", iris["species"].unique())
import plotly.express as px
fig = px.scatter(iris, x="sepal_width", y="sepal_length", color="species")
fig.show()
x_new = np.array([[5, 2.9, 1, 0.2]])
prediction = knn.predict(x_new)
print("Prediction: {}".format(prediction))
