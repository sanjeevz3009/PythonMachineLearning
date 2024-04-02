import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
# print(data.head())

predict = "G3"

x = np.array(data.drop([predict], axis=1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# best = 0
# for _ in range(30):
#     x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

#     linear = linear_model.LinearRegression()
#     linear.fit(x_train, y_train)
#     accuracy = linear.score(x_test, y_test)
#     print(accuracy)

#     if accuracy > best:
#         best = accuracy
#         with open("studentmodel.pickle", "wb") as f:
#             pickle.dump(linear, f)

pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

# print("Co: ", linear.coef_)
# print("Intercept : ", linear.intercept_)

predictions = linear.predict(x_test)

for i in range(len(predictions)):
    print(predictions[i], x_test[i], y_test[i])

style.use("ggplot")

p = "absences"
plt.scatter(data[p], data["G3"])
plt.xlabel(p)
plt.ylabel("Final grade")
plt.show()
