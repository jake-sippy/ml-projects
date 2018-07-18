import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

CSV_PATH = "iris.csv"
CSV_COLUMN_NAMES = ["SepalLength",
                    "SepalWidth",
                    "PetalLength",
                    "PetalWidth",
                    "Class"]

CLASS_NAMES = ["Iris-setosa",
               "Iris-versicolor",
               "Iris-virginca"]

data_table = pd.read_table(CSV_PATH, sep=",", names= CSV_COLUMN_NAMES)
print(data_table)
clf = svm.SVC()

# Remove the classification column
y = data_table.pop("Class")
X = data_table

X_train, X_test, y_train, y_test = train_test_split(X, y)

clf.fit(X_train, y_train)

print(clf.predict([[5.1, 3.2, 1.6, 0.4]]))

