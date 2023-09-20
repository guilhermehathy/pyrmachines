from sklearn.model_selection import train_test_split
from pyrmachines import RandomMachinesClassifier
from sklearn.metrics import check_scoring
import pandas as pd


# Teste com o Iris
df = pd.read_csv("data/iris.csv")
y = df["variety"]
X = df.drop("variety", axis=1)
metric = 'accuracy'
rm = RandomMachinesClassifier(seed_bootstrap=123, metric=metric)
rm.fit(X, y)#.predict(X)
metric_score = check_scoring(rm, metric)(rm, X, y)
print(f"Score na base Iris: {metric_score}")


# Teste com a base de frutas

fruits = pd.read_csv(
    "data/fruit.csv", sep="\t")

feature_names = ['mass', 'width', 'height', 'color_score']
X = fruits[feature_names]
y = fruits['fruit_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=11)
rm = RandomMachinesClassifier(seed_bootstrap=123)
res = rm.fit(X_train, y_train).predict(X_test)
acc = accuracy_score(res, y_test)
print(f"Accuracy na base fruit: {acc}")
