from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from pyrmachines import RandomMachines
from sklearn.metrics import accuracy_score
import pandas as pd

# Pegando os dados para teste

df = pd.read_csv("data/iris.csv")
y = df["variety"]
X = df.drop("variety", axis=1)


predict_base = X
rm = RandomMachines()
res = rm.fit(X, y).predict(predict_base)

print(accuracy_score(y, res))

fruits = pd.read_csv(
    "https://raw.githubusercontent.com/susanli2016/Machine-Learning-with-Python/master/fruit_data_with_colors.txt", sep="\t")

feature_names = ['mass', 'width', 'height', 'color_score']
X = fruits[feature_names]
y = fruits['fruit_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

rm = RandomMachines(boots_size=1)
res = rm.fit(X_train, y_train).predict(X_test)

print(accuracy_score(y_test, res))
