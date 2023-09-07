from pyrmachines import RandomMachinesRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate, GridSearchCV
import pandas as pd


# Cars
# df = pd.read_csv("data/cars.csv")
# y = df["dist"]
# X = df.drop("dist", axis=1)
# rm = RandomMachinesRegression(seed_bootstrap=123)
# res = rm.fit(X, y).predict(X)
# print(mean_squared_error(y, res, squared=False))

# housing


# colunas = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
#            'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
# # colunas do dataframe (apenas necessario uma vez que os dados nao estao organizados da maneira usual no csv)

data = pd.read_csv('data/boston.csv')  # importando os dados
X = data.drop(["MEDV"], axis=1)
y = data["MEDV"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

# rm = RandomMachinesRegression(seed_bootstrap=123)
# y_pred = rm.fit(X_train, y_train).predict(X_test)
# RMSE = mean_squared_error(y_test, y_pred, squared=False)
# print(RMSE)


grd = GridSearchCV(estimator=RandomMachinesRegression(),
                   param_grid={'cost': [0.1, 1, 100, 1000],
                               'epsilon': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10], },
                   cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

grid_result = grd.fit(X_train, y_train)
best_params = grid_result.best_params_
