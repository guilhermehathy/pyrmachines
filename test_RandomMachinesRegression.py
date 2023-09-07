from pyrmachines import RandomMachinesRegression
from sklearn.metrics import mean_squared_error
import pandas as pd


# Cars
df = pd.read_csv("data/cars.csv")
y = df["dist"]
X = df.drop("dist", axis=1)
rm = RandomMachinesRegression(seed_bootstrap=123)
res = rm.fit(X, y).predict(X)
print(mean_squared_error(y, res, squared=False))
