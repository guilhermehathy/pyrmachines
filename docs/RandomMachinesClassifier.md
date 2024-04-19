# Class: RandomMachinesClassifier

The `RandomMachinesClassifier` is an ensemble learning method based on a bagged-weighted support vector model that allows free kernel choice depending on the performance of each kernel during training. It is suitable for classification tasks and supports various kernel methods including linear, polynomial, radial basis function (RBF), and laplacian.

## Parameters:
- **poly_scale** (float, default=2): Scaling factor for the polynomial ('poly') kernel.
- **coef0_poly** (float, default=0): Independent term in the polynomial kernel.
- **gamma_rbf** (float, default=1): Kernel coefficient for the RBF kernel.
- **gamma_lap** (float, default=1): Kernel coefficient for the laplacian kernel.
- **degree** (int, default=2): Degree of the polynomial kernel.
- **cost** (float, default=1): Regularization parameter; the strength of the regularization is inversely proportional to cost.
- **metric** (str, default='accuracy'): Performance metric used to evaluate the model.
- **boots_size** (int, default=25): Number of bootstrap samples used for training.
- **seed_bootstrap** (int, optional): Random seed for bootstrap sampling and kernel selection.
- **automatic_tuning** (bool, default=False): If True, the kernel coefficients (gamma_rbf and gamma_lap) are tuned automatically based on the data.

## Methods:

### fit(X, y)
Fit the classifier using the provided training data and labels.

**Parameters:**
- **X** (array-like or sparse matrix, shape (n_samples, n_features)): Training data.
- **y** (array-like, shape (n_samples,)): Target values (class labels).

**Returns:**
- **self** (object): Fitted estimator.

### predict(X)
Perform classification on an array of test vectors X.

**Parameters:**
- **X** (array-like or sparse matrix, shape (n_samples, n_features)): Test data.

**Returns:**
- **y_pred** (array, shape (n_samples,)): Predicted class labels for the samples in X.

### sigest(X)
Estimate the gamma parameter for the RBF and laplacian kernels using a heuristic based on data spread.

**Parameters:**
- **X** (array-like, shape (n_samples, n_features)): Data used for estimation.

**Returns:**
- **srange** (float): Estimated gamma value.

### fit_kernel(X_train, y_train, kernel)
Fit an individual support vector classifier using a specified kernel.

**Parameters:**
- **X_train** (array-like, shape (n_samples, n_features)): Training data.
- **y_train** (array-like, shape (n_samples,)): Labels for training data.
- **kernel** (str): Specifies the kernel type to be used in the model.

**Returns:**
- **model** (SVC object): Trained model.

## Example of Usage

```
from sklearn.datasets import load_iris
from your_package_name import RandomMachinesClassifier
from sklearn.model_selection import train_test_split

# Load data
iris = load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=11)

# Initialize and train RandomMachinesClassifier
classifier = RandomMachinesClassifier(seed_bootstrap=123)
classifier.fit(X_train, y_train)

# Predict
predictions = classifier.predict(X_test)
print(predictions)```

