from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.svm import SVR
from sklearn.metrics.pairwise import laplacian_kernel
from sklearn.metrics import mean_squared_error
import numpy as np


class RandomMachinesRegression(BaseEstimator, RegressorMixin):

    def __init__(self,
                 poly_scale=2,
                 coef0_poly=0,
                 gamma_rbf=1,
                 gamma_lap=1,
                 degree=2,
                 cost=1,
                 boots_size=25,
                 epsilon=0.1,
                 beta=2,
                 seed_bootstrap=None,
                 automatic_tuning=False):
        """
        Parameters:
            poly_scale: float, default=2
            coef0_poly: float, default=0
            gamma_rbf: float, default=1
            gamma_lap: float, default=1
            degree: float, default=2
            cost: float, default=10
            boots_size: float, default=25
            automatic_tuning: bool, default=False
            epsilon: float, default=0.1
            beta: float, default=2
        """
        self.poly_scale = poly_scale
        self.coef0_poly = coef0_poly
        self.gamma_rbf = gamma_rbf
        self.gamma_lap = gamma_lap
        self.degree = degree
        self.cost = cost
        self.epsilon = epsilon
        self.beta = beta
        self.boots_size = boots_size
        self.seed_bootstrap = seed_bootstrap
        self.automatic_tuning = automatic_tuning

    def fit(self, X, y):
        """Fit the Random Machines Regression model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) \
                or (n_samples, n_samples)
            Training vectors, where `n_samples` is the number of samples
            and `n_features` is the number of features.
            For kernel="precomputed", the expected shape of X is
            (n_samples, n_samples).

        y : array-like of shape (n_samples,)
            Target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape (n_samples,), default=None
            Per-sample weights. Rescale C per sample. Higher weights
            force the classifier to put more emphasis on these points.

        Returns
        -------
        self : object
            Fitted estimator.
        """

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.X_ = X
        self.y_ = y

        # Set seed for bootstrap and kernel selection
        if (self.seed_bootstrap is not None):
            np.random.seed(self.seed_bootstrap)

        # Kernel types
        kernel_type = ["linear", "poly", "rbf", "laplacian"]

        # Training single model and calculating error metric
        early_models = []
        for kernel in kernel_type:
            model = self.fit_kernel(X, y, kernel)
            predict = model.predict(X)
            rmse = mean_squared_error(y, predict, squared=False)
            early_models.append(
                {'kernel': kernel, "model": model, 'metric': rmse})
            print(f"Kernel: {kernel} - RMSE: {rmse}")

        # Calculating the probability of each kernel
        sd_rmse = np.std([item["metric"] for item in early_models], ddof=1)
        rmse = np.divide(np.array([item["metric"]
                         for item in early_models]), sd_rmse)
        inv_rmse = np.exp(np.dot(-rmse, self.beta))
        prob_weights = np.divide(inv_rmse, np.sum(inv_rmse))
        prob_weights[prob_weights < 0] = 0

        #  sampling a kernel function
        random_kernel = np.random.choice(
            kernel_type, self.boots_size, replace=True, p=prob_weights)

        # Creating the bootstrap sample
        boots_samples_index = []
        for i in range(self.boots_size):
            nrow = len(X)
            train_index = np.random.choice(
                range(nrow), size=nrow, replace=True)
            boots_samples_index.append(train_index)

        # Training the models
        models = []
        for index in range(len(random_kernel)):
            kernel = random_kernel[index]
            boot_sample_index = boots_samples_index[index]
            X_train = X[boot_sample_index]
            y_train = y[boot_sample_index]
            X_test = np.delete(X, boot_sample_index, axis=0)
            y_test = np.delete(y, boot_sample_index, axis=0)
            model = self.fit_kernel(
                X_train, y_train, kernel)
            predict_oobg = model.predict(X_test)
            loss_function = mean_squared_error(
                y_test, predict_oobg, squared=False)
            models.append({'model': model, 'kernel': kernel,
                           'loss_function': loss_function})

        all_loss_function = [item["loss_function"] for item in models]
        sd_loss_function = np.std(all_loss_function, ddof=1)
        kernel_weight = np.divide(
            np.array(all_loss_function), sd_loss_function)
        kernel_weight = np.exp(np.dot(-kernel_weight, self.beta))
        kernel_weight_norm = np.divide(kernel_weight, np.sum(kernel_weight))
        # Adding kernel_weight_norm to models
        for index in range(len(models)):
            models[index]["kernel_weight"] = kernel_weight_norm[index]

        self.models = models

        return self

    def predict(self, X):
        """Perform regression on samples in X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            For kernel="precomputed", the expected shape of X is
            (n_samples_test, n_samples_train).

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted values.
        """
        # Check if fit has been called
        check_is_fitted(self)
        # Input validation
        X = check_array(X)

        models = self.models
        final_predict = []
        for model in models:
            model_weights = model["kernel_weight"]
            predict = model["model"].predict(X)
            multiply = np.multiply(model_weights, predict)
            final_predict.append(multiply)
        return list(np.sum(final_predict, axis=0))

    def fit_kernel(self, X_train, y_train, kernel):
        if (self.automatic_tuning):
            if (kernel == "laplacian"):
                model = SVR(kernel=laplacian_kernel,
                            gamma=self.gamma_lap).fit(X_train, y_train)
            else:
                model = SVR(kernel=kernel).fit(X_train, y_train)
            return model
        else:
            if (kernel == "linear"):
                model = SVR(kernel="linear",
                            C=self.cost,
                            epsilon=self.epsilon,
                            verbose=0).fit(X_train, y_train)
            elif (kernel == "poly"):
                model = SVR(kernel="poly",
                            C=self.cost,
                            epsilon=self.epsilon,
                            gamma=self.poly_scale,
                            coef0=self.coef0_poly,
                            degree=self.degree,
                            verbose=0).fit(X_train, y_train)
            elif (kernel == "rbf"):
                model = SVR(kernel="rbf",
                            C=self.cost,
                            epsilon=self.epsilon,
                            gamma=self.gamma_rbf,
                            verbose=0).fit(X_train, y_train)
            elif (kernel == "laplacian"):
                model = SVR(kernel=laplacian_kernel,
                            C=self.cost,
                            epsilon=self.epsilon,
                            verbose=0).fit(X_train, y_train)
            return model
