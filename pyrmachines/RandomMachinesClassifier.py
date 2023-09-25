from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.svm import SVC
from sklearn.metrics import get_scorer
from sklearn.metrics.pairwise import laplacian_kernel
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class RandomMachinesClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self,
                 poly_scale=2,
                 coef0_poly=0,
                 gamma_rbf=1,
                 gamma_lap=1,
                 degree=2,
                 cost=10,
                 metric='accuracy',
                 boots_size=25, seed_bootstrap=None, automatic_tuning=False):
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
        """
        self.poly_scale = poly_scale
        self.coef0_poly = coef0_poly
        self.gamma_rbf = gamma_rbf
        self.gamma_lap = gamma_lap
        self.degree = degree
        self.cost = cost
        self.metric = metric
        self.boots_size = boots_size
        self.seed_bootstrap = seed_bootstrap
        self.automatic_tuning = automatic_tuning

    def fit(self, X, y):
        """Fit the Random Machines Classifier model according to the given training data.

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
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y

        # Set seed for bootstrap and kernel selection
        if (self.seed_bootstrap is not None):
            np.random.seed(self.seed_bootstrap)

        # Kernel types
        kernel_type = ["linear", "poly", "rbf", "laplacian"]

        # Training single model and calculating accuracy
        early_models = []

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=self.seed_bootstrap, test_size=0.2)

        for kernel in kernel_type:
            model = self.fit_kernel(X_train, y_train, kernel)
            # predict = model.predict(X_test)
            # accuracy = accuracy_score(y_test, predict)
            metric_score = get_scorer(self.metric)(model, X_test, y_test)
            if (metric_score == 1):
                metric_score_log = 6.906755
            elif (metric_score <= 0.5):
                metric_score_log = 0
            else:
                # metric_score_log = np.log(np.divide(metric_score, np.subtract(1, metric_score)))
                metric_score_log = np.log(metric_score / (1 - metric_score))
            if (np.isinf(metric_score_log)):
                metric_score_log = 1
            early_models.append(
                {'kernel': kernel, "model": model, 'metric_score': metric_score, 'metric_score_log': metric_score_log})
            # print(f"Kernel: {kernel} - Score: {metric_score} - Log: {metric_score_log}")

        # Calculating the probability of each kernel
        prob_weights_sum = sum(item["metric_score_log"]
                               for item in early_models)
        lambda_values = {}
        for model in early_models:
            prob_weights = max(model["metric_score_log"] / prob_weights_sum, 0)
            # if (prob_weights < 0):
            #     prob_weights = 0
            model["prob_weights"] = prob_weights
            lambda_values[model["kernel"]] = prob_weights
            print(
                f"Kernel: {model['kernel']} - Score: {model['metric_score']} - Prob_weights: {prob_weights}")

        #  sampling a kernel function with probability = lambda_values
        p = [item["prob_weights"] for item in early_models]
        random_kernel = np.random.choice(
            kernel_type, self.boots_size, replace=True, p=p)

        # Creating the bootstrap sample
        boots_samples_index = []
        for i in range(self.boots_size):
            nrow = len(X)
            # nclass = len(self.classes_)
            train_index = np.random.choice(
                range(nrow), size=nrow, replace=True)
            table = np.unique(y[train_index], return_counts=True)
            ntable = len(table[0])
            while (ntable < 2):
                train_index = np.random.choice(
                    range(nrow), size=nrow, replace=True)
                table = np.unique(y[train_index], return_counts=True)
                ntable = len(table[0])
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
            model = self.fit_kernel(X_train, y_train, kernel)
            # out of bag
            # predict_oobg = model.predict(X_test)
            # accuracy = accuracy_score(y_test, predict_oobg)
            metric_score = get_scorer(self.metric)(model, X_test, y_test)
            if (metric_score == 1):
                kernel_weight = 1e+16
            else:
                kernel_weight = 1 / ((1 - metric_score) ** 2)
            print(
                f"Kernel: {kernel} - Score: {metric_score} - Weight: {kernel_weight}")
            models.append({'model': model, 'kernel': kernel,
                           'accuracy': metric_score, 'kernel_weight': kernel_weight, index: boot_sample_index})
        self.models = models

        return self

    def predict(self, X):
        """Perform classification on samples in X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) or \
                (n_samples_test, n_samples_train)
            For kernel="precomputed", the expected shape of X is
            (n_samples_test, n_samples_train).

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Class labels for samples in X.
        """
        # Check if fit has been called
        check_is_fitted(self)
        # Input validation
        X = check_array(X)

        nrow = X.shape[0]
        ncol = len(self.classes_)
        models = self.models
        predict_df = pd.DataFrame(
            np.zeros((nrow, ncol)), columns=self.classes_)
        for model in models:
            model_weights = model["kernel_weight"]
            predict = model["model"].predict(X)
            for i in range(len(predict)):
                predict_df.loc[i, predict[i]] += model_weights
        return list(predict_df.idxmax(axis=1))

    def fit_kernel(self, X_train, y_train, kernel):
        if (kernel == "linear"):
            model = SVC(kernel="linear",
                        C=self.cost,
                        probability=False,
                        verbose=0).fit(X_train, y_train)
        elif (kernel == "poly"):
            model = SVC(kernel="poly",
                        C=self.cost,
                        gamma=self.poly_scale,
                        probability=False,
                        coef0=self.coef0_poly,
                        degree=self.degree,
                        verbose=0).fit(X_train, y_train)
        elif (kernel == "rbf"):
            model = SVC(kernel="rbf",
                        C=self.cost,
                        probability=False,
                        gamma='scale' if self.automatic_tuning else self.gamma_rbf,
                        verbose=0).fit(X_train, y_train)
        elif (kernel == "laplacian"):
            model = SVC(kernel=laplacian_kernel,
                        gamma='scale' if self.automatic_tuning else self.gamma_lap,
                        C=self.cost,
                        probability=False,
                        verbose=0).fit(X_train, y_train)
        return model
