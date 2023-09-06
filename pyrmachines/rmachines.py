import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import laplacian_kernel


class RandomMachines:

    def __init__(self,
                 poly_scale=2,
                 coef0_poly=0,
                 gamma_rbf=1,
                 degree=2,
                 cost=10,
                 boots_size=25, seed_bootstrap=None, automatic_tuning=False):
        """
        Parameters:
            poly_scale: float, default=2
            coef0_poly: float, default=0
            gamma_rbf: float, default=1
            degree: float, default=2
            cost: float, default=10
            boots_size: float, default=25
            automatic_tuning: bool, default=False
        """
        self.poly_scale = poly_scale
        self.coef0_poly = coef0_poly
        self.gamma_rbf = gamma_rbf
        self.degree = degree
        self.cost = cost
        self.boots_size = boots_size
        self.seed_bootstrap = seed_bootstrap
        self.automatic_tuning = automatic_tuning

    def create_bootstrap_sample(self, X_train, y_train):
        X_train_size = X_train.shape[0]
        X_train_range = range(X_train_size)
        train_index = np.random.choice(
            range(X_train_size), size=X_train_size, replace=True)
        boots_sample_y = y_train.iloc[train_index]
        frequency_table = boots_sample_y.value_counts()
        while (frequency_table.iloc[1] < 2):
            train_index = np.random.choice(
                X_train_range, size=X_train_size, replace=True)
            boots_sample_y = y_train.iloc[train_index]
            frequency_table = boots_sample_y.value_counts()

        return ({
            'X_train': X_train.iloc[train_index],
            'y_train': y_train.iloc[train_index],
            'X_test': X_train.drop(train_index),
            'y_test': y_train.drop(train_index)})

    def fit_kernel(self, X_train, y_train, kernel):
        if (self.automatic_tuning):
            if (kernel == "laplacian"):
                model = SVC(kernel=laplacian_kernel).fit(X_train, y_train)
            else:
                model = SVC(kernel=kernel).fit(X_train, y_train)
            return model
        else:
            if (kernel == "linear"):
                model = SVC(kernel="linear",
                            C=self.cost,
                            probability=True,
                            verbose=0).fit(X_train, y_train)
            elif (kernel == "poly"):
                model = SVC(kernel="poly",
                            C=self.cost,
                            gamma=self.poly_scale,
                            probability=True,
                            coef0=self.coef0_poly,
                            degree=self.degree,
                            verbose=0).fit(X_train, y_train)
            elif (kernel == "rbf"):
                model = SVC(kernel="rbf",
                            C=self.cost,
                            probability=True,
                            gamma=self.gamma_rbf,
                            verbose=0).fit(X_train, y_train)
            elif (kernel == "laplacian"):
                model = SVC(kernel=laplacian_kernel,
                            C=self.cost,
                            probability=True,
                            verbose=0).fit(X_train, y_train)
            return model

    def fit(self, X_train, y_train):
        # TODO: Implementar automatic_tuning
        # TODO: Implementar parametros para o kernel laplacian

        kernel_type = ["linear", "poly", "rbf", "laplacian"]

        # Set seed
        if (self.seed_bootstrap is not None):
            np.random.seed(self.seed_bootstrap)
        # Reset index
        X_train = X_train.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)

        # Get Levels
        levels = y_train.unique()

        early_models = []
        for kernel in kernel_type:
            model = self.fit_kernel(X_train, y_train, kernel)
            # TODO: Verificar é para usar X_train
            # TODO: Trocar a funcao log pois esta penalizando acc = 1
            predict = model.predict(X_train)
            accuracy = accuracy_score(y_train, predict)
            if (accuracy == 1):
                log_acc = 1
            else:
                log_acc = np.log(np.divide(accuracy, np.subtract(1, accuracy)))
            if (np.isinf(log_acc)):
                log_acc = 1
            early_models.append(
                {'kernel': kernel, "model": model, 'accuracy': accuracy, 'metric': log_acc})

        # Calculando as probabilidades para cada kernel
        prob_weights_sum = sum(item["metric"] for item in early_models)
        lambda_values = {}
        for model in early_models:
            prob_weights = model["metric"] / prob_weights_sum
            if (prob_weights < 0):
                prob_weights = 0
            model["prob_weights"] = prob_weights
            lambda_values[model["kernel"]] = prob_weights
            print(
                f"Kernel: {model['kernel']} - Accuracy: {model['accuracy']} - Prob_weights: {prob_weights}")

        # Sorteando os kernels
        p = [item["prob_weights"] for item in early_models]
        random_kernel = np.random.choice(
            kernel_type, self.boots_size, replace=True, p=p)

        # Criando as amostras de bootstrap
        boots_sample = [self.create_bootstrap_sample(
            X_train, y_train) for i in range(self.boots_size)]

        models = []
        for index in range(len(random_kernel)):
            kernel = random_kernel[index]
            boot_sample = boots_sample[index]
            model = self.fit_kernel(
                boot_sample["X_train"], boot_sample["y_train"], kernel)
            # Predict na base de out of bag
            predict_oobg = model.predict(boot_sample["X_test"])
            accuracy = accuracy_score(boot_sample["y_test"], predict_oobg)
            kernel_weight = 1 / (accuracy ** 2)
            models.append({'model': model, 'kernel': kernel,
                           'accuracy': accuracy, 'kernel_weight': kernel_weight})

        model_result = {
            "levels": levels,
            "lambda_values": lambda_values,
            "model_params": {
                "poly_scale": self.poly_scale,
                "coef0_poly": self.coef0_poly,
                "gamma_rbf": self.gamma_rbf,
                "degree": self.degree,
                "cost": self.cost,
                "boots_size": self.boots_size,
            },
            "bootstrap_models": models,
            "bootstrap_samples": boots_sample
        }
        self.model_result = model_result
        return self

    def predict(self, X):
        # TODO: Varificar se nrow é maior que 0
        nrow = X.shape[0]
        ncol = len(self.model_result["levels"])
        models = self.model_result['bootstrap_models']
        predict_df = pd.DataFrame(
            np.zeros((nrow, ncol)), columns=self.model_result["levels"])
        for model in models:
            model_weights = model["kernel_weight"]
            predict = model["model"].predict(X)
            for i in range(len(predict)):
                predict_df.loc[i, predict[i]] += model_weights
        return list(predict_df.idxmax(axis=1))
