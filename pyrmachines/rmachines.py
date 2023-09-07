import numpy as np
import pandas as pd
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.metrics.pairwise import laplacian_kernel


class RandomMachinesClassifier:

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
        self.create_bootstrap_sample = utils().create_bootstrap_sample
        self.type = "classifier"

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
            X_train, y_train, self.type) for i in range(self.boots_size)]

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


class RandomMachinesRegression:

    def __init__(self,
                 poly_scale=2,
                 coef0_poly=0,
                 gamma_rbf=1,
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
        self.degree = degree
        self.cost = cost
        self.epsilon = epsilon
        self.beta = beta
        self.boots_size = boots_size
        self.seed_bootstrap = seed_bootstrap
        self.automatic_tuning = automatic_tuning
        self.create_bootstrap_sample = utils().create_bootstrap_sample
        self.type = "regression"

    def fit_kernel(self, X_train, y_train, kernel):
        if (self.automatic_tuning):
            if (kernel == "laplacian"):
                model = SVR(kernel=laplacian_kernel).fit(X_train, y_train)
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

    def fit(self, X_train, y_train):
        kernel_type = ["linear", "poly", "rbf", "laplacian"]

        # Set seed
        if (self.seed_bootstrap is not None):
            np.random.seed(self.seed_bootstrap)
        # Reset index
        X_train = X_train.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)

        early_models = []
        for kernel in kernel_type:
            model = self.fit_kernel(X_train, y_train, kernel)
            predict = model.predict(X_train)
            rmse = mean_squared_error(y_train, predict, squared=False)
            early_models.append(
                {'kernel': kernel, "model": model, 'metric': rmse})
            print(f"Kernel: {kernel} - RMSE: {rmse}")

        sd_rmse = np.std([item["metric"] for item in early_models], ddof=1)
        rmse = np.divide(np.array([item["metric"]
                         for item in early_models]), sd_rmse)
        inv_rmse = np.exp(np.dot(-rmse, self.beta))
        prob_weights = np.divide(inv_rmse, np.sum(inv_rmse))
        prob_weights[prob_weights < 0] = 0

        # Sorteando os kernels
        random_kernel = np.random.choice(
            kernel_type, self.boots_size, replace=True, p=prob_weights)

        # Criando as amostras de bootstrap
        boots_sample = [self.create_bootstrap_sample(
            X_train, y_train, self.type) for i in range(self.boots_size)]

        models = []
        for index in range(len(random_kernel)):
            kernel = random_kernel[index]
            boot_sample = boots_sample[index]
            model = self.fit_kernel(
                boot_sample["X_train"], boot_sample["y_train"], kernel)
            # Predict na base de out of bag
            predict_oobg = model.predict(boot_sample["X_test"])
            loss_function = mean_squared_error(
                boot_sample["y_test"], predict_oobg, squared=False)
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

        model_result = {
            "lambda_values": prob_weights,
            "model_params": {
                "poly_scale": self.poly_scale,
                "coef0_poly": self.coef0_poly,
                "gamma_rbf": self.gamma_rbf,
                "degree": self.degree,
                "cost": self.cost,
                "boots_size": self.boots_size,
            },
            "bootstrap_models": models,
        }
        self.model_result = model_result
        return self

    def predict(self, X):
        models = self.model_result['bootstrap_models']
        final_predict = []
        for model in models:
            model_weights = model["kernel_weight"]
            predict = model["model"].predict(X)
            multiply = np.multiply(model_weights, predict)
            final_predict.append(multiply)
        return list(np.sum(final_predict, axis=0))


class utils:
    def __init__(self):
        pass

    def create_bootstrap_sample(self, X_train, y_train, type):
        X_train_size = X_train.shape[0]
        X_train_range = range(X_train_size)
        train_index = np.random.choice(
            range(X_train_size), size=X_train_size, replace=True)
        boots_sample_y = y_train.iloc[train_index]
        if (type == "classifier"):
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
