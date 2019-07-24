from sklearn.linear_model import ElasticNet
import pandas as pd
import numpy as np


class ElasticLOO:
    """
    Leave-one-out CV for Elastic Net
    """
    def __init__(self, x, y=None, y_col=None, verbose=False, score='mse', name='ElasticModel'):
        """
        init function for this class
        """
        # todo: explain arguments here
        # load data
        if y is None and y_col is None:
            raise ValueError('please provide either y or y_col')
        elif y is None:
            self.y = x[y_col]
        else:
            self.y = y
        self.x = x

        # other arguments
        if callable(score):
            self.score = score
        elif score == 'mse':
            self.score = lambda pred_real: (pred_real[0] - pred_real[1])**2

        self.n_samples = x.shape[0]
        self.verbose = verbose
        self.name = name
        self.error = 0

    def train(self, **params):
        for val_index in range(self.n_samples):
            if self.verbose:
                print("running cv at {0}/{1}".format(val_index, self.n_samples))
            train_index = list(range(self.n_samples))
            train_index.remove(val_index)
            x_train = self.x.iloc[train_index, :]
            y_train = self.y.iloc[train_index]

            x_test = np.array(self.x.iloc[val_index, :]).reshape(1, -1)
            y_test = self.y.iloc[val_index]

            model = ElasticNet(**params)
            model.fit(x_train, y_train)
            self.error += float(self.score((model.predict(x_test), y_test)))
        self.error /= self.n_samples
        print("mean for model:{0} is {1:.3f}.".format(self.name, self.error))


if __name__ == "__main__":
    # only a runnable demo for class ElasticLOO
    from sklearn import datasets

    EPSILON = 1e-4

    diabetes = datasets.load_diabetes()
    X = pd.DataFrame(diabetes.data)
    y = pd.DataFrame(diabetes.target)

    loo = ElasticLOO(X, y=y, verbose=True)
    loo.train()
