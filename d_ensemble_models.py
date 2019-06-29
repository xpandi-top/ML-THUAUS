import pandas as pd
import numpy as np
import re
import xgboost as xgb

import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline 内嵌画图

import plotly.offline as py

# py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

from c_xgb_model import plot_confusion_matrix

import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, \
    ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

filename = './data/under_sample_data.csv'
data = pd.read_csv(filename)
under_sample_x = data.iloc[:, data.columns != 'Class']
under_sample_x = under_sample_x.drop(['Time'], axis=1)
under_sample_y = data.iloc[:, data.columns == 'Class']
X_train, X_test, y_train, y_test = train_test_split(under_sample_x, under_sample_y, test_size=0.3, random_state=10)

# Pearson Correlation Heatmap use seabone, showing the relationships between features

colormap = plt.cm.RdBu
plt.figure(figsize=(14, 12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(X_train.astype(float).corr(), linewidths=0.1, vmax=1.0,
            square=True, cmap=colormap, linecolor='white', annot=True)
plt.show()

### Helpers via Python Classes
ntrain = X_train.shape[0]
ntest = X_test.shape[0]
SEED = 0  # for reproducibility
NFOLDS = 5  # set folds for out-of-fold prediction
kf = KFold(n_splits=NFOLDS, random_state=SEED)


# Class to extend the Sklearn classifier
class SklearnHelper(object):
    """
    Class to extend the Sklearn classifier
    """

    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

    def fit(self, x, y):
        return self.clf.fit(x, y)

    def feature_importances(self, x, y):
        print(self.clf.fit(x, y).feature_importances_)


def get_oof(clf, x_train, y_train, x_test):
    """
    get off
    :param clf: training with kfold
    :param x_train: data
    :param y_train: data
    :param x_test: data
    :return:
    """
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf.split(X_train)):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


# Put in our parameters for said classifiers
# Random Forest parameters
rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
    'warm_start': True,
    # 'max_features': 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'verbose': 0
}

# Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    'n_estimators': 500,
    # 'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 500,
    'learning_rate': 0.75
}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': 500,
    # 'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
}

# Support Vector Classifier parameters
svc_params = {
    'kernel': 'linear',
    'C': 0.025
}

# Create 5 objects that represent our 4 models
rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)

x_train = X_train.values  # Creates an array of the train data
x_test = X_test.values  # Creats an array of the test data
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# Create our OOF train and test predictions. These base results will be used as new features
et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test)  # Extra Trees
rf_oof_train, rf_oof_test = get_oof(rf, x_train, y_train, x_test)  # Random Forest
ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test)  # AdaBoost
gb_oof_train, gb_oof_test = get_oof(gb, x_train, y_train, x_test)  # Gradient Boost
svc_oof_train, svc_oof_test = get_oof(svc, x_train, y_train, x_test)  # Support Vector Classifier

print("Training is complete")

rf_feature = rf.feature_importances(x_train, y_train)
et_feature = et.feature_importances(x_train, y_train)
ada_feature = ada.feature_importances(x_train, y_train)
gb_feature = gb.feature_importances(x_train, y_train)

rf_features = rf.clf.feature_importances_
et_features = et.clf.feature_importances_
ada_features = ada.clf.feature_importances_
gb_features = gb.clf.feature_importances_

cols = X_train.columns.values
# Create a dataframe with features
feature_dataframe = pd.DataFrame({'features': cols,
                                  'Random Forest feature importances': rf_features,
                                  'Extra Trees  feature importances': et_features,
                                  'AdaBoost feature importances': ada_features,
                                  'Gradient Boost feature importances': gb_features
                                  })

feature_dataframe['mean'] = feature_dataframe.mean(axis=1)  # axis = 1 computes the mean row-wise
feature_dataframe.head(3)

###########
## Second-Level Predictions from the First-level Output
base_predictions_train = pd.DataFrame({'RandomForest': rf_oof_train.ravel(),
                                       'ExtraTrees': et_oof_train.ravel(),
                                       'AdaBoost': ada_oof_train.ravel(),
                                       'GradientBoost': gb_oof_train.ravel()
                                       })
base_predictions_train.head()

data = [
    go.Heatmap(
        z=base_predictions_train.astype(float).corr().values,
        x=base_predictions_train.columns.values,
        y=base_predictions_train.columns.values,
        colorscale='Viridis',
        showscale=True,
        reversescale=True
    )
]
py.iplot(data, filename='labelled-heatmap')

x_train = np.concatenate((et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)
x_test = np.concatenate((et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)

gbm = xgb.XGBClassifier(
    # learning_rate = 0.02,
    n_estimators=2000,
    max_depth=4,
    min_child_weight=2,
    # gamma=1,
    gamma=0.9,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    nthread=-1,
    scale_pos_weight=1).fit(x_train, y_train)
predictions = gbm.predict(x_test)

confusion_matrix(y_test, predictions)
plot_confusion_matrix(confusion_matrix(y_test, predictions), [0, 1], 'ensemble model')

# submit file
# Generate Submission File
StackingSubmission = pd.DataFrame({'PassengerId': X_test.index.tolist(),
                                   'Class': predictions})
StackingSubmission.to_csv("./StackingSubmission.csv", index=False)
