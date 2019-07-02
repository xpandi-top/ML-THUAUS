# Using the prediction pipeline in a grid search
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor

# Random Forest parameters
rf_params = {
    'n_jobs':           -1,
    'n_estimators':     500,
    'warm_start':       True,
    # 'max_features': 0.2,
    'max_depth':        6,
    'min_samples_leaf': 2,
    'max_features':     'sqrt',
    'verbose':          0
}

# Extra Trees Parameters
et_params = {
    'n_jobs':           -1,
    'n_estimators':     500,
    # 'max_features': 0.5,
    'max_depth':        8,
    'min_samples_leaf': 2,
    'verbose':          0
}

# AdaBoost parameters
ada_params = {
    'n_estimators':  500,
    'learning_rate': 0.75
}

# Gradient Boosting parameters
gb_params = {
    'n_estimators':     500,
    # 'max_features': 0.2,
    'max_depth':        5,
    'min_samples_leaf': 2,
    'verbose':          0
}

# Support Vector Classifier parameters
svc_params = {
    'kernel': 'linear',
    'C':      0.025
}
params = {
    'n_estimators': [100, 200, 500, 800, 1000]
}
clf = RandomForestRegressor(**rf_params)
gv = GridSearchCV(clf, params, cv=3, scoring='neg_mean_squared_error', verbose=True, n_jobs=-1)
x_train = x_train_paid.values  # Creates an array of the train data
#x_test = x_test_paid.values  # Creats an array of the test data
# gv.fit(x_train, y_train_paid)
gv.fit(x_train_paid, y_train_paid)
print(gv.best_params_)
