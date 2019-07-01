from operator import itemgetter
from time import time

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from b_preprocessing import split_save
from costcla import savings_score, cost_loss
from costcla.models import BayesMinimumRiskClassifier, CostSensitiveDecisionTreeClassifier


# Utility function to report best scores
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
            score.mean_validation_score,
            np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")


if __name__ == "__main__":
    df = pd.read_csv('./data/under_sample_data.csv')
    df_train, df_test = split_save(df)
    x_train, y_train = df_train.loc[:, df_train.columns != 'Class'], df_train.loc[:, 'Class']
    x_test, y_test = df_test.loc[:, df_train.columns != 'Class'], df_test.loc[:, 'Class']
    # y_train = pd.DataFrame(np.array(y_train).reshape(-1, 1))
    print(x_train.shape, y_train.shape)
    # print(x_test.shape, y_test.shape)

    # only counting for training sample
    counts = pd.DataFrame(pd.Series(df_train['Class'].value_counts()))
    counts['Percentage'] = counts['Class'] / counts['Class'].sum()
    print(counts)

    classifiers = {
        "RF": {"f": RandomForestClassifier()},  # each model is defined as dict to save results together with the model
        "DT": {"f": DecisionTreeClassifier()},
        "LR": {"f": LogisticRegression()}
    }

    for model in classifiers:
        classifiers[model]["f"].fit(x_train, y_train)
        classifiers[model]["c"] = classifiers[model]["f"].predict(x_test)
        classifiers[model]["p"] = classifiers[model]["f"].predict_proba(x_test)
        classifiers[model]["p_train"] = classifiers[model]["f"].predict_proba(x_train)

    measures = {"f1": f1_score, "pre": precision_score,
                "rec": recall_score, "acc": accuracy_score}
    results = pd.DataFrame(columns=measures.keys())

    for model in classifiers.keys():
        results.loc[model] = [measures[measure](y_test, classifiers[model]["c"]) for measure in measures.keys()]

    cost_mat_test = np.array([(100, 100, 0., 0.) for _ in range(x_test.shape[0])]) # todo: how is this calculated

    results["sav"] = 0
    for model in classifiers.keys():
        results.loc[model, "sav"] = savings_score(y_test, classifiers[model]["c"], cost_mat_test)

    for model in list(classifiers.keys()):
        classifiers[model + "-BMR"] = {"f": BayesMinimumRiskClassifier()}
        classifiers[model + "-BMR"]["f"].fit(y_test, classifiers[model]["p"])
        classifiers[model + "-BMR"]["c"] = classifiers[model + "-BMR"]["f"].predict(classifiers[model]["p"],
                                                                                    cost_mat_test)
        results.loc[model + "-BMR"] = 0
        results.loc[model + "-BMR", measures.keys()] = [measures[measure](y_test, classifiers[model + "-BMR"]["c"]) for measure in measures.keys()]
        results.loc[model + "-BMR", "sav"] = savings_score(y_test, classifiers[model + "-BMR"]["c"], cost_mat_test)

    classifiers["CSDT"] = {"f": CostSensitiveDecisionTreeClassifier()}
    cost_mat_train = np.array([(100, 100, 10, 1) for _ in range(x_train.shape[0])])
    classifiers["CSDT"]["f"].fit(x_train, y_train, cost_mat_train)
    classifiers["CSDT"]["c"] = classifiers["CSDT"]["f"].predict(x_test)
    results.loc["CSDT"] = 0
    results.loc["CSDT", measures.keys()] = [measures[measure](y_test, classifiers["CSDT"]["c"]) for measure in measures.keys()]
    results["sav"].loc["CSDT"] = savings_score(y_test, classifiers["CSDT"]["c"], cost_mat_test)

    print(results)
    # todo: CostSensitiveRandomPatchesClassifier

    clf = RandomForestClassifier()

    # specify parameters and distributions to sample from
    # from http://scikit-learn.org/stable/auto_examples/model_selection/randomized_search.html
    param_dist = {"n_estimators": [10, 20, 50, 100, 1000],
                  "max_depth": [3, None],
                  # "max_features": sp_randint(1, 10),
                  # "min_samples_split": sp_randint(1, 100),
                  # "min_samples_leaf": sp_randint(1, 100),
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}

    # run randomized search
    classifiers["RS-RF"] = {"f": RandomizedSearchCV(clf, param_distributions=param_dist,
                                                    n_iter=20, n_jobs=4, verbose=1)}
    # Fit
    start = time()
    classifiers["RS-RF"]["f"].fit(x_train, y_train)
    print("RandomizedSearchCV took %.2f seconds"
          " parameter settings." % ((time() - start),))

    report(classifiers["RS-RF"]["f"].grid_scores_)
