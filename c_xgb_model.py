import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools
import pandas as pd
import graphviz


def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    plot confusion matrix
    :param cm:
    :param classes:
    :param title: str
    :param cmap: color map
    :return: plot
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0

    print("thresh:", thresh, )
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def df2dmatrix(df, column_name='Class'):
    """
    transforming dataFrame to dmatrix data
    :param df: dataframe
    :param column_name: the label named
    :return: dmatrix data
    """
    data = df.iloc[:, df.columns != column_name]
    label = df[column_name].values
    dmatrx_data = xgb.DMatrix(data, label)
    return dmatrx_data


# loading data type:txt
df = pd.read_csv('./data/under_sample_data.csv')
dtrain = df2dmatrix(df)
# dtrain = xgb.DMatrix('./data/agaricus.txt.train')
# dtest = xgb.DMatrix('./data/agaricus.txt.test')
dtest = dtrain

# setting parameters
param = {
    'max_depth': 2,
    'eta': 1,
    'silent': 1,
    'objective': 'binary:logistic'
}
num_round = 2

# use parameters to train
bst = xgb.train(param, dtrain, num_round)
train_preds = bst.predict(dtrain)
train_preds = [round(train_pred) for train_pred in train_preds]

# preds data
preds = bst.predict(dtest)
preds = [round(pred) for pred in preds]
y_train = dtrain.get_label()
y_test = dtest.get_label()

# showing accuracy
test_accuracy = accuracy_score(y_test, preds)
train_accuracy = accuracy_score(y_train, train_preds)
print('*' * 10, '\n training accuracy is: ', train_accuracy)
print('test acc is:', test_accuracy)

# visualization
train_cm = confusion_matrix(y_train, train_preds)
test_cm = confusion_matrix(y_test, preds)

plot_confusion_matrix(train_cm, [0, 1], 'train cf')
plt.show()

plot_confusion_matrix(test_cm, [0, 1], 'test CF')
plt.show()

# using xgb to plot

xgb.plot_importance(bst)
plt.show()

# todo: something wrong, did not work out
# xgb.plot_tree(bst, num_trees=1)
# plt.show()
