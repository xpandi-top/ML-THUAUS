# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, recall_score


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# import os
# print(os.listdir("../input"))
def printing_Kfold_scores(x_train_data, y_train_data):
    kf = KFold(n_splits=5)
    # fold = KFold(len(y_train_data),5, shuffle=False)
    c_param_range = [0.01, 0.1, 1, 10, 100]
    results_table = pd.DataFrame(index=range(len(c_param_range), 2), columns=['C_parameter', 'Mean recall score'])
    results_table['C_parameter'] = c_param_range

    for j, c_param in enumerate(c_param_range):
        print('-----C parameter: ', c_param, '-----\n')
        recall_accs = []
        iteration = 1
        for train_index, test_index in kf.split(x_train_data):
            lr = LogisticRegression(C=c_param, penalty='l1')
            lr.fit(x_train_data.iloc[train_index, :], y_train_data.iloc[train_index, :].values.ravel())
            y_predict_undersample = lr.predict(x_train_data.iloc[test_index, :].values)
            recall_acc = recall_score(y_train_data.iloc[test_index, :].values, y_predict_undersample)
            recall_accs.append(recall_acc)
            print('Interation ', iteration, 'recall score = ', recall_acc)

        results_table.loc[j, "Mean recall score"] = np.mean(recall_accs)
        print('\n Mean recall score ', np.mean(recall_accs), '\n')

    best_c = results_table.loc[results_table['Mean recall score'].astype(float).idxmax()]['C_parameter']
    print('*' * 20, '\n best c is ', best_c, '\n', "*" * 20)
    return best_c


def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
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


# Any results you write to the current directory are saved as output.
## 读取数据
credit_data = pd.read_csv('./data/preprocessed_data.csv')
# print(credit_data[:5]) # data.head() 查看数据
## 查看数据
credit_data.head()

### 查看class分布
count_classes = pd.value_counts(credit_data['Class'], sort=True).sort_index()
count_classes.plot(kind='bar')
### data balance undersample
data = credit_data.iloc[:, credit_data.columns != 'Class']
classes = credit_data.iloc[:, credit_data.columns == "Class"]
number_fraud = len(classes[classes.Class == 1])
print('fraud number is ', number_fraud)
print('no fraud is ', len(classes) - number_fraud)

normal_indices = credit_data[credit_data.Class == 0].index
fraud_indices = credit_data[credit_data.Class == 1].index

random_norm_indices = np.random.choice(normal_indices, number_fraud, replace=False)
# print(len(normal_indices))
print(fraud_indices[:5])
print(random_norm_indices[:5])
under_sample_indices = np.concatenate([fraud_indices, random_norm_indices])
print(len(under_sample_indices))
print(under_sample_indices[:5])
under_sample_data = credit_data.iloc[under_sample_indices, :]
under_sample_data.head()
under_sample_data.to_csv('under_sample_data.csv', index=False)

## get data and label

under_sample_x = under_sample_data.iloc[:, under_sample_data.columns != 'Class']
under_sample_y = under_sample_data.iloc[:, under_sample_data.columns == 'Class']
under_sample_x.head()
under_sample_y.head()
# counts_classes = pd.value_counts(under_sample_y['Class'],sort=True).sort_index()
# counts_classes.plot(kind='bar')

#### data precessing
# data split
X_train, X_test, y_train, y_test = train_test_split(under_sample_x, under_sample_y, test_size=0.3, random_state=10)

best_c = printing_Kfold_scores(X_train, y_train)
print(best_c)

lr = LogisticRegression(C=0.001, penalty='l1')
lr.fit(X_train, y_train.values.ravel())
y_test_pred = lr.predict(X_test.values)
cnf_matrix = confusion_matrix(y_test, y_test_pred)
np.set_printoptions(precision=2)

print(y_test_pred)
print(lr)

print('recall metric in the testing dataset: ', cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))

class_names = [0, 1]
plt.figure()

plot_confusion_matrix(cnf_matrix, classes=class_names)
plt.show()
