"""
模型结构呈现
模型参数保存
数据结果保存
各特征贡献概览-可视化
以及一些好用的可视化技巧

"""
import matplotlib.pyplot as plt
import itertools
import numpy as np
import pandas as pd


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