"""
空值处理， 空值填充， 与删除
根据不同的数据类型，将数据进行转化， one hot ，ordinary
数据归一化：standard scale， minmax scale

降维， PCA， LDA
特征生成，比如log

拆分测试集与训练集-preprocessing.py
样本不均匀情况处理
"""
# preprocess data
from collections import Counter

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from imblearn.under_sampling import (ClusterCentroids, RandomUnderSampler,
                                     NearMiss,
                                     InstanceHardnessThreshold,
                                     CondensedNearestNeighbour,
                                     EditedNearestNeighbours,
                                     RepeatedEditedNearestNeighbours,
                                     AllKNN,
                                     NeighbourhoodCleaningRule,
                                     OneSidedSelection)
from imblearn.over_sampling import (ADASYN, SMOTE, BorderlineSMOTE, SVMSMOTE, SMOTENC,
                                    KMeansSMOTE, RandomOverSampler)


def process_nan_data(df):
    print("===" * 4, 'processing nan data', "===" * 4)
    origin_nan_situation = df.isnull().sum()
    print('nan situations', origin_nan_situation)

    df.fillna(df.mean(), axis=0)
    # df.fillna(df.median(), axis=0)
    # axis: {0 or ‘index’, 1 or ‘columns’}

    nan_situation = df.isnull().sum()
    print('nan situations', nan_situation)
    print('===' * 4, 'process finished')
    return df


def min_max_scale_data(df, min_max_names=['Time'], replace=False):
    """
    scale data
    :param df:
    :param min_max_names:
    :param replace:
    :return:
    """
    min_max_label = 'min_max_scale_' if replace is not True else ''
    for name in min_max_names:
        df.loc[:, min_max_label + name] = MinMaxScaler().fit_transform(df[name].values.reshape(-1, 1))
        print(min_max_label + name)
    print('===' * 4, 'min max scale finished')
    return df


def std_scale_data(df, std_names=['Amount'], replace=False):
    std_label = 'std_scale_' if replace is not True else ''
    for name in std_names:
        df.loc[:, std_label + name] = StandardScaler().fit_transform(df[name].values.reshape(-1, 1))
        print(std_label + name)
    print('===' * 4, 'standard scale finished')
    return df


# todo: unfininshed
def one_hot_data(df, column_names, categories, replace=False):
    print("===" * 4, 'transforming to one hot encoding', "===" * 4)
    one_hot_label = 'one_hot_' if replace is not True else ''
    enc = OneHotEncoder(categories=categories)
    for name in column_names:
        df[one_hot_label + name] = enc
    print('===' * 4, 'one hot encoding finished')
    return df


def under_sample(X, y, sampler="RandomUnderSampler"):
    # list of all samplers, in case you want to iterate all of them
    samplers_list = ['RandomUnderSampler', 'ClusterCentroids', 'NearMiss', 'InstanceHardnessThreshold',
                     'CondensedNearestNeighbour', 'EditedNearestNeighbours', 'RepeatedEditedNearestNeighbours',
                     'AllKNN', 'NeighbourhoodCleaningRule', 'OneSidedSelection']
    print(samplers_list)

    # currently there is no parameters sampler
    # this dict is used to choose a resampler by user. default is random
    samplers = {
        "RandomUnderSampler": RandomUnderSampler(),
        "ClusterCentroids": ClusterCentroids(),
        "NearMiss": NearMiss(),
        "InstanceHardnessThreshold": InstanceHardnessThreshold(),
        "CondensedNearestNeighbour": CondensedNearestNeighbour(),
        "EditedNearestNeighbours": EditedNearestNeighbours(),
        "RepeatedEditedNearestNeighbours": RepeatedEditedNearestNeighbours(),
        "AllKNN": AllKNN(),
        "NeighbourhoodCleaningRule": NeighbourhoodCleaningRule(),
        "OneSidedSelection": OneSidedSelection(),
    }
    sampler = samplers[sampler]

    # plot y class count before and after resample
    print("before", sorted(Counter(y).items()))

    # to resample simply call fit_resample method of sampler
    X_resampled, y_resampled = sampler.fit_resample(X, y)

    print("after", sorted(Counter(y_resampled).items()))

    print('===' * 4, 'under_sample finished')

    return X_resampled, y_resampled


def up_sample(X, y, sampler="RandomUnderSampler"):
    samplers = {
        "RandomOverSampler": RandomOverSampler(),
        "KMeansSMOTE": KMeansSMOTE(),
        "ADASYN": ADASYN(),
        "SMOTE": SMOTE(),
        "BorderlineSMOTE": BorderlineSMOTE(),
        "SVMSMOTE": SVMSMOTE(),
        "SMOTENC": SMOTENC(),
    }
    sampler = samplers[sampler]

    # plot y class count before and after resample
    print("before", sorted(Counter(y).items()))

    # to resample simply call fit_resample method of sampler
    X_resampled, y_resampled = sampler.fit_resample(X, y)

    print("after", sorted(Counter(y_resampled).items()))

    print('===' * 4, 'under_sample finished')

    return X_resampled, y_resampled


def smote_sample(X, y):
    return up_sample(X, y, sampler="Smote")


def split_save(df, random_seed=10, size=0.3, name=''):
    """
    split dataset
    :param df:
    :param random_seed:
    :param size:
    :param name:
    :return:
    """
    from sklearn.model_selection import train_test_split
    df_train, df_test = train_test_split(df, test_size=size, random_state=random_seed)
    df_train.to_csv('./data/' + name + 'df_train.csv')
    df_test.to_csv('./data/' + name + 'df_test.csv')
    print('===' * 4, 'split and save finished')
    return df_train, df_test


def pipeline_and_gv(x_train, y_train):
    pipeline = Pipeline(steps=[
        # ('minmax', MinMaxScaler()),  # each tuple here defines a step in pipeline
        ('logistic', LogisticRegression())
    ], memory='memory')
    gv_params = {
        'logistic__C': [0.5, 1]
    }
    gv = GridSearchCV(pipeline, gv_params, cv=3)
    gv.fit(x_train, y_train)
    return gv.best_params_


def pipeline4column(x_train, y_train):
    features_1 = ['V1']
    transformer_1 = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    features_2 = ['V2', 'V3']
    transformer_2 = MinMaxScaler()

    colTrans = ColumnTransformer(
        transformers=[
            ('use_pipeline', transformer_1, features_1),
            ('or_single_transformer', transformer_2, features_2)])

    print(colTrans.fit_transform(x_train, y_train).shape)  # colTrans can be chained in pipeline or not

    pipeline = Pipeline(steps=[
        ('colTrans', colTrans),  # each tuple here defines a step in pipeline
        ('logistic', LogisticRegression())]
        # , memory='memory'
    )
    pipeline.fit(x_train, y_train)
    return pipeline.score(x_train, y_train)


# this is a process for specific columns
params = {
    'min_max_scale_columns': (min_max_scale_data, ['Time'])
    , 'std_scale_columns': (std_scale_data, ['Amount'])
    # 'one_hot_columns' : []
}


def create_imbalance_dataset(n_samples=1000, weights=(0.01, 0.01, 0.98), n_classes=3,
                             class_sep=0.8, n_clusters=1):
    return make_classification(n_samples=n_samples, n_features=2,
                               n_informative=2, n_redundant=0, n_repeated=0,
                               n_classes=n_classes,
                               n_clusters_per_class=n_clusters,
                               weights=list(weights),
                               class_sep=class_sep, random_state=0)


if __name__ == '__main__':
    filename = './data/under_sample_data.csv'
    load_data = pd.read_csv(filename)
    df_train, df_test = split_save(load_data)
    # # feature_names = load_data.columns.tolist()
    # # data = scale_data(load_data)
    # # data.columns.tolist()
    # # load_data.columns.tolist()
    # # df = df_train

    x_train, y_train = df_train.loc[:, df_train.columns != 'Class'], df_train.loc[:, 'Class']
    x, y = under_sample(x_train, y_train)
    print(x.shape, y.shape)
    # y_train = pd.DataFrame(np.array(y_train).reshape(-1, 1))
    # print(pipeline4column(x_train, y_train))
    # X, y = create_imbalance_dataset(n_samples=5000, weights=(0.01, 0.05, 0.94),
    #                                 class_sep=0.8)
    # under_sample(X, y)
