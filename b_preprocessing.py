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
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder


def process_nan_data(df):
    print("===" * 4, 'processing nan data', "===" * 4)
    origin_nan_situation = df.isnull().sum()
    print('nan situations', origin_nan_situation)
    print('===' * 4, 'process finished')
    return df


def scale_data(df, min_max_names=['Time'], std_names=['Amount'], replace=False):
    """
    scale data
    :param df:
    :param min_max_names:
    :param std_names:
    :param replace:
    :return:
    """
    print("===" * 4, 'transforming to one hot encoding', "===" * 4)
    print(" min max scaling: ", min_max_names, '\n', "standard scaling : ", std_names)
    min_max_label = 'min_max_scale_' if replace is not True else ''
    std_label = 'std_scale_' if replace is not True else ''
    for name in min_max_names:
        df[min_max_label + name] = MinMaxScaler().fit_transform(df[name].values.reshape(-1, 1))
        print(min_max_label + name)

    for name in std_names:
        df[std_label + name] = StandardScaler().fit_transform(df[name].values.reshape(-1, 1))
        print(std_label + name)

    print('===' * 4, 'scale finished')
    return df


# todo one hot encoding
def one_hot_data(df, column_names, categories, replace=False):
    one_hot_label = 'one_hot_' if replace is not True else ''
    enc = OneHotEncoder(categories=categories)
    for name in column_names:
        df[one_hot_label + name] = enc
    print('===' * 4, 'one hot encoding finished')
    return df


# todo: not finished
def under_sample(df):
    under_sample_df = df
    print('===' * 4, 'under_sample finished')
    return under_sample_df


# todo: not finished
def up_sample(df):
    up_sample_df = df
    print('===' * 4, 'up sampling finished')
    return up_sample_df


# todo: not finished
def smote_sample(df):
    smote_df = df
    print('===' * 4, 'smote sampling finished')
    return smote_df


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
    df_test.to_csv('./data/' + name + 'df_train.csv')
    print('===' * 4, 'split and save finished')
    return df_train, df_test


params = {
    'min_max_scale_columns': (scale_data, ['TIme'])
    , 'std_scale_columns': ['Amount']
    # 'one_hot_columns' : []
}

if __name__ == '__main__':
    filename = './data/under_sample_data.csv'
    load_data = pd.read_csv(filename)
    feature_names = load_data.columns.tolist()

    data = scale_data(load_data)
    data.columns.tolist()
    load_data.columns.tolist()
