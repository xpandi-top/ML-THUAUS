import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import datetime
import json
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from imblearn.over_sampling import (ADASYN, SMOTE, BorderlineSMOTE, SVMSMOTE, SMOTENC,
                                    RandomOverSampler)
from collections import Counter
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression,ElasticNet


def json_load(arr):
    if arr["jsonstr"]:
        try:
            return json.loads(arr["jsonstr"])
        except json.decoder.JSONDecodeError:
            return arr["jsonstr"]
    else:
        return None


def extract_info(arr):
    _json_str = ""
    _amount = 0
    _reason = ""
    _status, _result = arr["info"].split(" for ", 1)
    if _status == "Quote Completed" or _status == "Quote Incomplete":
        _customer, _json_str = _result.split(" with json payload ")
        _customer = _customer[-6:]
        _json_str = _json_str.replace("\'", "\"")
    elif _status == "Claim Accepted":
        _customer, _amount = _result.split(" - paid $")
        _amount = float(_amount)
        _customer = _customer[-6:]
    elif _status == "Claim Denied":
        _customer, _reason = _result.split(" - reason : ")
        _customer = _customer[-6:]
    else:
        _customer = _result[-6:]

    return _status, _customer, _amount, _reason, _json_str


def load_data(file, header=True):
    if header:
        _data = pd.read_csv(file, header=0, index_col=None)
    else:
        _data = pd.read_csv(file, header=None, index_col=None)
        _data.columns = ["message", "timestamp"]
    #     _data["time"] = _data.apply(lambda x: datetime.datetime.fromtimestamp(x["timestamp"]), axis=1)
    _data["time"] = _data["timestamp"]

    _split_tmp = _data["message"].str.split(" - ", 2, expand=True)
    _split_tmp.columns = ["id", "browser", "info"]

    _data = pd.concat([_data, _split_tmp], axis=1)

    _data["status"], _data["customer"], _data["amount"], _data["reason"], _data["jsonstr"] = zip(
            *_data.apply(extract_info, axis=1))
    _data = _data.drop(["message", "timestamp", "info"], axis=1)

    _data["finalDict"] = _data.apply(json_load, axis=1)
    _data = _data.drop("jsonstr", axis=1)
    return _data


def get_dict_detail(customer_dict, keys_ordered):
    result = {k: None for k in keys_ordered}
    if customer_dict:
        for k, v in customer_dict.items():
            if k != "home":
                result[k] = v
            # elif k == "household":
            #    result[k] = len(v)
            else:
                for kk, vv in v.items():
                    result[kk] = vv
    return result


def convert_data(train_data):
    _groups = train_data.groupby("customer")
    _customer_rows = []
    _all_statuses = ['Quote Started', 'Quote Completed', 'Payment Completed', 'Claim Started', 'Claim Accepted',
                     'Policy Cancelled', 'Quote Incomplete', 'Claim Denied']
    _has_detail = [[], ["finalDict"], [], [], ["amount"], [], ["finalDict"], ["reason"]]
    keys_ordered = ['gender', 'name', 'household', 'age', 'address', 'email',
                    'square_footage', 'number_of_floors', 'type', 'number_of_bedrooms']

    _all_columns = ['Quote Started_time', 'Quote Started_platform',
                    'Quote Completed_time', 'Quote Completed_platform',
                    'Payment Completed_time', 'Payment Completed_platform',
                    'Claim Started_time', 'Claim Started_platform',
                    'Claim Accepted_time', 'Claim Accepted_platform',
                    'Policy Cancelled_time', 'Policy Cancelled_platform',
                    'Quote Incomplete', 'Quote Incomplete_platform',
                    'Claim Denied_time', 'Claim Denied_platform',
                    'id', 'reason', 'amount'
                    ]
    for customer, group in _groups:
        row = {k: None for k in _all_columns}
        row["id"] = customer
        _detail_dict = None
        for idx in group.index:
            row[group.loc[idx, "status"] + "_time"] = group.loc[idx, "time"]
            row[group.loc[idx, "status"] + "_platform"] = group.loc[idx, "browser"]
            if _has_detail[_all_statuses.index(group.loc[idx, "status"])]:
                row[_has_detail[_all_statuses.index(group.loc[idx, "status"])][0]] = \
                    group.loc[idx, _has_detail[_all_statuses.index(group.loc[idx, "status"])][0]]
            if isinstance(group.loc[idx, "finalDict"], dict):
                _detail_dict = group.loc[idx, "finalDict"]

        row.update(get_dict_detail(_detail_dict, keys_ordered))
        _customer_rows.append(row)

    _all_columns.extend(keys_ordered)
    return pd.DataFrame(_customer_rows, columns=_all_columns)


def transform_data(filename):
    test_data = load_data(filename)
    converted = convert_data(test_data)
    return converted


def load_raw_data():
    test_converted = transform_data('data/test.csv')
    train_data = load_data('data/train.csv')
    train_converted = convert_data(train_data)

    train_converted.to_csv('data/train_converted.csv')
    test_converted.to_csv('data/test_converted.csv')

    # train_num_customers = train_converted.shape[0]
    # test_num_customers = test_converted.shape[0]
    # train_none = train_converted.isnull().sum()
    # test_none = test_converted.isnull().sum()
    # print("Total customers in trainsing set:", train_num_customers)
    # print("NULL count in training set:", train_none, sep="\n")
    # print("========" * 2 + "following is test" + "========" * 2)
    # print("Total customers in test set:", test_num_customers)
    # print("NULL count in test set:", test_none, sep="\n")


def load_saved_data():
    _train_converted = pd.read_csv('data/train_converted.csv', index_col=0)
    _test_converted = pd.read_csv('data/test_converted.csv', index_col=0)
    _submission = pd.read_csv('data/sample_submission.csv')
    return _train_converted, _test_converted, _submission


def transform_time_features(converted, time_features):
    # converted['claim_start_time'] = pd.to_datetime(converted['Claim Started_time'], unit='s')
    converted['quote_time'] = converted[time_features[1]] - converted[time_features[0]]
    converted['payment_time'] = converted[time_features[2]] - converted[time_features[1]]
    converted['cancelled_time'] = converted[time_features[3]] - converted[time_features[2]]
    converted['claim_time'] = converted[time_features[4]] - converted[time_features[2]]
    converted['accepted_time'] = converted[time_features[5]] - converted[time_features[4]]
    converted['denied_time'] = converted[time_features[6]] - converted[time_features[4]]
    return converted


def transform_categorical_features(converted, types, categorical_features):
    for i in range(8):
        converted['convert' + categorical_features[i]] = converted[categorical_features[i]].map(types[0])
    converted['convert' + categorical_features[8]] = converted[categorical_features[8]].map(types[1])
    converted['convert' + categorical_features[9]] = converted[categorical_features[9]].map(types[2])
    return converted


def dummy_categorical_features(converted, types, categorical_features):
    dum_features = pd.get_dummies(converted[categorical_features])
    converted = pd.concat([converted, dum_features], axis=1)
    return converted


def time_numeric(converted):
    # transform seconds to hour, except the first time.
    converted['payment_time'] = converted['payment_time'].apply(lambda x: x / 3600)
    converted['cancelled_time'] = converted['cancelled_time'].apply(lambda x: x / 3600)
    converted['claim_time'] = converted['claim_time'].apply(lambda x: x / 3600)
    converted['accepted_time'] = converted['accepted_time'].apply(lambda x: x / 3600)
    converted['denied_time'] = converted['denied_time'].apply(lambda x: x / 3600)
    return converted


def get_household_num(converted, other_features):
    def get_len(arr):
        if arr[other_features] is not np.nan:
            return min(len(json.loads(arr[other_features].replace("\'", "\""))), 9)
        return None

    converted['num_household'] = converted.apply(get_len, axis=1)
    converted['num_household'] = converted['num_household'].fillna(0).astype(int)
    return converted


# standard scale for data
def scale_data(df, std_names, replace=False):
    std_label = 'scale_' if replace is not True else ''
    for name in std_names:
        df.loc[:, std_label + name] = StandardScaler().fit_transform(df[name].values.reshape(-1, 1))
        # df.loc[:, std_label + name] = MinMaxScaler().fit_transform(df[name].values.reshape(-1, 1))
        print(std_label + name)
    # print('===' * 4, 'standard scale finished')
    return df


def general_describe(df):
    """
    general describe about the data
    :param df: data frame type
    :return:
    """
    df.head()
    features = df.columns.tolist()
    print('the number of features is ', len(features))
    print('the features are \n', np.array(features))
    print(df.dtypes)


def plot_hist(df, name=''):
    """
    visualize the dataset hist data, if want to dip into the specific data ,use df['feature'].hist()
    :param df:
    :return:
    """
    df.hist(xlabelsize=7, ylabelsize=7, figsize=(12, 10))
    # plt.savefig('./pic/'+name+'hist.png')
    plt.show()


def correlation_map(df, name=''):
    """
    correlation map for total data
    notice: the df can only be number no text. so some unique type of values should be transformed first
    :param df: data frame data
    :return:
    """
    colormap = plt.cm.RdBu
    plt.figure(figsize=(14, 12))
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    sns.heatmap(df.astype(float).corr(), linewidths=0.1, vmax=1.0,
                square=True, cmap=colormap, linecolor='white', annot=True)
    # plt.savefig('./pic/' + name + 'correlation_map.png')
    plt.show()


def plot_data(train_converted, test_converted):
    general_describe(train_converted)
    plot_hist(train_converted)


def over_sample(X, y, sampler="SMOTE"):
    samplers = {
        "RandomOverSampler": RandomOverSampler(),
        "ADASYN":            ADASYN(),
        "SMOTE":             SMOTE(),
        "BorderlineSMOTE":   BorderlineSMOTE(),
        "SVMSMOTE":          SVMSMOTE(),
        "SMOTENC":           SMOTENC(categorical_features=[]),
    }
    sampler = samplers[sampler]

    # to resample simply call fit_resample method of sampler
    X_resampled, y_resampled = sampler.fit_resample(X, y)

    return X_resampled, y_resampled


def add_address(str_list):
    s = ''
    for i in range(len(str_list)):
        s = s + str_list[i]
    return s


def encode_address(train_data):
    # add the street name
    address_list = list(train_data['address'])
    address_list_1 = [x.split(',')[0] for x in address_list]
    address_list_1 = [add_address(x.split()[1:]) for x in address_list_1]
    address_list_2 = [x.split(',')[1] for x in address_list]
    address_list_2 = [add_address(x.split()) for x in address_list_2]

    enc_address_1 = LabelEncoder().fit(address_list_1)
    encoded_address_1 = enc_address_1.transform(address_list_1)
    enc_address_2 = LabelEncoder().fit(address_list_2)
    encoded_address_2 = enc_address_2.transform(address_list_2)

    train_data['address_part_1_encoded'] = encoded_address_1
    train_data['address_part_2_encoded'] = encoded_address_2
    return train_data


def new_trans(data):
    data['root_area'] = data.apply(lambda x: x['square_footage'] ** 0.5, axis=1)
    data['data_time'] = data['Payment Completed_time'].apply(lambda x: datetime.datetime.fromtimestamp(x) if x >0 else datetime.datetime.fromtimestamp(0))
    # dtime = pd.Series([datetime.datetime.fromtimestamp(i) for i in data['Payment Completed_time']])
    data['year'] = 2019 - pd.DatetimeIndex(data['data_time']).year
    # data['mul'] = data['num_household'] * data['number_of_bedrooms']

    data = data.drop(['data_time'], axis=1)

    # data = encode_address(data)
    
    return data


def split_people(data):
    paid_people = data[data['Payment Completed_time'] > 0]
    unpaid_people = data.loc[np.isnan(data['Payment Completed_time']), :]
    claimed_people = paid_people[paid_people['Claim Started_time'] > 0]
    never_claimed_people = paid_people.loc[np.isnan(paid_people['Claim Started_time']), :]
    return paid_people, unpaid_people, claimed_people, never_claimed_people


def transform_converted(_train_converted, _test_converted):
    # define features
    categorical_features = ['Quote Started_platform', 'Quote Completed_platform', 'Payment Completed_platform',
                            'Claim Started_platform', 'Claim Accepted_platform', 'Policy Cancelled_platform',
                            'Quote Incomplete_platform', 'Claim Denied_platform', 'reason', 'gender', 'type']

    time_features = ['Quote Started_time', 'Quote Completed_time', 'Payment Completed_time', 'Policy Cancelled_time',
                     'Claim Started_time', 'Claim Accepted_time', 'Claim Denied_time', 'Quote Incomplete']

    numeric_features = ['age', 'square_footage', "area_per_room", "area_per_person"
                        ]
    str_features = ['name', 'address', 'email']
    label = ['amount']

    platforms = {'phone_call':     int(1),
                 'pc_browser':     int(2),
                 'mobile_browser': int(3),
                 'mobile_app':     int(4)
                 }
    fraud_type = {'fraud': int(1)
                  }
    gender = {
        'male':   int(1),
        'female': int(2)
    }
    house_type = [0, 1]
    types = [platforms, fraud_type, gender]

    _test_converted = get_household_num(_test_converted, 'household')
    _test_converted["area_per_room"] = _test_converted["square_footage"] / _test_converted["number_of_bedrooms"]
    _test_converted["area_per_person"] = _test_converted["square_footage"] / (_test_converted["num_household"] + 1)
    _test_converted = transform_time_features(_test_converted, time_features)
    _test_converted = time_numeric(_test_converted)
    _test_converted = dummy_categorical_features(_test_converted, types, categorical_features)
    # _test_house = pd.get_dummies(_test_converted["num_household"].astype(str), prefix='household').astype(int)
    # _test_converted = pd.concat([_test_converted, _test_house], axis=1)
    _test_converted = scale_data(_test_converted, std_names=numeric_features)
    _test_converted = scale_data(_test_converted,
                                     std_names=['quote_time', 'payment_time', 'cancelled_time', 'claim_time',
                                                'accepted_time', 'denied_time'],
                                     replace=True)

    _train_converted = get_household_num(_train_converted, 'household')
    _train_converted["area_per_room"] = _train_converted["square_footage"] / _train_converted["number_of_bedrooms"]
    _train_converted["area_per_person"] = _train_converted["square_footage"] / (_train_converted["num_household"] + 1)
    _train_converted = transform_time_features(_train_converted, time_features)
    _train_converted = time_numeric(_train_converted)
    _train_converted = dummy_categorical_features(_train_converted, types, categorical_features)
    # _train_house = pd.get_dummies(_train_converted["num_household"].astype(str), prefix='household').astype(int)
    # _train_converted = pd.concat([_train_converted, _train_house], axis=1)
    _train_converted = scale_data(_train_converted, std_names=numeric_features)
    _train_converted = scale_data(_train_converted,
                                      std_names=['quote_time', 'payment_time', 'cancelled_time', 'claim_time',
                                                 'accepted_time', 'denied_time'],
                                      replace=True)

##########
    _train_converted = new_trans(_train_converted)
    _test_converted = new_trans(_test_converted)
############

    _test_converted = _test_converted.drop(categorical_features, axis=1)
    _test_converted = _test_converted.drop(time_features, axis=1)
    _test_converted = _test_converted.drop(str_features, axis=1)
    _test_converted = _test_converted.drop('household', axis=1)
    _test_converted = _test_converted.drop(numeric_features, axis=1)

    _train_converted = _train_converted.drop(categorical_features, axis=1)
    _train_converted = _train_converted.drop(time_features, axis=1)
    _train_converted = _train_converted.drop(str_features, axis=1)
    _train_converted = _train_converted.drop('household', axis=1)
    _train_converted = _train_converted.drop(numeric_features, axis=1)

    _test_converted = _test_converted.drop([
                                            # 'scale_square_footage',
                                            # 'quote_time', 'payment_time',
                                            # 'cancelled_time', 'claim_time',
                                            # 'accepted_time', 'denied_time',
                                            # 'Quote Started_platform_mobile_app',
                                            # 'Quote Started_platform_mobile_browser',
                                            # 'Quote Started_platform_pc_browser',
                                            # 'Quote Started_platform_phone_call',
                                            # 'Quote Completed_platform_mobile_app',
                                            # 'Quote Completed_platform_mobile_browser',
                                            # 'Quote Completed_platform_pc_browser',
                                            # 'Quote Completed_platform_phone_call',
                                            # 'Payment Completed_platform_mobile_app',
                                            # 'Payment Completed_platform_mobile_browser',
                                            # 'Payment Completed_platform_pc_browser',
                                            # 'Claim Started_platform_mobile_app',
                                            # 'Claim Started_platform_mobile_browser',
                                            # 'Claim Started_platform_pc_browser',
                                            # 'Quote Incomplete_platform_mobile_app',
                                            # 'Quote Incomplete_platform_mobile_browser',
                                            # 'Quote Incomplete_platform_pc_browser',
                                            # 'Quote Incomplete_platform_phone_call'
                                            ], axis=1)

    diff = list(set(_train_converted.columns) - set(_test_converted.columns))
    _train_converted = _train_converted.drop(diff, axis=1)

    _train_converted = _train_converted.fillna(0)
    _test_converted = _test_converted.fillna(0)
    return _train_converted, _test_converted


def matrix_correlation(paid_people, unpaid_people, claimed_people, never_claimed_people):
    sample = 8890
    paid = paid_people.sample(n=sample, random_state=1)
    unpaid = unpaid_people.sample(n=sample, random_state=2)
    claimed =claimed_people.sample(n=sample, random_state=3)
    neverclaimed = never_claimed_people.sample(n=sample, random_state=3)
    paid = paid.values
    unpaid = unpaid.values
    claimed = claimed.values
    neverclaimed = neverclaimed.values
    return paid, unpaid, claimed, neverclaimed


if __name__ == "__main__":
    train_data, test_data, submission = load_saved_data()
    # train_data, test_data = transform_converted(train_data, test_data)
    eval_data = train_data[train_data['Payment Completed_time'] > 0]
    eval_convert, t = transform_converted(eval_data, test_data)
    eval_convert = eval_convert.drop(["id", "amount"], axis=1)

    train_paid, test_converted = transform_converted(train_data[train_data['amount'] > 0], test_data)


    # train_unpaid, _ = transform_converted(train_data[np.isnan(train_data['amount'])], test_data)

    x_paid, y_paid = train_paid.drop(["id", "amount"], axis=1), train_paid["amount"]
    x_train_paid, x_test_paid, y_train_paid, y_test_paid = train_test_split(x_paid, y_paid, test_size=0.3, random_state=42)
    train_paid.to_csv('paid.csv')
    test_converted.to_csv('test_converted.csv')

    default_params = {'learning_rate':    0.1,
                      'n_estimators':     90,
                      'max_depth':        4,
                      'min_child_weight': 4,
                      'subsample':        0.8,
                      'colsample_bytree': 0.8,
                      'gamma':            0,
                      'seed':             42,
                      'n_jobs':           6,
                      'eta': 0.05,
                      'num_boost_round': 100
                      }
    xgb = xgboost.XGBRegressor(**default_params)
    xgb.fit(x_train_paid, y_train_paid)
    print(mean_squared_error(y_train_paid, xgb.predict(x_train_paid)) ** 0.5)

    y_pred_paid = xgb.predict(x_test_paid)
    print(mean_squared_error(y_pred_paid, y_test_paid) ** 0.5)
    #
    # eval = xgb.predict(eval_convert)
    #
    # # params = {
    # #     # 'gamma':            [0, 0.1],
    # #     # 'subsample':        [0.6, 0.8, 0.7],
    # #     # 'colsample_bytree': [0.6, 0.8, 1.0],
    # #     # 'num_boost_round': [100, 250, 500],
    # #     # 'eta':             [0.05, 0.1, 0.2],
    # #     # 'learning_rate':   [0.05, 0.1, 0.2],
    # #     # 'max_depth':        [3, 4, 5],
    # #     # 'min_child_weight': [3, 4, 5, 6, 7],
    # #     # 'n_estimators':     [70, 80, 90, 100, 110, 120, 130]
    # # }
    # # gv = GridSearchCV(xgb, params, cv=3, scoring='neg_mean_squared_error', verbose=True, n_jobs=1)
    # # gv.fit(x_train_paid, y_train_paid)
    # # print(gv.best_params_)

    y_pred_test = xgb.predict(test_converted.drop(['id', 'amount'], axis=1))
    my_submission = test_converted.loc[:, ['id', 'amount']]
    my_submission.columns = ['customer_id', 'claim_amount']
    my_submission["claim_amount"] = y_pred_test
    my_submission.to_csv('data/g.csv', index=None)
    print("saved to：", 'data/g.csv')

    # x_all, y_all = train_converted.drop(["id", "amount"], axis=1), train_converted["amount"]
    # test_all = test_converted.drop(["id", "amount"], axis=1)
    # ids = test_converted['id']
    # y_all = np.array([True if i > 0 else False for i in y_all.values])
    #
    # rx_all, ry_all = over_sample(x_all, y_all)
    # x_train_all, x_test_all, y_train_all, y_test_all = train_test_split(rx_all, ry_all, test_size=0.3, random_state=42)
    #
    # rf_params = {
    #     'n_estimators':     500,
    #     'warm_start':       True,
    #     'max_depth':        10,
    #     'min_samples_leaf': 2,
    #     'verbose':          0
    # }
    # rf = RandomForestClassifier(**rf_params)
    # rf.fit(x_train_all, y_train_all)
    # y_pred_all = rf.predict(x_test_all)
    # print(y_pred_all[:20])
    # print(y_test_all[:20])
    # print(accuracy_score(y_test_all, y_pred_all))
    # print(recall_score(y_test_all, y_pred_all))
    # print(f1_score(y_test_all, y_pred_all))
    #
    # lr_params = {
    #
    # }
    # lr = ElasticNet().fit(x_train_paid, y_train_paid)
    # print(mean_squared_error(y_train_paid, lr.predict(x_train_paid)) ** 0.5)
    #
    # # y_pred_unpaid = xgb.predict(train_unpaid.drop(["id", "amount"], axis=1))
    # y_pred_paid = lr.predict(x_test_paid)
    # print(mean_squared_error(y_pred_paid, y_test_paid) ** 0.5)
    # y_pred_test = lr.predict(test_converted.drop(['id', 'amount'], axis=1))
    # my_submission = test_converted.loc[:, ['id', 'amount']]
    # my_submission.columns = ['customer_id', 'claim_amount']
    # my_submission["claim_amount"] = y_pred_test
    # my_submission.to_csv('data/lr_g.csv', index=None)
    # print("saved to：", 'data/lr_g.csv')