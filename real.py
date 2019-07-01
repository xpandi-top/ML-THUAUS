import pandas as pd
import datetime
import json


# customers_file = "data/sample_submission.csv"
# data = pd.read_csv(customers_file, header=0, sep=",", index_col=None)
# customers = list(data["customer_id"])

def json_load(arr):
    if arr["double"]:
        try:
            return json.loads(arr["double"])
        except json.decoder.JSONDecodeError:
            return arr["double"]
    else:
        return "None"


def load_data(file):
    _data = pd.read_csv(file, header=0, index_col=None)
    _data["time"] = _data.apply(lambda x: datetime.datetime.fromtimestamp(x["timestamp"]), axis=1)

    _split_tmp = _data["message"].str.split(" - ", 2, expand=True)
    _split_tmp.columns = ["id", "browser", "info"]

    _data = pd.concat([_data, _split_tmp], axis=1)
    _data = _data.drop("message", axis=1)

    _split_tmp = _data["info"].str.split(":", 1, expand=True)
    _split_tmp.columns = ["status_raw", "info2"]

    _data = pd.concat([_data, _split_tmp], axis=1)
    _data["status"] = _data["status_raw"].str.rstrip("for customer")
    _data = _data.drop(["info", "status_raw"], axis=1)

    _split_tmp = _data["info2"].str.split(" ", 5, expand=True)
    _split_tmp.columns = ["nothing", "customer", "all-with", "all-json", "payload", "json"]
    _split_tmp = _split_tmp.drop(["nothing", "all-with", "all-json", "payload"], axis=1)

    _data = pd.concat([_data, _split_tmp], axis=1)
    _data["double"] = _data["json"].str.replace("\'", "\"")
    _data = _data.drop(["json", "info2", "timestamp"], axis=1)

    _data["finalDict"] = _data.apply(json_load, axis=1)
    _data = _data.drop("double", axis=1)
    return _data


def get_dict_detail(customer_dict, keys_ordered, home_ordered):
    result = []
    for key in keys_ordered:
        result.append(customer_dict.get(key, None))
        if key == "home" and isinstance(customer_dict.get("home", None), dict):
            for h in home_ordered:
                result.append(customer_dict["home"].get(h, None))
        elif key == "home" :
            result.append([None] * 4)
    return result


def convert_data():
    _groups = train_data.groupby("customer")
    _customer_rows = []
    _all_statuses = ['Quote Started', 'Quote Completed', 'Payment Completed', 'Claim Started', 'Claim Accepted',
                     'Policy Cancelled', 'Quote Incompl', 'Claim Denied']
    keys_ordered = ['gender', 'name', 'household', 'age', 'address', 'email', 'home']
    home_ordered = ['square_footage', 'number_of_floors', 'type', 'number_of_bedrooms']
    for customer, group in _groups:
        row = [0] * (2 * len(_all_statuses)) + [customer]
        row.append(0)
        row.append(0)
        for idx in group.index:
            row[_all_statuses.index(group.loc[idx, "status"]) * 2] = group.loc[idx, "time"]
            row[_all_statuses.index(group.loc[idx, "status"]) * 2 + 1] = group.loc[idx, "browser"]
            if isinstance(group.loc[idx, "finalDict"], dict):
                row[-1] = group.loc[idx, "finalDict"]
            elif group.loc[idx, "finalDict"] == "fraud":
                row[-2] = 1
            elif not group.loc[idx, "finalDict"].startswith("None") or group.loc[idx, "finalDict"] is None:
                print(group.loc[idx, "finalDict"])
        if row[-1]:
            row.extend(get_dict_detail(row[-1], keys_ordered, home_ordered))
        else:
            row.extend([None] * 11)
        _customer_rows.append(row)

    _all_columns = ['Quote Started_time', 'Quote Started_platform',
                    'Quote Completed_time', 'Quote Completed_platform',
                    'Payment Completed_time', 'Payment Completed_platform',
                    'Claim Started_time', 'Claim Started_platform',
                    'Claim Accepted_time', 'Claim Accepted_platform',
                    'Policy Cancelled_time', 'Policy Cancelled_platform',
                    'Quote Incompl', 'Quote Incompl_platform',
                    'Claim Denied_time', 'Claim Denied_platform',
                    'id', 'isFraud', 'detail'
                    ]
    _all_columns.extend(keys_ordered)
    _all_columns.extend(home_ordered)
    return pd.DataFrame(_customer_rows, columns=_all_columns)
    # todo: deal with house holds


train_data = load_data("data/small_train.csv")
converted = convert_data()
print(converted.shape)
