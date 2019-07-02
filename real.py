import pandas as pd
import datetime
import json

filename = "data/train.csv"


# customers_file = "data/sample_submission.csv"
# data = pd.read_csv(customers_file, header=0, sep=",", index_col=None)
# customers = list(data["customer_id"])

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
    _data["time"] = _data.apply(lambda x: datetime.datetime.fromtimestamp(x["timestamp"]), axis=1)

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
    # todo: deal with house holds


train_data = load_data(filename)
converted = convert_data()
print(converted.shape)
