import pandas as pd
import datetime
import json

# customers_file = "data/sample_submission.csv"
# data = pd.read_csv(customers_file, header=0, sep=",", index_col=None)
# customers = list(data["customer_id"])

train_file = "data/small_train.csv"
train_data = pd.read_csv(train_file, header=0, index_col=None)
train_data["time"] = train_data.apply(lambda x: datetime.datetime.fromtimestamp(x["timestamp"]), axis=1)

split_tmp = train_data["message"].str.split(" - ", 2, expand=True)
split_tmp.columns = ["id", "browser", "info"]

train_data = pd.concat([train_data, split_tmp], axis=1)
train_data = train_data.drop("message", axis=1)

split_tmp = train_data["info"].str.split(":", 1, expand=True)
split_tmp.columns = ["status_raw", "info2"]

train_data = pd.concat([train_data, split_tmp], axis=1)
train_data["status"] = train_data["status_raw"].str.rstrip("for customer")
train_data = train_data.drop(["info", "status_raw"], axis=1)

split_tmp = train_data["info2"].str.split(" ", 5, expand=True)
split_tmp.columns = ["nothing", "customer", "all-with", "all-json", "payload", "json"]
split_tmp = split_tmp.drop(["nothing", "all-with", "all-json", "payload"], axis=1)

train_data = pd.concat([train_data, split_tmp], axis=1)
train_data["double"] = train_data["json"].str.replace("\'", "\"")
train_data = train_data.drop(["json", "info2", "timestamp"], axis=1)


def json_load(arr):
    if arr["double"]:
        try:
            return json.loads(arr["double"])
        except json.decoder.JSONDecodeError:
            return arr["double"]
    else:
        return "None"


train_data["finalDict"] = train_data.apply(json_load, axis=1)
train_data = train_data.drop("double", axis=1)

print(train_data[:5])
