from realRun import *
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.layers.advanced_activations import LeakyReLU

train_data, test_data, submission = load_saved_data()
train_paid, test_converted = transform_converted(train_data[train_data['amount'] > 0], test_data)
x_paid, y_paid = train_paid.drop(["id", "amount"], axis=1), train_paid["amount"]
x_train_paid, x_test_paid, y_train_paid, y_test_paid = train_test_split(x_paid, y_paid, test_size=0.3, random_state=42)

model = Sequential()
model.add(Dense(128, input_dim=x_train_paid.shape[1]))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('relu'))
model.compile(loss='mean_squared_error', optimizer="adam", metrics=["mean_squared_error"])

model.fit(x_train_paid, y_train_paid, epochs=40, shuffle=True, verbose=1)

print(mean_squared_error(y_train_paid, model.predict(x_train_paid)) ** 0.5)
y_pred_paid = model.predict(x_test_paid)
print(mean_squared_error(y_pred_paid, y_test_paid) ** 0.5)

# y_pred_test = model.predict(test_converted.drop(['id', 'amount'], axis=1))
# my_submission = test_converted.loc[:, ['id', 'amount']]
# my_submission.columns = ['customer_id', 'claim_amount']
# my_submission["claim_amount"] = y_pred_test
# my_submission.to_csv('data/nn.csv', index=None)
