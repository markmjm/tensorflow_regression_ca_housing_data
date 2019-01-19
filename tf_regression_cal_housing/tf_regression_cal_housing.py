import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
#
# import data
housing_df = pd.read_csv('cal_housing_clean.csv')
print(housing_df.head())
y_val = housing_df['medianHouseValue']
x_data = housing_df.drop('medianHouseValue', axis=1)
cols = x_data.columns
#
# Split data 70/30
X_train, X_test, y_train, y_test = train_test_split(x_data, y_val, test_size=0.30, random_state=101)
#
# Scale the data
scalar = MinMaxScaler()
rescaled_x_train = scalar.fit(X_train)
X_train = pd.DataFrame(data=scalar.transform(X_train), columns=X_train.columns, index=X_train.index)
X_test = pd.DataFrame(data=scalar.transform(X_test), columns=X_test.columns, index=X_test.index)
print(X_train.shape)
print(y_train.shape)
#
# FEATURE COLUMNS
# Numeric
age = tf.feature_column.numeric_column('housingMedianAge')
rooms = tf.feature_column.numeric_column('totalRooms')
bedrooms = tf.feature_column.numeric_column('totalBedrooms')
pop = tf.feature_column.numeric_column('population')
households = tf.feature_column.numeric_column('households')
income = tf.feature_column.numeric_column('medianIncome')
feat_cols = [age, rooms, bedrooms, pop, households, income]
#
# CREATE/TRAIN THE MODEL
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train, batch_size=128, num_epochs=1000, shuffle=True)
model = tf.estimator.DNNRegressor(feature_columns=feat_cols,
                                  hidden_units=[6, 6, 6, 6])  # 4 hidden layers, 6,6,6,6 nodes each
model.train(input_fn=input_func, steps=10000)
#
#
# PREDICT USING THE X_train... I want to plot predicted y_train to original y_train in a plot
train_input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, batch_size=128, num_epochs=10, shuffle=False)
model_gen_train = model.predict(train_input_func)
print('\n')
train_preds = list(model_gen_train)
print(f'PREDICT USING THE MODEL Results: {train_preds}')
train_preds_list = []
for pred in train_preds:
    train_preds_list.append(pred['predictions'])
y_train_pred = train_preds_list[-len(y_train):]
plt.scatter(x=y_train_pred, y=y_train)
plt.xlabel('y_train_pred')
plt.ylabel('y_train')
plt.title('True Y_TRAIN (from input DS) vs. Predicted Y_TRAIN (From Model')
plt.show()
df_training_ys = pd.DataFrame()
df_training_ys['Y_TRAIN_PRED'] = [x[0] for x in y_train_pred]
df_training_ys['Y_TRAIN'] = y_train.tolist()
df_training_ys.columns =['Y_TRAIN_PRED','Y_TRAIN']
df_training_ys['Y_TRAIN_DELTA'] = df_training_ys['Y_TRAIN_PRED'] - df_training_ys['Y_TRAIN']
df_training_ys.hist(column='Y_TRAIN_DELTA', bins=25 )
plt.show()
#
# Get metrics using sklearn
print('\n')
print(f'TRAIN mean_squared_error: {mean_squared_error(y_train, train_preds_list[-len(y_train):]) ** 0.5}')
#
#
# PREDICT USING THE X_test ... I want to plot predicted y_test to original y_test in a plot
pred_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test, batch_size=128, num_epochs=10, shuffle=False)
model_gen_test = model.predict(pred_input_func)
print('\n')
test_preds = list(model_gen_test)
# print(f'PREDICT USING THE MODEL Results: {test_preds}')
test_preds_list = []
for pred in test_preds:
    test_preds_list.append(pred['predictions'])
y_test_pred = test_preds_list[-len(y_test):]
plt.scatter(x=y_test_pred, y=y_test)
plt.xlabel('y_test_pred')
plt.ylabel('y_test')
plt.title('True Y_TEST (from input DS) vs. Predicted Y_TEST (From Model')
plt.show()
df_test_ys = pd.DataFrame()
df_test_ys['Y_TEST_PRED'] = [x[0] for x in y_test_pred]
df_test_ys['Y_TEST'] = y_test.tolist()
df_test_ys.columns =['Y_TEST_PRED','Y_TEST']
df_test_ys['Y_TEST_DELTA'] = df_test_ys['Y_TEST_PRED'] - df_test_ys['Y_TEST']
df_test_ys.hist(column='Y_TEST_DELTA', bins=25)
plt.show()
# Get metrics using sklearn
print('\n')
print(f'TEST mean_squared_error: {mean_squared_error(y_test, test_preds_list[-len(y_test):]) ** 0.5}')