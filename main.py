
# importing libraries
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

import seaborn as sns


print("loading data")

# Read the data
#train = pd.read_csv('train-v3.csv', index_col='id')
train = pd.read_csv('train-v3.csv', index_col='id')
test = pd.read_csv('test-v3.csv', index_col='id')

valid = pd.read_csv('valid-v3.csv', index_col='id')


# print first five rows
train.head()


sns.set(rc = {'figure.figsize':(25,8)})
sns.heatmap(train.corr(), annot = True, fmt='.2g',cmap= 'coolwarm')

# column names
train.columns


# columns with null values
train_col_null = train.columns[train.isnull().any()==True].tolist()
# null values in these columns
train[train_col_null].isnull().sum()



####
# columns with null values
valid_col_null = valid.columns[valid.isnull().any()==True].tolist()
# null values in these columns
valid[valid_col_null].isnull().sum()


# print first five rows
test.head()


# column names
test.columns


# columns with null values
test_col_null = test.columns[test.isnull().any()==True].tolist()
# null values in these columns
test[test_col_null].isnull().sum()


# Remove rows with missing target
X = train.dropna(axis=0, subset=['price'])

# separate target from predictors
y = X.price
X.drop(['price'], axis=1, inplace=True)




###
# Remove rows with missing target
X_valid = valid.dropna(axis=0, subset=['price'])

# separate target from predictors
y_valid = X_valid.price
X_valid.drop(['price'], axis=1, inplace=True)



# Break off validation set from training data
'''
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y,
                                                                train_size=0.8,
                                                                test_size=0.2,
                                                                random_state=0)
'''
X_train_full = X
y_train = y

X_valid_full = X_valid
y_valid = y_valid


# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)

low_cardinality_cols = [cname for cname in X_train_full.columns
                        if X_train_full[cname].nunique() < 10 and
                        X_train_full[cname].dtype == "object"]



# Select numeric columns
numeric_cols = [cname for cname in X_train_full.columns
                if X_train_full[cname].dtype in ['int64', 'float64']]


# Keep selected columns only
my_cols = low_cardinality_cols + numeric_cols

X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()

# for test data also
X_test = test[my_cols].copy()


# One-hot encode the data
X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_test = pd.get_dummies(X_test)

X_train, X_valid = X_train.align(X_valid, join='left', axis=1)
X_train, X_test = X_train.align(X_test, join='left', axis=1)


# Define the model
xgb =  XGBRegressor(n_estimators=1800,learning_rate=0.04,max_depth = 7)


print("fitting the model...")

# Fit the model
xgb.fit(X_train, y_train)


# Get predictions
y_pred = xgb.predict(X_valid)


# Calculate MAE
mae = mean_absolute_error(y_pred, y_valid)
print("mae: " , mae)


# prediction
prediction = xgb.predict(X_test)



# Submission file

output = pd.DataFrame({'id': X_test.index,
                       'price': prediction})
output.to_csv('submission_test_OKOK.csv', index=False)
output.head()


print("output csv file: submission_test_OKOK.csv")


