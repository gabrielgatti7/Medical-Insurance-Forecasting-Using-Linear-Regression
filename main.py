import pandas as pd
from sklearn import linear_model
from sklearn.calibration import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

# ================= Load Data =================
df = pd.read_csv('insurance.csv')

# ================= Data Preprocessing =================
# df.info()
# le = LabelEncoder()
# df['age'] = le.fit_transform(df['age'])
# print(le.classes_)
# df['sex'] = le.fit_transform(df['sex'])
# print(le.classes_)
# df['children'] = le.fit_transform(df['children'])
# print(le.classes_)
# df['smoker'] = le.fit_transform(df['smoker'])
# print(le.classes_)
# df['region'] = le.fit_transform(df['region'])
# print(le.classes_)

df['age'] = df['age'].astype('int8')
df['children'] = df['children'].astype('int8')

# Convert the "sex" column to numeric values (0 for male, 1 for female)
df['sex'] = df['sex'].map({'male': 0, 'female': 1})

# Convert "smoker" column to numeric values (0 for no, 1 for yes)
df['smoker'] = df['smoker'].map({'no': 0, 'yes': 1})

# Convert the "region" column to one-hot encoding
df = pd.get_dummies(df, columns=['region'])
# df.info()

# ================= Split Data =================
X = df.drop(['charges'], axis=1)
y = df['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ================= Model Training =================
linear_model = linear_model.LinearRegression()
linear_model.fit(X_train, y_train)

y_pred = linear_model.predict(X_test)

# ================= Model Evaluation =================
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')
rmse = root_mean_squared_error(y_test, y_pred)
print(f'Root Mean Squared Error: {rmse}')
r2 = r2_score(y_test, y_pred)
print(f'R-squared: {r2}')