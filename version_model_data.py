
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

data = pd.read_csv('medical_insurance.csv')

model = DecisionTreeRegressor(**{'max_depth': 15, 'min_samples_leaf': 1, 'min_samples_split': 2, 'random_state':22})

X = data.drop('charges', axis=1)
y = data['charges']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first'))
])

variables_numericas = list(data.select_dtypes(include=['int64', 'float64']).columns)[:-1]
variables_categoricas = list(data.select_dtypes(include=['object']).columns)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, variables_numericas),
        ('cat', categorical_transformer, variables_categoricas)
    ])


pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', model)])

pipeline.fit(X, y)

y_pred = pipeline.predict(X)

r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = mse**0.5
mae = mean_absolute_error(y, y_pred)

print(f'R2: {r2}')
print(f'RMSE: {rmse}')
print(f'MAE: {mae}')

joblib.dump(pipeline, 'pipeline.joblib')