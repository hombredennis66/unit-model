import pandas as pd
import numpy as np
import json
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Step 1 — Load the data
df = pd.read_csv('student_spending_dataset_extended.csv')

# Step 2 — Feature engineering (One-Hot Encoding)
cat_cols = [
    'Gender', 'Course', 'Student_Background',
    'Accommodation', 'Transport_Type', 'Meal_Habit'
]

ohe = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
cat_encoded = ohe.fit_transform(df[cat_cols])
cat_feature_names = ohe.get_feature_names_out(cat_cols)
df_cat = pd.DataFrame(cat_encoded, columns=cat_feature_names)

num_cols = [
    'Monthly_Allowance_KES', 'Club_Events_Attended', 'Cafeteria_Visits_Per_Month',
    'Distance_From_Campus_KM', 'Relationship_Status', 'Age', 'Mobile_Data_Usage_GB',
    'Ride_Hailing_Trips_Per_Month', 'Outings_Per_Month', 'Gaming_Hours_Per_Week',
    'Online_Shopping_Orders_Per_Month', 'Printing_Frequency', 'Year_of_Study',
    'Gym_Membership'
]

X = pd.concat([df[num_cols], df_cat], axis=1)
y = df['Semester_Spending_KES'].values

feature_names = X.columns.tolist()

# Step 3 — Scaling and Split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Step 4-6 — Models comparison
models = {
    'Linear Regression': LinearRegression(),
    'Ridge': RidgeCV(alphas=np.logspace(-3, 5, 100), cv=kf),
    'Lasso': LassoCV(alphas=np.logspace(-3, 5, 100), cv=kf, max_iter=10000)
}

results = {}
predictions = {'Actual': y_test.tolist()}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cv_r2 = cross_val_score(model, X_scaled, y, cv=kf, scoring='r2').mean()
    cv_rmse = np.sqrt(-cross_val_score(model, X_scaled, y, cv=kf, scoring='neg_mean_squared_error')).mean()

    results[name] = {
        'Test R2': r2_score(y_test, y_pred),
        'Test RMSE': float(np.sqrt(mean_squared_error(y_test, y_pred))),
        'CV R2': float(cv_r2),
        'CV RMSE': float(cv_rmse)
    }
    predictions[name] = y_pred.tolist()

    if name == 'Ridge':
        best_alpha_ridge = model.alpha_
    if name == 'Lasso':
        best_alpha_lasso = model.alpha_

# Step 7 — Feature Importances (Random Forest)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)
importances = rf.feature_importances_
feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

top_features = [f[0] for f in feat_imp[:10]]
top_importances = [float(f[1] * 100) for f in feat_imp[:10]]

# Alpha sensitivity for Ridge
alpha_range = np.logspace(-3, 5, 12)
alpha_scores = []
for a in alpha_range:
    r = Ridge(alpha=a)
    score = cross_val_score(r, X_scaled, y, cv=kf, scoring='r2').mean()
    alpha_scores.append(float(score))

# Save to results.json
output_data = {
    'metrics': results,
    'predictions': {
        'labels': [f'S{i+1}' for i in range(10)], # Show first 10
        'actual': [float(x) for x in y_test[:10]],
        'linear': [float(x) for x in predictions['Linear Regression'][:10]],
        'ridge': [float(x) for x in predictions['Ridge'][:10]],
        'lasso': [float(x) for x in predictions['Lasso'][:10]]
    },
    'importances': {
        'labels': top_features,
        'values': top_importances
    },
    'alpha_search': {
        'labels': [f'{a:.3f}' for a in alpha_range],
        'values': alpha_scores,
        'baseline': float(results['Linear Regression']['CV R2'])
    }
}

with open('results.json', 'w') as f:
    json.dump(output_data, f, indent=4)

print("Training complete. Results saved to results.json.")
