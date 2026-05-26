import pandas as pd
import numpy as np
import json
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, LassoCV, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, f1_score
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
y_class = df['Spending_Category'].values

feature_names = X.columns.tolist()

# Step 3 — Scaling and Split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Finer Alpha Search for RidgeCV
alphas_fine = np.logspace(-3, 1, 50)

# Step 4-6 — Models comparison (Regression)
models = {
    'Linear Regression': LinearRegression(),
    'Ridge': RidgeCV(alphas=alphas_fine, cv=kf),
    'Lasso': LassoCV(alphas=np.logspace(-3, 3, 100), cv=kf, max_iter=10000)
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

# Step 7 — Feature Importances (Random Forest)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)
importances = rf.feature_importances_
feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

top_features = [f[0] for f in feat_imp[:10]]
top_importances = [float(f[1] * 100) for f in feat_imp[:10]]

# Step 8 — Reduced Model (Feature Selection)
X_reduced = X[top_features]
scaler_red = StandardScaler()
X_reduced_scaled = scaler_red.fit_transform(X_reduced)

ridge_red = RidgeCV(alphas=alphas_fine, cv=kf)
ridge_red.fit(X_reduced_scaled, y) # Full fit for CV

cv_r2_red = cross_val_score(ridge_red, X_reduced_scaled, y, cv=kf, scoring='r2').mean()
cv_rmse_red = np.sqrt(-cross_val_score(ridge_red, X_reduced_scaled, y, cv=kf, scoring='neg_mean_squared_error')).mean()

results['Ridge (Top 10)'] = {
    'CV R2': float(cv_r2_red),
    'CV RMSE': float(cv_rmse_red)
}

# Step 9 — Classification Pipeline
le = LabelEncoder()
y_class_enc = le.fit_transform(y_class)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf = LogisticRegression(class_weight='balanced', max_iter=1000)

cv_f1_macro = cross_val_score(clf, X_scaled, y_class_enc, cv=skf, scoring='f1_macro').mean()
cv_acc = cross_val_score(clf, X_scaled, y_class_enc, cv=skf, scoring='accuracy').mean()

classification_results = {
    'CV F1 Macro': float(cv_f1_macro),
    'CV Accuracy': float(cv_acc)
}

# Alpha sensitivity for Ridge (Dashboard visualization)
alpha_range = np.logspace(-3, 5, 20)
alpha_scores = []
for a in alpha_range:
    r = Ridge(alpha=a)
    score = cross_val_score(r, X_scaled, y, cv=kf, scoring='r2').mean()
    alpha_scores.append(float(score))

# Save to results.json
output_data = {
    'metrics': results,
    'classification': classification_results,
    'predictions': {
        'labels': [f'S{i+1}' for i in range(10)],
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
