import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Step 1 — Load and explore the data
df = pd.read_csv('student_spending_dataset_extended.csv')

print("Shape:", df.shape)
print("\nSpending Category value counts:")
print(df['Spending_Category'].value_counts())

# Step 2 — Feature engineering
cat_cols = [
    'Gender', 'Course', 'Student_Background',
    'Accommodation', 'Transport_Type', 'Meal_Habit'
]

le = LabelEncoder()
for col in cat_cols:
    df[col + '_enc'] = le.fit_transform(df[col])

feature_cols = [
    'Monthly_Allowance_KES', 'Club_Events_Attended', 'Cafeteria_Visits_Per_Month',
    'Distance_From_Campus_KM', 'Relationship_Status', 'Age', 'Mobile_Data_Usage_GB',
    'Ride_Hailing_Trips_Per_Month', 'Outings_Per_Month', 'Gaming_Hours_Per_Week',
    'Online_Shopping_Orders_Per_Month', 'Printing_Frequency', 'Year_of_Study',
    'Gym_Membership', 'Gender_enc', 'Accommodation_enc', 'Meal_Habit_enc',
    'Transport_Type_enc', 'Student_Background_enc'
]

X = df[feature_cols].values
y = df['Semester_Spending_KES'].values

# Step 3 — Train/test split and scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Step 4 — Linear Regression model
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
lr_cv_r2 = cross_val_score(lr, X_scaled, y, cv=kf, scoring='r2').mean()

print("\n--- Linear Regression Results ---")
print(f"Test R²:    {r2_score(y_test, y_pred_lr):.4f}")
print(f"Test RMSE:  {np.sqrt(mean_squared_error(y_test, y_pred_lr)):.0f}")
print(f"CV R²:      {lr_cv_r2:.4f}")

# Step 5 — Ridge Regression model
alphas = np.logspace(-3, 5, 200)
ridge_cv = RidgeCV(alphas=alphas, cv=kf)
ridge_cv.fit(X_train, y_train)
best_alpha = ridge_cv.alpha_

ridge = Ridge(alpha=best_alpha)
ridge.fit(X_train, y_train)
y_pred_rg = ridge.predict(X_test)

rg_cv_r2 = cross_val_score(ridge, X_scaled, y, cv=kf, scoring='r2').mean()

print("\n--- Ridge Regression Results ---")
print(f"Best Alpha: {best_alpha:.4f}")
print(f"Test R²:    {r2_score(y_test, y_pred_rg):.4f}")
print(f"Test RMSE:  {np.sqrt(mean_squared_error(y_test, y_pred_rg)):.0f}")
print(f"CV R²:      {rg_cv_r2:.4f}")

# Step 7 — Feature importances (Random Forest)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(df[feature_cols].values, y)

importances = rf.feature_importances_
feat_imp = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)

print("\n--- Top 10 Feature Importances ---")
for name, imp in feat_imp[:10]:
    print(f"{name:<40} {imp:.4f}  ({imp*100:.1f}%)")

# Export predictions for Step 8 verification
print("\nActual vs Predicted for Test Set:")
for i in range(len(y_test)):
    print(f"Student {i+1}: Actual={y_test[i]:.0f}, LR={y_pred_lr[i]:.0f}, Ridge={y_pred_rg[i]:.0f}")
