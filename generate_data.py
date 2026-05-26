import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

n_rows = 250

# Define columns
data = {
    'Age': np.random.randint(18, 26, n_rows),
    'Gender': np.random.choice(['Male', 'Female'], n_rows),
    'Year_of_Study': np.random.randint(1, 5, n_rows),
    'Course': np.random.choice(['Engineering', 'Business', 'Medicine', 'Arts', 'Science'], n_rows),
    'Student_Background': np.random.choice(['Urban', 'Rural'], n_rows),
    'Monthly_Allowance_KES': np.random.randint(5000, 25000, n_rows),
    'Accommodation': np.random.choice(['On-campus', 'Off-campus'], n_rows),
    'Distance_From_Campus_KM': np.random.uniform(0, 15, n_rows).round(1),
    'Transport_Type': np.random.choice(['Walk', 'Bike', 'Bus', 'Car'], n_rows),
    'Outings_Per_Month': np.random.randint(0, 12, n_rows),
    'Gym_Membership': np.random.choice([0, 1], n_rows),
    'Relationship_Status': np.random.choice([0, 1], n_rows),
    'Gaming_Hours_Per_Week': np.random.randint(0, 25, n_rows),
    'Cafeteria_Visits_Per_Month': np.random.randint(0, 30, n_rows),
    'Ride_Hailing_Trips_Per_Month': np.random.randint(0, 15, n_rows),
    'Online_Shopping_Orders_Per_Month': np.random.randint(0, 10, n_rows),
    'Club_Events_Attended': np.random.randint(0, 10, n_rows),
    'Printing_Frequency': np.random.randint(1, 15, n_rows),
    'Mobile_Data_Usage_GB': np.random.randint(1, 60, n_rows),
    'Meal_Habit': np.random.choice(['Canteen', 'Cooking', 'Takeout'], n_rows),
}

df = pd.DataFrame(data)

# Logic to generate Semester_Spending_KES with specific importance weights
# Base spending
spending = 20000

# Positive correlations
spending += df['Distance_From_Campus_KM'] * 1500
spending += df['Monthly_Allowance_KES'] * 0.8
spending += df['Cafeteria_Visits_Per_Month'] * 300
spending += df['Gaming_Hours_Per_Week'] * 200
spending += df['Club_Events_Attended'] * 500
spending += df['Mobile_Data_Usage_GB'] * 100
spending += df['Outings_Per_Month'] * 800
spending += df['Ride_Hailing_Trips_Per_Month'] * 400
spending += df['Relationship_Status'] * 2000

# Add noise
noise = np.random.normal(0, 3000, n_rows)
df['Semester_Spending_KES'] = (spending + noise).astype(int)

# Categorical mapping for Spending_Category using quantiles for better balance
# Low (bottom 33%), Medium (33-66%), High (top 33%)
df['Spending_Category'] = pd.qcut(
    df['Semester_Spending_KES'],
    q=3,
    labels=['Low', 'Medium', 'High']
)

df.to_csv('student_spending_dataset_extended.csv', index=False)
print(f"Dataset created with {n_rows} records.")
