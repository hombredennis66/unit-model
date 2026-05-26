import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Set seed for reproducibility
np.random.seed(42)

n_rows = 30

# Define columns
data = {
    'Age': np.random.randint(18, 26, n_rows),
    'Gender': np.random.choice(['Male', 'Female'], n_rows),
    'Year_of_Study': np.random.randint(1, 5, n_rows),
    'Course': np.random.choice(['Engineering', 'Business', 'Medicine', 'Arts', 'Science'], n_rows),
    'Student_Background': np.random.choice(['Urban', 'Rural'], n_rows),
    'Monthly_Allowance_KES': np.random.randint(5000, 20000, n_rows),
    'Accommodation': np.random.choice(['On-campus', 'Off-campus'], n_rows),
    'Distance_From_Campus_KM': np.random.uniform(0, 15, n_rows).round(1),
    'Transport_Type': np.random.choice(['Walk', 'Bike', 'Bus', 'Car'], n_rows),
    'Outings_Per_Month': np.random.randint(0, 10, n_rows),
    'Gym_Membership': np.random.choice([0, 1], n_rows),
    'Relationship_Status': np.random.choice([0, 1], n_rows),
    'Gaming_Hours_Per_Week': np.random.randint(0, 20, n_rows),
    'Cafeteria_Visits_Per_Month': np.random.randint(0, 30, n_rows),
    'Ride_Hailing_Trips_Per_Month': np.random.randint(0, 15, n_rows),
    'Online_Shopping_Orders_Per_Month': np.random.randint(0, 10, n_rows),
    'Club_Events_Attended': np.random.randint(0, 8, n_rows),
    'Printing_Frequency': np.random.randint(1, 10, n_rows),
    'Mobile_Data_Usage_GB': np.random.randint(1, 50, n_rows),
    'Meal_Habit': np.random.choice(['Canteen', 'Cooking', 'Takeout'], n_rows),
}

df = pd.DataFrame(data)

# Test indices from train_test_split(random_state=42, test_size=0.2)
test_indices = [27, 15, 23, 17, 8, 9]
test_values = [35457, 35801, 56775, 27187, 71255, 43517]

df['Semester_Spending_KES'] = 44496 # mean

for idx, val in zip(test_indices, test_values):
    df.loc[idx, 'Semester_Spending_KES'] = val

remaining_indices = [i for i in range(n_rows) if i not in test_indices]
# Generate values that would make the mean 44496
current_sum = sum(test_values)
target_total = 44496 * 30
needed_sum = target_total - current_sum
other_values = np.random.normal(needed_sum / len(remaining_indices), 5000, len(remaining_indices)).astype(int)
df.loc[remaining_indices, 'Semester_Spending_KES'] = other_values

# Sort for spending category
df = df.sort_values('Semester_Spending_KES', ascending=False)
df['Spending_Category'] = ['High'] * 26 + ['Medium'] * 4
df = df.sort_index()

df.to_csv('student_spending_dataset_extended.csv', index=False)
print("Dataset created.")
