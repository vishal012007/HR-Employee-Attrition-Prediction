import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

print("🚀 Starting Model Training Process...")

# 1. Load the dataset
# '../' navigates one directory up from the model folder into the Dataset folder
try:
    df = pd.read_csv('../Dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')
    print("✅ Dataset loaded successfully!")
except FileNotFoundError:
    print("❌ Error: Dataset file not found. Please check the directory path and folder name.")
    exit()

# 2. Data Preprocessing
# Convert the target variable to binary (1 for 'Yes', 0 for 'No')
df['Attrition'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)

# Main columns used in the Streamlit dashboard
selected_columns = [
    'Age', 'MonthlyIncome', 'NumCompaniesWorked', 
    'YearsAtCompany', 'YearsSinceLastPromotion', 
    'EnvironmentSatisfaction', 'JobInvolvement',
    'OverTime', 'BusinessTravel'
]

X = df[selected_columns]
y = df['Attrition']

# Convert categorical text data into numerical values using One-Hot Encoding
X_encoded = pd.get_dummies(X, drop_first=True)

# 3. Save the features.pkl file
feature_cols = X_encoded.columns.tolist()
pickle.dump(feature_cols, open('features.pkl', 'wb'))
print("✅ features.pkl saved successfully!")

# 4. Split the data (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# 5. Scale the features and save the scaler.pkl file
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
pickle.dump(scaler, open('scaler.pkl', 'wb'))
print("✅ scaler.pkl saved successfully!")

# 6. Train the Random Forest Model and save the rf_model.pkl file
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
pickle.dump(model, open('rf_model.pkl', 'wb'))
print("✅ rf_model.pkl saved successfully!")

print("🎉 Model Training Complete! All 3 files are ready.")