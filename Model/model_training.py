import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

print("🚀 Starting Model Training Process...")

# 1. Dataset ko load karna
# '../' ka matlab hai model folder se ek step bahar aao aur Dataset folder mein jao
try:
    df = pd.read_csv('../Dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')
    print("✅ Dataset loaded successfully!")
except FileNotFoundError:
    print("❌ Error: Dataset file nahi mili. Folder ka naam check karo.")
    exit()

# 2. Data Preprocessing
# Target variable ko 0 aur 1 mein badalna
df['Attrition'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)

# Main columns jo dashboard mein use hue hain
selected_columns = [
    'Age', 'MonthlyIncome', 'NumCompaniesWorked', 
    'YearsAtCompany', 'YearsSinceLastPromotion', 
    'EnvironmentSatisfaction', 'JobInvolvement',
    'OverTime', 'BusinessTravel'
]

X = df[selected_columns]
y = df['Attrition']

# Text wale data ko numbers mein badalna
X_encoded = pd.get_dummies(X, drop_first=True)

# 3. features.pkl ko save karna
feature_cols = X_encoded.columns.tolist()
pickle.dump(feature_cols, open('features.pkl', 'wb'))
print("✅ features.pkl saved successfully!")

# 4. Data ko split karna (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# 5. scaler.pkl ko save karna (Data Balancing)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
pickle.dump(scaler, open('scaler.pkl', 'wb'))
print("✅ scaler.pkl saved successfully!")

# 6. Random Forest Model train karna aur rf_model.pkl save karna
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
pickle.dump(model, open('rf_model.pkl', 'wb'))
print("✅ rf_model.pkl saved successfully!")

print("🎉 Model Training Complete! All 3 files are ready.")