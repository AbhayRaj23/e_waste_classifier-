import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Generate synthetic dataset
device_types = ['Phone', 'Laptop', 'TV', 'Fridge', 'Washing Machine', 'Tablet', 'Air Conditioner']
battery_types = ['Lithium-ion', 'Lead Acid', 'None']
e_waste_categories = {
    'Phone': 1,
    'Laptop': 1,
    'Tablet': 1,
    'TV': 2,
    'Fridge': 0,
    'Washing Machine': 0,
    'Air Conditioner': 0
}

n_samples = 500
data = {
    'device_type': np.random.choice(device_types, n_samples),
    'usage_years': np.random.randint(1, 10, n_samples),
    'weight_kg': np.round(np.random.uniform(0.2, 80.0, n_samples), 2),
    'battery_type': np.random.choice(battery_types, n_samples),
    'material_metal_ratio': np.round(np.random.uniform(0.1, 0.9, n_samples), 2),
    'repair_count': np.random.randint(0, 5, n_samples)
}

df = pd.DataFrame(data)
df['e_waste_category'] = df['device_type'].map(e_waste_categories)

# Encode
le_device = LabelEncoder()
le_battery = LabelEncoder()
df['device_type_enc'] = le_device.fit_transform(df['device_type'])
df['battery_type_enc'] = le_battery.fit_transform(df['battery_type'])

X = df[['device_type_enc', 'usage_years', 'weight_kg', 'battery_type_enc', 'material_metal_ratio', 'repair_count']]
y = df['e_waste_category']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model and encoders
joblib.dump(model, 'e_waste_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump({'device': le_device, 'battery': le_battery}, 'label_encoders.pkl')

# Optional: print classification report
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
