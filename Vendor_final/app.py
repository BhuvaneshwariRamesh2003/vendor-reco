import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_excel("VENDOR_FINAL.xlsx")

# Ensure 'Selected' column exists for positives
df['Selected'] = 1

# List of all vendors
vendors = df['Vendor_ID'].unique()

# Create negative samples
negatives = []
for index, row in df.iterrows():
    current_vendor = row['Vendor_ID']
    unselected_vendors = np.random.choice([v for v in vendors if v != current_vendor], size=2, replace=False)
    for v in unselected_vendors:
        new_row = row.copy()
        new_row['Vendor_ID'] = v
        new_row['Selected'] = 0
        negatives.append(new_row)

# Combine positives and negatives
df_neg = pd.DataFrame(negatives)
df_combined = pd.concat([df, df_neg], ignore_index=True)

# Define feature groups
categorical_features = ['Product', 'Currency', 'Region', 'Delivery_Preference', 'Vendor_ID']
ordinal_features = ['Risk_Level', 'ESG_Score']
risk_order = ['Low', 'Medium', 'High']
esg_order = ['Low', 'Medium', 'High']
numerical_features = [
    'Quantity', 'Score', 'Expected_Delivery_Days', 'Disputes',
    'Order_Handled', 'Customer_Satisfaction', 'Suggested_Price'
]

# Preprocessing pipeline
preprocessor = ColumnTransformer(transformers=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ('ordinal', OrdinalEncoder(categories=[risk_order, esg_order]), ordinal_features),
    ('scale', StandardScaler(), numerical_features)
])

# Prepare data
X = df_combined[categorical_features + ordinal_features + numerical_features]
y = df_combined['Selected']
X_encoded = preprocessor.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Base model
rf = RandomForestClassifier(class_weight='balanced', random_state=42)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=20,
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# Train the model
random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_

# Evaluate
y_pred = best_model.predict(X_test)
print("✅ Best Parameters:", random_search.best_params_)
print("📊 Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Recommendation function
def recommend_vendors(new_request, vendors, preprocessor, model, top_n=5):
    vendor_candidates = []
    for vendor in vendors:
        row = new_request.copy()
        row['Vendor_ID'] = vendor
        vendor_candidates.append(row)
    df_candidates = pd.DataFrame(vendor_candidates)
    X_new_encoded = preprocessor.transform(df_candidates)
    probabilities = model.predict_proba(X_new_encoded)[:, 1]
    top_n_indices = np.argsort(probabilities)[-top_n:][::-1]
    top_vendors = df_candidates.iloc[top_n_indices].copy()
    top_vendors['Probability'] = probabilities[top_n_indices]
    return top_vendors[['Vendor_ID', 'Probability']]

# Sample input
new_request = {
    'Product': 'Laptop',
    'Quantity': 50000,
    'Currency': 'INR',
    'Region': 'South',
    'Delivery_Preference': 'Fast',
    'Score': 95,
    'Expected_Delivery_Days': 3,
    'Risk_Level': 'Low',
    'Disputes': 10,
    'ESG_Score': 'High',
    'Order_Handled': 100,
    'Customer_Satisfaction': 95,
    'Suggested_Price': 15000
}

# Recommend top 5 vendors
print("\n🎯 Top Recommended Vendors:")
top_vendors = recommend_vendors(new_request, vendors, preprocessor, best_model, top_n=5)
print(top_vendors)
