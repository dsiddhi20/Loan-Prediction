import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load datasets
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# Combine for consistent preprocessing
combined = pd.concat(
    [train_df.drop('Loan_Status', axis=1), test_df],
    ignore_index=True
)

# ---- CLEANING (NO WARNINGS) ----
combined['LoanAmount'] = combined['LoanAmount'].fillna(combined['LoanAmount'].median())
combined['Credit_History'] = combined['Credit_History'].fillna(combined['Credit_History'].mode()[0])
combined = combined.ffill()

# ---- ENCODING ----
le = LabelEncoder()
for col in combined.select_dtypes(include='object'):
    combined[col] = le.fit_transform(combined[col])

# Split back
X_train = combined.iloc[:len(train_df)].drop('Loan_ID', axis=1)
X_test = combined.iloc[len(train_df):].drop('Loan_ID', axis=1)

y_train = train_df['Loan_Status'].map({'N': 0, 'Y': 1})

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on test data
test_predictions = model.predict(X_test)

# Save predictions
output = pd.DataFrame({
    'Loan_ID': test_df['Loan_ID'],
    'Loan_Status_Predicted': test_predictions
})

output.to_csv("loan_predictions.csv", index=False)

print("✅ Predictions saved as loan_predictions.csv")
