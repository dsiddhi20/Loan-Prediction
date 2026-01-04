import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# -------- Load data --------
df = pd.read_csv("data/train.csv")

# -------- Clean data --------
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])
df = df.ffill()

# -------- Encode categorical --------
le = LabelEncoder()
for col in df.select_dtypes(include='object'):
    df[col] = le.fit_transform(df[col])

# -------- Features / Target --------
X = df.drop(['Loan_Status', 'Loan_ID'], axis=1)
y = df['Loan_Status']

# -------- Split --------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------- Train model --------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ================= SHAP (FINAL FIX) =================

# Create explainer
explainer = shap.Explainer(model, X_train)

# Compute SHAP values
shap_values = explainer(X_test)

# 👉 SELECT CLASS 1 (Approved loans)
shap_values_class1 = shap_values[:, :, 1]

# -------- Global explanation --------
shap.plots.beeswarm(shap_values_class1)
plt.show()

# -------- Global importance (bar) --------
shap.plots.bar(shap_values_class1)
plt.show()

# -------- Local explanation (first test sample) --------
shap.plots.waterfall(shap_values_class1[0])
plt.show()
