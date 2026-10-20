import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
data = {
    'Invoice_Value': [1500000, 500000, 3000000, 1200000, 4500000, 800000],
    'Payment_Method': [1, 0, 1, 0, 1, 0], # 1: Kredit, 0: Tunai
    'Customer_History_Score': [85, 40, 90, 60, 95, 30],
    'Late_Payment': [0, 1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)
X = df[['Invoice_Value', 'Payment_Method', 'Customer_History_Score']]
y = df['Late_Payment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Akurasi Model: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))