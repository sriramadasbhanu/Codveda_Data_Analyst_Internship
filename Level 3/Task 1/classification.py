import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data = pd.read_csv("Churn Prdiction Data/churn-bigml-80.csv")
print(data.head())
print(data.info())
print(data.columns)
print(data.isnull().sum())



# Convert text columns into numbers
encoder = LabelEncoder()

data["State"] = encoder.fit_transform(data["State"])
data["International plan"] = encoder.fit_transform(data["International plan"])
data["Voice mail plan"] = encoder.fit_transform(data["Voice mail plan"])

# Features (X)
X = data.drop("Churn", axis=1)

# Target (y)
y = data["Churn"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = DecisionTreeClassifier(random_state=42)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Results
print("Accuracy:", accuracy_score(y_test, y_pred))
print()
print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))