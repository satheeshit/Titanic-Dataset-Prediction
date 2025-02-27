
import pandas as pd
df=pd.read_csv(r"C:/Users/ELCOT/tasks/titanic.csv")
df
x=df.iloc[:,1:6]
y=df.iloc[:,-1]

df.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'], inplace=True)
print(df.isnull().sum())

df['Age'] = df['Age'].fillna(df['Age'].median()) 
df['Embarked'] = df['Embarked'].fillna('N')
print(df['Embarked'].unique())


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df['Sex'])  
df['Embarked'] = label_encoder.fit_transform(df['Embarked'])

x = df.drop(columns=['Survived'])
y = df['Survived']


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=47)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train[['Age', 'Fare']] = scaler.fit_transform(x_train[['Age', 'Fare']])
x_test[['Age', 'Fare']] = scaler.transform(x_test[['Age', 'Fare']])

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
from sklearn.metrics import accuracy_score, classification_report
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

random = model.predict(x_test)

from sklearn.metrics import accuracy_score, classification_report
accuracy = accuracy_score(y_test, random)
report = classification_report(y_test, random)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42, max_depth=5)
model.fit(x_train, y_train)

decision = model.predict(x_test)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
accuracy = accuracy_score(y_test, decision)
report = classification_report(y_test, decision)
cm = confusion_matrix(y_test, decision)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Generate confusion matrix
cm = confusion_matrix(y_test, decision)

# Plot the confusion matrix using a heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Decision Tree - Confusion Matrix')
plt.show()

import numpy as np

# Get feature importance from Random Forest model
feature_importance = model.feature_importances_
features = x.columns

# Plot feature importance
plt.figure(figsize=(8, 5))
sns.barplot(x=feature_importance, y=features)
plt.xlabel('Feature Importance')
plt.ylabel('Feature Names')
plt.title('Feature Importance for Random Forest')
plt.show()

#ROC Curve for Logistic Regression

from sklearn.metrics import roc_curve, auc

# Get probabilities for the positive class
y_prob = model.predict_proba(x_test)[:, 1]

# Compute ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Plot the ROC Curve
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Logistic Regression')
plt.legend()
plt.show()

# Count plot for survival distribution
plt.figure(figsize=(6, 4))
sns.countplot(x=df['Survived'], palette='coolwarm')
plt.xticks([0, 1], ['Not Survived', 'Survived'])
plt.xlabel('Survival')
plt.ylabel('Count')
plt.title('Survival Distribution')
plt.show()

#Age Distribution by Survival using Histogram
plt.figure(figsize=(8, 5))
sns.histplot(df[df['Survived'] == 1]['Age'], bins=30, kde=True, color='green', label='Survived')
sns.histplot(df[df['Survived'] == 0]['Age'], bins=30, kde=True, color='red', label='Not Survived')
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Age Distribution by Survival')
plt.legend()
plt.show()

#Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.show()

#Survival Rate by Passenger Class using Bar plot
plt.figure(figsize=(7, 5))
sns.barplot(x='Pclass', y='Survived', data=df, palette='coolwarm')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')
plt.title('Survival Rate by Passenger Class')
plt.show()

#Survival Rate by Gender using Bar plot
plt.figure(figsize=(6, 4))
sns.barplot(x='Sex', y='Survived', data=df, palette='coolwarm')
plt.xticks([0, 1], ['Female', 'Male'])
plt.xlabel('Gender')
plt.ylabel('Survival Rate')
plt.title('Survival Rate by Gender')
plt.show()

#Survival Rate by Embarked Port using Bar plot 
plt.figure(figsize=(6, 4))
sns.barplot(x='Embarked', y='Survived', data=df, palette='coolwarm')
plt.xticks([0, 1, 2], ['Cherbourg', 'Queenstown', 'Southampton'])
plt.xlabel('Embarked Port')
plt.ylabel('Survival Rate')
plt.title('Survival Rate by Embarked Port')
plt.show()

#model performance comparison
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
}

results = {}

for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(results)

# Plot accuracy comparison
plt.figure(figsize=(7, 4))
sns.barplot(x=list(results.keys()), y=list(results.values()), palette="coolwarm")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.show()



