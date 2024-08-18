import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('all_data.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

selector = SelectKBest(f_classif, k=100)
X_new = selector.fit_transform(X, y)

scaler = StandardScaler()
X_new_scaled = scaler.fit_transform(X_new)


X_train, X_test, y_train, y_test = train_test_split(X_new_scaled, y, test_size=0.3, random_state=42)

# using best model from activity 2
best_model = SVC(C=0.1, gamma=0.1, kernel='rbf') 

# Train
best_model.fit(X_train, y_train)
train_test_accuracy = best_model.score(X_test, y_test)
print(f'Train-Test Accuracy with feature selection and hyperparameter tuning: {train_test_accuracy}')

# 10-Fold Cross-Validation Accuracy
cv_scores = cross_val_score(best_model, X_new_scaled, y, cv=10)
print(f'Cross-Validation Accuracy with feature selection and hyperparameter tuning: {cv_scores.mean()}')
