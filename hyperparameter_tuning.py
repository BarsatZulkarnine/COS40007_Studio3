import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


data = pd.read_csv('all_data.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [0.1, 1, 10, 100], 'kernel': ['rbf']}

# GridSearchCV
grid_search = GridSearchCV(SVC(), param_grid, cv=10, verbose=3)
grid_search.fit(X_train, y_train)

# Best parameters and model
print("Best parameters found: ", grid_search.best_params_)
best_model = grid_search.best_estimator_

# Evaluate the model
y_pred = best_model.predict(X_test)
print(f'Train-Test Accuracy with hyperparameter tuning: {accuracy_score(y_test, y_pred)}')
