import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score

data = pd.read_csv('all_data.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

#SGDClassifier
sgd_clf = SGDClassifier(random_state=1)
sgd_clf.fit(X_train, y_train)
sgd_train_acc = sgd_clf.score(X_test, y_test)
sgd_cv_acc = cross_val_score(sgd_clf, X, y, cv=10).mean()

#RandomForestClassifier
rf_clf = RandomForestClassifier(random_state=1)
rf_clf.fit(X_train, y_train)
rf_train_acc = rf_clf.score(X_test, y_test)
rf_cv_acc = cross_val_score(rf_clf, X, y, cv=10).mean()

#MLPClassifier
mlp_clf = MLPClassifier(random_state=1)
mlp_clf.fit(X_train, y_train)
mlp_train_acc = mlp_clf.score(X_test, y_test)
mlp_cv_acc = cross_val_score(mlp_clf, X, y, cv=10).mean()


print(f'SGDClassifier: Train-Test Accuracy: {sgd_train_acc}, Cross-Validation Accuracy: {sgd_cv_acc}')
print(f'RandomForestClassifier: Train-Test Accuracy: {rf_train_acc}, Cross-Validation Accuracy: {rf_cv_acc}')
print(f'MLPClassifier: Train-Test Accuracy: {mlp_train_acc}, Cross-Validation Accuracy: {mlp_cv_acc}')
