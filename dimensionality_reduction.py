import pandas as pd
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score

data = pd.read_csv('all_data.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

pca = PCA(n_components=10)
X_pca = pca.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=1)

# Train 
best_clf = svm.SVC(C=1, gamma=0.1, kernel='rbf')  # Use the best params from hyperparameter_tuning
best_clf.fit(X_train, y_train)
print(f'Train-Test Accuracy with PCA: {best_clf.score(X_test, y_test)}')

# 10-fold cross-validation
scores = cross_val_score(best_clf, X_pca, y, cv=10, n_jobs=-1)
print(f'Cross-Validation Accuracy with PCA: {scores.mean()}')
