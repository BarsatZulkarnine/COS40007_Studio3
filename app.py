import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Load the CSV files
#filenames  = ['./ampc/ampc/w1.csv', './ampc/ampc/w2.csv', './ampc/ampc/w3.csv', './ampc/ampc/w4.csv']
#combined_data = pd.concat([pd.read_csv(f) for f in filenames])
#combined_data.to_csv('combined_data.csv', index=False)

# 2. Shuffle the data and save it
#shuffled_data = combined_data.sample(frac=1, random_state=1).reset_index(drop=True)
#shuffled_data.to_csv('all_data.csv', index=False)

# Studio Activity 2: Model Training

# Load the data
data = pd.read_csv('./ampc/ampc/w5.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 1a. Train an SVM model using a 70/30 train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
clf = SVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
train_test_accuracy = accuracy_score(y_test, y_pred)

# 1b. 10-fold cross-validation
cv_scores = cross_val_score(clf, X, y, cv=10)
cv_accuracy = np.mean(cv_scores)

# Studio Activity 3: Hyperparameter Tuning

# 1. Use RBF kernel and GridSearchCV to find optimal hyperparameters
param_grid = {'C': [0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1]}
grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=10)
grid_search.fit(X, y)
best_params = grid_search.best_params_

# 2. Update the SVM model with optimal hyperparameters
clf = SVC(kernel='rbf', C=best_params['C'], gamma=best_params['gamma'])

# Train-test split with hyperparameter tuning
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
train_test_tuned_accuracy = accuracy_score(y_test, y_pred)

# 10-fold cross-validation with hyperparameter tuning
cv_scores_tuned = cross_val_score(clf, X, y, cv=10)
cv_tuned_accuracy = np.mean(cv_scores_tuned)

# Studio Activity 4: Feature Selection

# 1. Select the top 100 features using SelectKBest
k_best = SelectKBest(f_classif, k=100)
X_new = k_best.fit_transform(X, y)

# Train-test split with feature selection and hyperparameter tuning
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3, random_state=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
train_test_fs_accuracy = accuracy_score(y_test, y_pred)

# 10-fold cross-validation with feature selection and hyperparameter tuning
cv_scores_fs = cross_val_score(clf, X_new, y, cv=10)
cv_fs_accuracy = np.mean(cv_scores_fs)

# Studio Activity 5: Dimensionality Reduction

# 1. Apply PCA to reduce dimensions to 10
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X)

# Train-test split with PCA and hyperparameter tuning
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
train_test_pca_accuracy = accuracy_score(y_test, y_pred)

# 10-fold cross-validation with PCA and hyperparameter tuning
cv_scores_pca = cross_val_score(clf, X_pca, y, cv=10)
cv_pca_accuracy = np.mean(cv_scores_pca)

# Studio Activity 6: Prepare Summary Table
summary = pd.DataFrame({
    'SVM model': ['Original features', 'With hyperparameter tuning', 
                  'With feature selection and hyperparameter tuning', 
                  'With PCA and hyperparameter tuning'],
    'Train-test split': [train_test_accuracy, train_test_tuned_accuracy, 
                         train_test_fs_accuracy, train_test_pca_accuracy],
    'Cross validation': [cv_accuracy, cv_tuned_accuracy, 
                         cv_fs_accuracy, cv_pca_accuracy]
})

print(summary)

# Studio Activity 7: Other classifiers

# Train with SGDClassifier
sgd_clf = SGDClassifier(random_state=1)
sgd_clf.fit(X_train, y_train)
sgd_y_pred = sgd_clf.predict(X_test)
sgd_train_test_accuracy = accuracy_score(y_test, sgd_y_pred)
sgd_cv_scores = cross_val_score(sgd_clf, X, y, cv=10)
sgd_cv_accuracy = np.mean(sgd_cv_scores)

# Train with RandomForestClassifier
rf_clf = RandomForestClassifier(random_state=1)
rf_clf.fit(X_train, y_train)
rf_y_pred = rf_clf.predict(X_test)
rf_train_test_accuracy = accuracy_score(y_test, rf_y_pred)
rf_cv_scores = cross_val_score(rf_clf, X, y, cv=10)
rf_cv_accuracy = np.mean(rf_cv_scores)

# Train with MLPClassifier
mlp_clf = MLPClassifier(random_state=1)
mlp_clf.fit(X_train, y_train)
mlp_y_pred = mlp_clf.predict(X_test)
mlp_train_test_accuracy = accuracy_score(y_test, mlp_y_pred)
mlp_cv_scores = cross_val_score(mlp_clf, X, y, cv=10)
mlp_cv_accuracy = np.mean(mlp_cv_scores)

# Prepare Summary Table for Other Classifiers
summary_classifiers = pd.DataFrame({
    'Model': ['SVM', 'SGD', 'RandomForest', 'MLP'],
    'Train-test split': [train_test_tuned_accuracy, sgd_train_test_accuracy, 
                         rf_train_test_accuracy, mlp_train_test_accuracy],
    'Cross validation': [cv_tuned_accuracy, sgd_cv_accuracy, 
                         rf_cv_accuracy, mlp_cv_accuracy]
})

print(summary_classifiers)