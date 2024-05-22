import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from collections import Counter

data = np.loadtxt('data_imbalance.txt', delimiter=',')
X = data[:, :-1]
y = data[:, -1]

print(f'Class distribution before resampling: {Counter(y)}')

over = SMOTE(sampling_strategy=0.4)
X_resampled, y_resampled = over.fit_resample(X, y)

print(f'Class distribution after resampling: {Counter(y_resampled)}')

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy after balancing: {accuracy}')
