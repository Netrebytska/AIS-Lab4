import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score

# Функція для класифікації
def classify_data(classifier_type):
    data = np.loadtxt('data_random_forests.txt', delimiter=',')
    X = data[:, :-1]
    y = data[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    if classifier_type == 'rf':
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        classifier = ExtraTreesClassifier(n_estimators=100, random_state=42)

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')

    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='viridis')
    plt.title(f'Classification using {"Random Forest" if classifier_type == "rf" else "Extremely Random Forest"}')
    plt.show()

# Основний блок
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Random Forest vs Extremely Random Forest Classifier')
    parser.add_argument('classifier_type', choices=['rf', 'erf'], nargs='?', default='rf', help='Type of classifier to use: rf or erf')
    args = parser.parse_args()
    classify_data(args.classifier_type)
