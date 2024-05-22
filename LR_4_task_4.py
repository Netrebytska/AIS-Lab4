import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier

X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

model = AdaBoostClassifier(n_estimators=50, random_state=42)
model.fit(X, y)

importance = model.feature_importances_

plt.bar(range(X.shape[1]), importance)
plt.xlabel('Ознаки')
plt.ylabel('Важливість')
plt.title('Відносна важливість ознак за допомогою AdaBoost')
plt.show()
