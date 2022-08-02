import numpy as np
import pandas as pd

import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

with open("params.yaml", "rb") as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

X = np.loadtxt("X.csv")
y = np.loadtxt("y.csv")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=params['train']['test_size'],
    random_state=params['train']['random_state']
)

lr = LogisticRegression(C=params['train']['C'])
lr.fit(X_train.reshape(-1, 1), y_train)


preds = lr.predict(X_test.reshape(-1, 1))
f1 = f1_score(y_test, preds)
print(f"F1 score: {f1:.4f}")