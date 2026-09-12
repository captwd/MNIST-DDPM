import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

TRADITIONAL_MODELS = {
    'SVM': lambda: SVC(kernel='rbf', gamma='scale'),
    'KNN': lambda: KNeighborsClassifier(n_neighbors=5),
    'RandomForest': lambda: RandomForestClassifier(n_estimators=100, random_state=42),
    'LogisticRegression': lambda: LogisticRegression(max_iter=100, solver='lbfgs'),
}


def run_traditional_all(x_train, y_train, x_test, y_test, names=None):
    names = names or list(TRADITIONAL_MODELS.keys())
    results = {}
    for name in names:
        model = TRADITIONAL_MODELS[name]()
        start = time.time()
        model.fit(x_train, y_train)
        acc = accuracy_score(y_test, model.predict(x_test))
        elapsed = time.time() - start
        results[name] = {'acc': acc, 'time': elapsed}
        print(f'[Traditional] {name}: acc={acc:.4f}, time={elapsed:.1f}s')
    return results
