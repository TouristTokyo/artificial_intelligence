import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


FILE_NAME = "letter-recognition.data"
data = pd.read_csv(FILE_NAME, sep=',')

TAG_KEY = "T"
OBJECT_RANGE = range(1, 16)

TEST_SIZE = 0.2
RANDOM_STATE = 43

objects = data.iloc[:, OBJECT_RANGE]
tag = data[TAG_KEY]


def find_best_num_neighbors():
    best_k = None
    best_accuracy = 0

    for k in range(1, 20):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, pred)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k
    return best_k


def create_conjugacy_table():
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)

    conf_matrix = confusion_matrix(y_test, pred)
    print("Таблица сопряженности:", np.array(conf_matrix))
    return pred


def calculate_classifier_error():
    error_rate = 1 - accuracy_score(y_test, y_pred)
    print(f"Процент ошибок: {error_rate * 100:.2f}%")


if __name__ == "__main__":
    print(data)

    X_train, X_test, y_train, y_test = train_test_split(
        objects, tag, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print("Наборы данных для обучения классификатора: ")
    print(X_train)
    print(y_train)

    print("Набор тестовых данных: ")
    print(X_test)
    print(y_test)
    print()

    n_neighbors = find_best_num_neighbors()

    print(f"Наилучшее значение k: {n_neighbors}")

    y_pred = create_conjugacy_table()
    calculate_classifier_error()
