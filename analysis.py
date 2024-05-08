import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import mode
import matplotlib.pyplot as plt

class K_Nearest_Neighbors_Classifier():
    def __init__(self, K):
        self.K = K

    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
        self.m, self.n = X_train.shape

    def predict(self, X_test, met):
        self.X_test = X_test
        self.m_test, self.n = X_test.shape
        Y_predict = np.zeros(self.m_test)
        for i in range(self.m_test):
            x = self.X_test[i]
            neighbors = np.zeros(self.K)
            neighbors = self.find_neighbors(x, met)
            Y_predict[i] = mode(neighbors)[0][0]
        return Y_predict

    def find_neighbors(self, x, met):
        euclidean_distances = np.zeros(self.m)
        for i in range(self.m):
            if met == 1:
                d = self.euclidean(x, self.X_train[i])
            elif met == 2:
                d = self.manhattan(x, self.X_train[i])
            else:
                d = self.cosine_similarity(x, self.X_train[i])
            euclidean_distances[i] = d
        inds = euclidean_distances.argsort()
        Y_train_sorted = self.Y_train[inds]
        return Y_train_sorted[:self.K]

    def euclidean(self, x, x_train):
        return np.sqrt(np.sum(np.square(x - x_train)))

    def manhattan(self, x, x_train):
        return np.sum(np.abs(x - x_train))

    def cosine_similarity(self, x, x_train):
        dot_product = np.dot(x, x_train)
        norm_x = np.linalg.norm(x)
        norm_x_train = np.linalg.norm(x_train)
        similarity = dot_product / (norm_x * norm_x_train)
        return similarity


def analyzeFile(filePath):
    # Загрузка данных и разделение их на обучающую и тестовую выборки
    df = pd.read_csv(filePath)
    X = df.iloc[:, :-1].values
    Y = df.iloc[:, -1:].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1 / 3, random_state=0)

    # Проведение экспериментов с разными значениями K и метриками расстояния
    k_v = [3, 4, 5]
    distance_metrics = [1, 2, 3]
    metrics_v = ['euclidean', 'manhattan', 'cosine']
    results = []

    for k in k_v:
        for metric in metrics_v:
            model = K_Nearest_Neighbors_Classifier(K=k)
            model.fit(X_train, Y_train)
            Y_pred = model.predict(X_test, metric)

            correctly_classified = 0
            for i in range(len(Y_pred)):
                if Y_test[i] == Y_pred[i]:
                    correctly_classified += 1

            accuracy = (correctly_classified / len(Y_pred)) * 100
            results.append((k, metric, accuracy))

    results_df = pd.DataFrame(results, columns=["K", "metrics", "accuracy"])
    print(results_df)

    # Вывод результатов классификации
    classification_results = []
    for i in range(len(Y_pred)):
        result = {
            'Features': X_test[i],
            'True Label': Y_test[i],
            'Predicted Label': Y_pred[i]
        }
        classification_results.append(result)

    for result in classification_results:
        print("Признаки:", result['Features'])
        print("Истинная метка:", result['True Label'])
        print("Предсказанная метка:", result['Predicted Label'])
        print()

    # Создание списков для значений K, метрик и точности
    k_values = [result[0] for result in results]
    metrics = [result[1] for result in results]
    accuracy_values = [result[2] for result in results]

    # Создание уникальных меток оси X для каждой комбинации параметров
    x_labels = [f"K={k}, metric={metric}" for k, metric in zip(k_values, metrics)]

    # Создание графика
    plt.figure(figsize=(10, 6))
    plt.bar(x_labels, accuracy_values)
    plt.xlabel('Комбинация параметров')
    plt.ylabel('Точность классификации (%)')
    plt.title('Точность классификации для разных значений K и метрик расстояния')
    plt.xticks(rotation=45, ha='right')

    # Отображение значений точности над столбцами графика
    for i, accuracy in enumerate(accuracy_values):
        plt.text(i, accuracy + 1, f'{accuracy:.2f}', ha='center')

    plt.show()