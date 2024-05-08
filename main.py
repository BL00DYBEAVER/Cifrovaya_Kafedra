import sys
from PyQt5.QtCore import QObject, QUrl, pyqtSignal, pyqtSlot,  QCoreApplication
from PyQt5.QtWidgets import QApplication
from PyQt5.QtQml import QQmlApplicationEngine, qmlRegisterType


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import mode
import matplotlib.pyplot as plt
from tkinter.messagebox import showerror, showwarning, showinfo

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



class MyPythonClass(QObject):
    def __init__(self):
        super().__init__()
        self.address1 = None
        self.address2 = None

    @pyqtSlot(str)
    def loadFile1(self, fileUrlTrain):
        print("----")
        self.address1 = fileUrlTrain.replace("file:///", "")
        print(type(self.address1), type(self.address2))
        print("----")

    @pyqtSlot(str)
    def loadFile2(self, fileUrlTest):
        print("----")
        self.address2 = fileUrlTest.replace("file:///", "")
        print(type(self.address1), type(self.address2))
        print("----")
    @pyqtSlot()
    def analyzeFile(self):

        #print("--analyzeFile--")
       #print("address1 ",self.address1)
       #print("address2 ",self.address2)
        if ((self.address1 == None)):
                showerror(title="Ошибка", message="Файл 1 не выбран")
        elif ((self.address2 == None)):
                showerror(title="Ошибка", message="Файл 2 не выбран")
        else:
            df = pd.read_csv(self.address1)
            X = df.iloc[:, :-1].values
            Y = df.iloc[:, -1:].values
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1 / 3, random_state=0)

            df2 = pd.read_csv(self.address2)
            X_test2 = df2.iloc[:, :-1].values

            # Проведение экспериментов с разными значениями K и метриками расстояния
            k_v = [3, 4, 5]
            metrics_v = [1, 2, 3]

            results = []
            for k in k_v:
                for metric in metrics_v:
                    model = K_Nearest_Neighbors_Classifier(K=k)
                    model.fit(X_train, Y_train)
                    Y_pred2 = model.predict(X_test2, metric)
                    results.append((k, metric, Y_pred2))

            results_df = pd.DataFrame(results, columns=["K", "metrics", "predictions"])
            print(results_df)

            # Вывод результатов классификации
            classification_results = []
            for i in range(len(Y_pred2)):
                result = {
                    'Features': X_test2[i],
                    'Predicted Label': Y_pred2[i]
                }
                classification_results.append(result)

            for result in classification_results:
                print("Признаки:", result['Features'])
                print("Предсказанная метка:", result['Predicted Label'])
                print()

            # Добавление диаграммы рассеяния
            plt.figure(figsize=(8, 6))
            for result in classification_results:
                features = result['Features']
                predicted_label = result['Predicted Label']
                color = 'green' if 0 == predicted_label else 'red'
                marker = 'o' if 0 == predicted_label else 'x'
                plt.scatter(features[0], features[1], color=color, marker=marker)

            # Создание отдельных точек для каждого класса в легенде
            class_0 = plt.scatter([], [], color='green', marker='o')
            class_1 = plt.scatter([], [], color='red', marker='x')
            plt.legend([class_0, class_1], ['0', '1'])

            plt.xlabel('Признак 1')
            plt.ylabel('Признак 2')
            plt.title('Результаты классификации')
            plt.show()

class MainWindow(QObject):
    switchPageSignal = pyqtSignal(str)

    def __init__(self):
        super().__init__()

    @pyqtSlot(str)
    def switchPage(self, page):
        self.switchPageSignal.emit(page)


if __name__ == "__main__":
    # Установка идентификаторов приложения
    QCoreApplication.setOrganizationName("YourOrganizationName")
    QCoreApplication.setOrganizationDomain("YourOrganizationDomain")

    app = QApplication(sys.argv)
    engine = QQmlApplicationEngine()
    # engine.rootContext().setContextProperty("analysis", analysis)
    my_object = MyPythonClass()
    engine.rootContext().setContextProperty("myObject", my_object)


    main_window = MainWindow()
    engine.rootContext().setContextProperty("mainWindow", main_window)
    engine.load(QUrl.fromLocalFile("main.qml"))

    main_window.switchPageSignal.connect(lambda page: engine.rootObjects()[0].setProperty("currentPage", page))

    sys.exit(app.exec_())