from PyQt5.QtCore import QObject, QUrl, pyqtSignal, pyqtSlot



class FileOperations(QObject):
    def __init__(self):
        super().__init__()

    @staticmethod
    def loadFile():
        # Добавьте код для загрузки файла
        print("File loaded")

    @staticmethod
    def analyzeData():
        # Добавьте код для анализа данных из файла
        print("Data analyzed")