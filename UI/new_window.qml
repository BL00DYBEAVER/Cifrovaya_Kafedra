import QtQuick 2.15
import QtQuick.Controls 2.15

ApplicationWindow {
    visible: true
    width: 400
    height: 300
    title: "New Window"

    Column {
        anchors.centerIn: parent
        spacing: 10

        Label {
            text: "Анализ"
        }

        Button {
            text: "Выбрать файл"
            onClicked: {
                // Добавьте код для обработки выбора файла
            }
        }

        Button {
            text: "Проанализировать"
            onClicked: {
                // Добавьте код для анализа данных
            }
        }
    }
}