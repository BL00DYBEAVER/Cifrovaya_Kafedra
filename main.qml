import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Dialogs 1.3
import QtQuick.Layouts 1.12


ApplicationWindow {
    visible: true
    width: 800
    height: 600
    title: "Main Window"

    property string currentPage: "home"

    function getPageComponent(page) {
        switch (page) {
            case "home":
                return homeComponent;
            case "page1":
                return page1Component;
        }
    }

    Row {
    id: menuRow
    anchors.top: parent.top

    Button {
        text: "Home"
        onClicked: mainWindow.switchPage("home")
    }

    Button {
        text: "Page 1"
        onClicked: mainWindow.switchPage("page1")
    }

    }

     Item {
        id: contentArea
        anchors.top: menuRow.bottom
        anchors.bottom: parent.bottom
        width: parent.width

        Loader {
            id: pageLoader
            sourceComponent: getPageComponent(currentPage)
            anchors.fill: parent
        }
    }


Component {
    id: homeComponent
    Item {
        Button {
            text: "Load File"
            onClicked: {
                file_dialog.open() // Открыть собственное диалоговое окно при нажатии на кнопку
                //myObject.loadFile1(file_dialog.fileUrl)
            }
            x: 20 // Отступ слева
            y: 100
        }

        Button {
            text: "Load File2"
            onClicked: {
                file_dialog2.open() // Открыть собственное диалоговое окно при нажатии на кнопку
                //myObject.loadFile2(file_dialog.fileUrl)
            }

            x: 20 // Отступ слева
            y: 220
        }
        Button {
            text: "Analyze"
            onClicked: {
                myObject.analyzeFile()
            }
            x: 20 // Отступ слева
            y: 340
        }

        Label {
            text: "Home Page"
            x: 20 // Отступ слева
            y: 50
        }
        Label {
            text: selectedFilePath // Связывание текста с выбранным путем к файлу
            x: 20
            y: 220
        }
    }
}
Component {
    id: page1Component
    Item {
        Button {
            text: "Load File"
            onClicked: {
                file_dialog.open() // Открыть собственное диалоговое окно при нажатии на кнопку
                //myObject.loadFile1(file_dialog.fileUrl)
            }
            x: 20 // Отступ слева
            y: 100
        }

        Button {
            text: "Load File2"
            onClicked: {
                file_dialog2.open() // Открыть собственное диалоговое окно при нажатии на кнопку
                //myObject.loadFile2(file_dialog.fileUrl)
            }

            x: 20 // Отступ слева
            y: 220
        }
        Button {
            text: "Analyze"
            onClicked: {
                myObject.analyzeFile()
            }
            x: 20 // Отступ слева
            y: 340
        }

        Label {
            text: "Home Page"
            x: 20 // Отступ слева
            y: 50
        }
        Label {
            text: selectedFilePath // Связывание текста с выбранным путем к файлу
            x: 20
            y: 220
        }
    }
}

    FileDialog {
        id: file_dialog
        title: "Please choose a file"
        folder: shortcuts.home
        onAccepted:  {
            var selectedFile = file_dialog.fileUrl
            //analysis.analyzeFile(selectedFile)
            myObject.loadFile1(selectedFile)
            //homeComponent.selectedFilePath = selectedFile // Присваивание выбранного пути к файлу свойству selectedFilePath
        }
    }
FileDialog {
    id: file_dialog2
    title: "Please choose a file"
    folder: shortcuts.home
    onAccepted: {
        var selectedFile = file_dialog2.fileUrl
        myObject.loadFile2(selectedFile)
        homeComponent.selectedFilePath = selectedFile
    }
}

}