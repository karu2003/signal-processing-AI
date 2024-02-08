from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_PlotsWindow(object):
    def setupUi(self, PlotsWindow):
        PlotsWindow.setObjectName("PlotsWindow")
        PlotsWindow.resize(807, 650)
        self.centralwidget = QtWidgets.QWidget(PlotsWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setContentsMargins(5, 5, 5, 5)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        # self.grid = QtWidgets.QGridLayout(self.verticalLayout)
        self.plot = PlotWidget(self.centralwidget)
        self.plot.setObjectName("plot")
        self.verticalLayout.addWidget(self.plot)
        PlotsWindow.setCentralWidget(self.centralwidget)
        self.dockWidget_2 = QtWidgets.QDockWidget(PlotsWindow)
        self.dockWidget_2.setMinimumSize(QtCore.QSize(190, 100))
        self.dockWidget_2.setMaximumSize(QtCore.QSize(524287, 300))
        self.dockWidget_2.setFeatures(QtWidgets.QDockWidget.DockWidgetFloatable|QtWidgets.QDockWidget.DockWidgetMovable|QtWidgets.QDockWidget.DockWidgetVerticalTitleBar)
        self.dockWidget_2.setAllowedAreas(QtCore.Qt.BottomDockWidgetArea|QtCore.Qt.TopDockWidgetArea)
        self.dockWidget_2.setObjectName("dockWidget_2")
        self.dockWidgetContents_2 = QtWidgets.QWidget()
        self.dockWidgetContents_2.setObjectName("dockWidgetContents_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.dockWidgetContents_2)
        self.horizontalLayout.setContentsMargins(5, 0, 5, 5)
        self.horizontalLayout.setSpacing(5)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.signal = PlotWidget(self.dockWidgetContents_2)
        self.signal.setObjectName("signal")
        self.horizontalLayout.addWidget(self.signal)
        self.dockWidget_2.setWidget(self.dockWidgetContents_2)
        PlotsWindow.addDockWidget(QtCore.Qt.DockWidgetArea(8), self.dockWidget_2)

        self.retranslateUi(PlotsWindow)
        QtCore.QMetaObject.connectSlotsByName(PlotsWindow)

    def retranslateUi(self, PlotsWindow):
        _translate = QtCore.QCoreApplication.translate
        PlotsWindow.setWindowTitle(_translate("PlotsWindow", "MainWindow"))
        self.dockWidget_2.setWindowTitle(_translate("PlotsWindow", "Time Series"))

from pyqtgraph import PlotWidget
from pyqtgraph.Qt.QtGui import *
from pyqtgraph.Qt.QtCore import *
from pyqtgraph.Qt.QtWidgets import *


class OverlayButtons(QWidget):

    def __init__(self, parent=None):
        QWidget.__init__(self)
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        btn_size = (60,60)
        self.verticalLayout = QVBoxLayout(self)
        self.verticalLayout.setContentsMargins(2, 2, 2, 2)
        self.verticalLayout.setSpacing(5)
        # self.grid = QtWidgets.QGridLayout(self)
        self.btn_1 = QPushButton()
        self.btn_1.setMinimumSize(QSize(int(btn_size[0]), int(btn_size[1])))
        self.btn_1.setMaximumSize(QSize(int(btn_size[0]), int(btn_size[1])))
        self.btn_1.hitButton
        self.btn_1.setText('CH1')
        self.verticalLayout.addWidget(self.btn_1)#, 0, QtCore.Qt.AlignRight | QtCore.Qt.AlignBottom)
        self.btn_2 = QPushButton()
        self.btn_2.setMinimumSize(QSize(int(btn_size[0]), int(btn_size[1])))
        self.btn_2.setMaximumSize(QSize(int(btn_size[0]), int(btn_size[1])))
        self.btn_2.setObjectName("btn_2")
        self.btn_2.setText('start')
        # self.btn_2.resize(60,60)
        # self.btn_2.move(100, 100)
        self.verticalLayout.addWidget(self.btn_2)#, 0, QtCore.Qt.AlignRight | QtCore.Qt.AlignBottom)
        self.btn_1.clicked.connect(self.onButtonClicked)
        self.btn_2.clicked.connect(self.onButtonClicked1)

        # self.grid.addWidget(self.btn_1, 0, 0, QtCore.Qt.AlignLeft | QtCore.Qt.AlignBottom)
        # self.grid.addWidget(self.btn_2, 0, 1, QtCore.Qt.AlignRight | QtCore.Qt.AlignBottom)


    def onButtonClicked(self):
        # sender = self.sender()
        text = self.btn_1.text()
        if text == "CH1":
            self.btn_1.setText("CH2")
        else:
            self.btn_1.setText("CH1")
        print("Clicked")

    def onButtonClicked1(self):
        print("Clicked1")


import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import pyqtgraph as pg


class Plots(QMainWindow):

    def __init__(self):
        super(Plots, self).__init__()

        self.ui = Ui_PlotsWindow()
        self.ui.setupUi(self)

        self.overlay_buttons = OverlayButtons(self.ui.plot)
        self.overlay_buttons.setParent(self.ui.plot)
        self.overlay_buttons.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Plots()
    window.show()
    sys.exit(app.exec_())