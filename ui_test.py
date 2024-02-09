# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'mainwindow.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from pyqtgraph import PlotWidget


class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(1040, 636)
        self.horizontalLayout = QHBoxLayout(Form)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.graphWidget = PlotWidget(Form)
        self.graphWidget.setObjectName(u"graphWidget")
        self.graphWidget.setMinimumSize(QSize(300, 0))

        self.horizontalLayout.addWidget(self.graphWidget)

        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.btn_save = QPushButton(Form)
        self.btn_save.setObjectName(u"btn_save")
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btn_save.sizePolicy().hasHeightForWidth())
        self.btn_save.setSizePolicy(sizePolicy)
        self.btn_save.setMinimumSize(QSize(60, 60))

        self.verticalLayout.addWidget(self.btn_save)

        self.btn_lear = QPushButton(Form)
        self.btn_lear.setObjectName(u"btn_lear")
        sizePolicy.setHeightForWidth(self.btn_lear.sizePolicy().hasHeightForWidth())
        self.btn_lear.setSizePolicy(sizePolicy)
        self.btn_lear.setMinimumSize(QSize(60, 60))

        self.verticalLayout.addWidget(self.btn_lear)

        self.btn_ch = QPushButton(Form)
        self.btn_ch.setObjectName(u"btn_ch")
        sizePolicy.setHeightForWidth(self.btn_ch.sizePolicy().hasHeightForWidth())
        self.btn_ch.setSizePolicy(sizePolicy)
        self.btn_ch.setMinimumSize(QSize(60, 60))

        self.verticalLayout.addWidget(self.btn_ch)

        self.btn_start = QPushButton(Form)
        self.btn_start.setObjectName(u"btn_start")
        sizePolicy1 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy1.setHorizontalStretch(1)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.btn_start.sizePolicy().hasHeightForWidth())
        self.btn_start.setSizePolicy(sizePolicy1)
        self.btn_start.setMinimumSize(QSize(60, 60))

        self.verticalLayout.addWidget(self.btn_start)


        self.horizontalLayout.addLayout(self.verticalLayout)

        self.horizontalLayout.setStretch(0, 10)
        self.horizontalLayout.setStretch(1, 1)

        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Form", None))
        self.btn_save.setText(QCoreApplication.translate("Form", u"PushButton", None))
        self.btn_lear.setText(QCoreApplication.translate("Form", u"PushButton", None))
        self.btn_ch.setText(QCoreApplication.translate("Form", u"PushButton", None))
        self.btn_start.setText(QCoreApplication.translate("Form", u"PushButton", None))
    # retranslateUi

