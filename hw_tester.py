import argparse
import os
import sys  # We need sys so that we can pass argv to QApplication
import threading
import time
from queue import Full, Queue
from random import randint

import matplotlib.pyplot as plt
import numpy as np
import pyqtgraph as pg

# from PyQt5 import QtCore, QtWidgets
from pyqtgraph import GraphicsLayoutWidget, PlotWidget, plot
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
from scipy import signal
from scipy.fftpack import fft
import redpctl as redpctl

pg.setConfigOptions(antialias=True)
pg.setConfigOption("background", "k")


class Plotter(object):
    def __init__(self):
        self.comp_timeseries = np.genfromtxt(
            "dataset/square.csv", delimiter=","
        )  # signal for comparison
        self.q = Queue(maxsize=20)
        self.rx_buffer_size = 20000
        self.rp_c = redpctl.RedCtl(dec=2)
        # self.rp_c.set_gen(wave_form="square")
        self.rp_c.set_burst(wave_form="sine")
        self.nrows = 1

        # self.app = QtWidgets.QApplication(sys.argv)
        self.app = pg.mkQApp("HW Tester")
        self.qmw = QtWidgets.QMainWindow()
        self.qmw.central_widget = QtWidgets.QWidget()
        self.qmw.vertical_layout = QtWidgets.QVBoxLayout()
        self.qmw.setCentralWidget(self.qmw.central_widget)
        self.qmw.central_widget.setLayout(self.qmw.vertical_layout)
        # self.qmw.showMaximized()
        # self.qmw.showFullScreen()

        #### Add Plot
        self.qmw.graphWidget = pg.PlotWidget()
        self.qmw.graphWidget.setBackground("black")
        # self.qmw.graphWidget.setBackground("red")

        self.traces = {}

        self.win = GraphicsLayoutWidget()

        # pauseBtn = QtGui.QPushButton('Pause')
        # self.win.addItem(pauseBtn)

        self.qmw.vertical_layout.addWidget(self.win, 1)
        self.win.setWindowTitle("HW Tester")
        # self.win.setGeometry(0, 0, 800, 600)
        self.win.setBackground(background="black")
        # self.win.setBackground(background="red")

        wf_xaxis = pg.AxisItem(orientation="left")
        wf_xaxis.setLabel("volt")

        cf_xaxis = pg.AxisItem(orientation="left")
        cf_xaxis.setLabel("amper")

        self.waveform = self.win.addPlot(
            title="AMP Output",
            row=1,
            col=1,
            axisItems={"left": wf_xaxis},
        )
        self.vb_w = self.waveform.getViewBox()

        self.currentform = self.win.addPlot(
            title="Current",
            row=2,
            col=1,
            axisItems={"left": cf_xaxis},
        )
        self.vb_c = self.currentform.getViewBox()

        self.waveform.showGrid(x=True, y=True)
        self.currentform.showGrid(x=True, y=True)

        # self.x = np.arange(0, self.rx_buffer_size)
        self.x_com = np.arange(0, len(self.comp_timeseries))
        self.counter = 0
        self.min = -100

        # self.legend = pg.LegendItem((80,60), offset=(70,20))
        # self.legend.setParentItem(self.waveform.graphicsItem())
        # self.legend.addItem(self.qmw.vertical_layout , str(self.counter))
        self.text = pg.TextItem()
        self.waveform.addItem(self.text)

        self.qmw.show()

        self.run_source = True
        self.thread = threading.Thread(target=self.source)
        self.thread.start()

        self.markers_added = False
        
        btn_Style = """
                        QPushButton { 
                            border: 1px solid black;
                            border-style: outset;
                            border-radius: 2px;
                            color: black;
                            font: 18px;
                            font-family: Baskerville;
                        }
                        QPushButton:pressed { 
                            background-color: rgba(255, 255, 255, 70);
                            border-style: inset;
                        }"""

        self.btn = QtWidgets.QPushButton("learning")
        self.btn.setStyleSheet(btn_Style)
        self.btn1 = QtWidgets.QPushButton("save graph")
        self.btn1.setStyleSheet(btn_Style)
        self.btn.clicked.connect(self.onButtonClicked)
        self.btn1.clicked.connect(self.onButtonClicked1)
        proxy = QtWidgets.QGraphicsProxyWidget()
        proxy1 = QtWidgets.QGraphicsProxyWidget()
        proxy.setWidget(self.btn)
        proxy1.setWidget(self.btn1)
        self.win.addItem(proxy, row=3, col=1)
        self.win.addItem(proxy1, row=0, col=1)
        # self.waveform = self.win.addItem(proxy, row=0, col=0)

    def onButtonClicked(self):
        print("Clicked")
    
    def onButtonClicked1(self):
        print("Clicked1")

    def x_edge(self, data, thresh=0.00):
        mask1 = (data[:-1] < thresh) & (data[1:] > thresh)
        mask2 = (data[:-1] > thresh) & (data[1:] < thresh)
        rising_edge = np.flatnonzero(mask1) + 1
        falling_edge = np.flatnonzero(mask2) + 1
        return rising_edge, falling_edge

    def source(self):
        print("Thread running")
        self.counter = 0
        while self.run_source:
            data = self.rp_c.read(counter=self.nrows, quantity=self.rx_buffer_size)
            timeseries = np.array(data)

            rising_edge, falling_edge = self.x_edge(timeseries[0])
            x_periods = []

            for i in timeseries:
                x_periods.append(i[rising_edge[0] : rising_edge[-1]])

            self.counter += 1
            try:
                self.q.put(x_periods, block=False, timeout=4)
            except Full:
                continue

    def start(self):
        self.app.exec_()

    def set_plotdata(self, name, data_x, data_y):
        if name in self.traces:
            self.traces[name].setData(data_x, data_y)
        elif name == "waveform":
            # self.vb_w.setBackgroundColor("red")
            self.traces[name] = self.waveform.plot(pen="c", width=3)
            # self.traces[name] = self.waveform.plot("",pen="c", width=3)
            self.waveform.setYRange(-1, 1, padding=0)
            self.waveform.setXRange(
                0,
                len(data_x),
                padding=0.005,
            )

        elif name == "comparison":
            self.traces[name] = self.waveform.plot(pen=0.25, width=3)
            self.waveform.setYRange(-1, 1, padding=0)
            self.waveform.setXRange(
                0,
                len(self.comp_timeseries),
                padding=0.005,
            )

        elif name == "current":
            # self.vb_c.setBackgroundColor("g")
            self.traces[name] = self.currentform.plot(pen="y", width=3)
            self.currentform.setYRange(-1, 1, padding=0)
            self.currentform.setXRange(
                0,
                len(data_x),
                padding=0.005,
            )

    def update(self):
        self.text.setText(str(self.counter))
        while not self.q.empty():
            wf_data = self.q.get()
            self.x = np.arange(0, len(wf_data[0]))
            self.set_plotdata(
                name="waveform",
                data_x=self.x,
                data_y=np.real(wf_data[0]),
            )
            self.set_plotdata(
                name="current",
                data_x=self.x,
                data_y=np.real(wf_data[0]),
            )
            # all_results = None
            self.set_plotdata(
                name="comparison",
                data_x=self.x_com,
                data_y=np.real(self.comp_timeseries),
            )

    def animation(self):
        timer = QtCore.QTimer()
        # timer.setInterval(30)
        timer.timeout.connect(self.update)
        timer.start(1)
        self.start()
        self.run_source = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sonar Tester")
    # parser.add_argument(
    #     "--class",
    #     help="pyadi class name to use as plot source",
    #     # type=str,
    #     # required=False,
    #     default="ad7476a",
    # )
    # parser.add_argument(
    #     "--uri",
    #     help="URI of target device",
    #     # type=str,
    #     # required=False,
    #     default="local:",
    # )
    # args = vars(parser.parse_args())

    app = Plotter()
    app.animation()
    print("Exiting...")
    app.thread.join()
