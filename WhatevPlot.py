import sys
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QMainWindow, QApplication

class WhatevPlot(QMainWindow):
    def __init__(self, bufferSize=1000, title="Whatev", window_size=(400, 400)):
        super().__init__()
        self.setWindowTitle(title)
        self.resize(*window_size)

        self.plot_widget = pg.PlotWidget()
        self.setCentralWidget(self.plot_widget)

        self.bufferSize = bufferSize
        self.data = np.zeros(self.bufferSize)
        self.curve = self.plot_widget.plot(self.data, pen=pg.mkPen('g', width=1.5))

        self.show()

    def display(self, newVal):
        self.data[:-1] = self.data[1:]
        self.data[-1] = newVal
        self.curve.setData(self.data)