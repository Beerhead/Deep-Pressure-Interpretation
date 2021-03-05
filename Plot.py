import pyqtgraph as pg
from PyQt5 import QtCore, QtGui
from PyQt5.Qt import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from io import BytesIO
from itertools import chain


class PlotWidget2(pg.PlotWidget):
    btn_spusk_pressed_signal = pyqtSignal()
    btn_podem_pressed_signal = pyqtSignal()
    line_signal_to_main = pyqtSignal(object, object, object)

    def __init__(self, parent=None):
        super(PlotWidget2, self).__init__(parent)
        self.axisItem = {'bottom': pg.DateAxisItem()}
        self.plotItem.getViewBox().setMouseMode(self.plotItem.getViewBox().RectMode)
        self.proxy1 = QtGui.QGraphicsProxyWidget()
        self.proxy2 = QtGui.QGraphicsProxyWidget()
        self.button1 = QtGui.QPushButton('Добавить полку на спуске')
        self.button1.setVisible(False)
        self.button2 = QtGui.QPushButton('Добавить полку на подъеме')
        self.button2.setVisible(False)
        self.proxy1.setWidget(self.button1)
        self.proxy2.setWidget(self.button2)
        self.proxy1.setGeometry(QtCore.QRectF(52, 0, 143, 23))
        self.proxy2.setGeometry(QtCore.QRectF(self.plotItem.geometry().getCoords()[2] - 197, 0, 153, 23))
        self.plotItem.scene().addItem(self.proxy1)
        self.plotItem.scene().addItem(self.proxy2)

        self.button1.clicked.connect(lambda: self.btn_spusk_pressed_signal.emit())
        self.button2.clicked.connect(lambda: self.btn_podem_pressed_signal.emit())

    def plot(self, datatoplot, times=None, save=False):
        datatoplot.to_clipboard()
        self.plotItem.clear()
        try:
            self.plotItem.scene().removeItem(self.plot2)
        except AttributeError:
            pass
        self.plotItem.setAxisItems(self.axisItem)
        self.plotItem.getAxis('left').setLabel('Pressure, at., Temperature, °C')
        X1 = datatoplot.iloc[:, 0]
        Y1 = datatoplot.iloc[:, 1]
        Y2 = datatoplot.iloc[:, 2]
        X2 = datatoplot.iloc[:, 3]
        Y3 = datatoplot.iloc[:, 4]
        for g in (X1, Y1, Y2, X2, Y3):
            g.dropna(inplace=True)
        X1 = [i.to_pydatetime().timestamp() for i in X1.to_list()]
        X2 = [i.to_pydatetime().timestamp() for i in X2.to_list()]
        self.plotItem.addLegend()
        self.plotItem.addLegend().clear()
        self.plotItem.plot(X1, Y1, pen=pg.mkPen('g', width=2), name='Pressure')
        self.plotItem.plot(X1, Y2, pen=pg.mkPen('r', width=2), name="Temperature")
        self.plot2 = pg.ViewBox()
        self.plot2.clear()
        self.plotItem.showAxis('right')
        self.plotItem.scene().addItem(self.plot2)
        self.plotItem.getAxis('right').linkToView(self.plot2)
        self.plot2.setXLink(self.plotItem)
        self.plotItem.getAxis('right').setLabel('Depth, m.')
        self.updateViews()
        self.plotItem.vb.sigResized.connect(self.updateViews)
        curve3 = pg.PlotDataItem(X2, Y3, name='Depth', pen=pg.mkPen('b', width=2))
        self.plot2.addItem(curve3)
        self.plotItem.addLegend().addItem(curve3, 'Depth')
        self.plotItem.getViewBox().autoRange()

        if save:
            # fig = BytesIO()
            # self.figure.savefig(fig, format='png')
            # return fig
            pass
        else:
            if times is not None:
                self.button1.setVisible(True)
                self.button2.setVisible(True)
                vlines1, vlines2 = [self.make_inf_line(time, 0) for time in times[0]], \
                                   [self.make_inf_line(time, 1) for time in times[1]]
                for line in chain(vlines1, vlines2):
                    self.plotItem.addItem(line)
                    line.inf_line_signal.connect(self.emitting_to_main)

    def emitting_to_main(self, line, start, stop):
        self.line_signal_to_main.emit(line, start.x(), stop)

    def make_inf_line(self, time, polka):
        return PolkaInfLine(pos=time.to_pydatetime().timestamp(), polka=polka)

    def updateViews(self):
        self.plot2.setGeometry(self.plotItem.getViewBox().sceneBoundingRect())
        self.proxy2.setGeometry(QtCore.QRectF(self.plotItem.geometry().getCoords()[2] - 197, 0, 153, 23))


class PolkaInfLine(pg.InfiniteLine):
    inf_line_signal = pyqtSignal(object, object, object)

    def __init__(self, pos, polka, parent=None):
        super(PolkaInfLine, self).__init__(parent)
        self.setMovable(True)
        self.setPos(pos)
        self.polka = polka
        self.sigPositionChangeFinished.connect(self.emitting)

    def emitting(self):
        self.inf_line_signal.emit(self, self.startPosition, self.getXPos())


class PlotWidget(QWidget):
    def __init__(self, parent=None):
        super(PlotWidget, self).__init__(parent)
        self.initUi()

    def initUi(self):
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.plotlayout = QVBoxLayout(self)
        self.plotlayout.addWidget(self.canvas)

    def plot(self, datatoplot, times=None, save=False):
        X1 = datatoplot.iloc[:, 0]
        Y1 = datatoplot.iloc[:, 1]
        Y2 = datatoplot.iloc[:, 2]
        X2 = datatoplot.iloc[:, 3]
        Y3 = datatoplot.iloc[:, 4]
        for g in (X1, Y1, Y2, X2, Y3):
            g.dropna(inplace=True)
        self.figure.clear()
        vis = self.figure.subplots()
        vis.plot(X1, Y1, "g", label='Pressure')
        vis.plot(X1, Y2, "r", label='Temperature')
        ax2 = vis.axes.twinx()
        ax2.plot(X2, Y3, label='Depth')
        self.figure.legend(loc=10, bbox_to_anchor=(0.6, 0.5, 0.5, 0.5))
        vis.set_ylabel('Pressure, at., Temperature, °C')
        ax2.set_ylabel('Depth, m.')
        # vis.xaxis.set_major_locator()
        if save:
            fig = BytesIO()
            self.figure.savefig(fig, format='png')
            return fig
        else:
            if times is not None:
                vis.vlines(times[0], Y1.min(), Y1.max())
                vis.vlines(times[1], Y1.min(), Y1.max())
            self.canvas.draw()
