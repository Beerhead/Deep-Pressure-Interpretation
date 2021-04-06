import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter
from PyQt5 import QtCore, QtGui
from PyQt5.Qt import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from io import BytesIO
from itertools import chain


class PlotWidget2(pg.PlotWidget):
    btnSpuskPressedSignal = pyqtSignal()
    btnPodemPressedSignal = pyqtSignal()
    lineSignalToMain = pyqtSignal(object, object, object)

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

        self.button1.clicked.connect(lambda: self.btnSpuskPressedSignal.emit())
        self.button2.clicked.connect(lambda: self.btnPodemPressedSignal.emit())

    def plot(self, dataToPlot, times=None, save=False):
        dataToPlot.to_clipboard()
        self.plotItem.clear()
        try:
            self.plotItem.scene().removeItem(self.plot2)
        except AttributeError:
            pass
        self.plotItem.setAxisItems(self.axisItem)
        self.plotItem.getAxis('left').setLabel('Pressure, at., Temperature, °C')
        x1 = dataToPlot.iloc[:, 0]
        y1 = dataToPlot.iloc[:, 1]
        y2 = dataToPlot.iloc[:, 2]
        x2 = dataToPlot.iloc[:, 3]
        y3 = dataToPlot.iloc[:, 4]
        for g in (x1, y1, y2, x2, y3):
            g.dropna(inplace=True)
        x1 = [i.to_pydatetime().timestamp() for i in x1.to_list()]
        x2 = [i.to_pydatetime().timestamp() for i in x2.to_list()]
        self.plotItem.addLegend()
        self.plotItem.addLegend().clear()
        self.plotItem.plot(x1, y1, pen=pg.mkPen('g', width=2), name='Pressure')
        self.plotItem.plot(x1, y2, pen=pg.mkPen('r', width=2), name="Temperature")
        self.plot2 = pg.ViewBox()
        self.plot2.clear()
        self.plotItem.showAxis('right')
        self.plotItem.scene().addItem(self.plot2)
        self.plotItem.getAxis('right').linkToView(self.plot2)
        self.plot2.setXLink(self.plotItem)
        self.plotItem.getAxis('right').setLabel('Depth, m.')
        self.updateViews()
        self.plotItem.vb.sigResized.connect(self.updateViews)
        curve3 = pg.PlotDataItem(x2, y3, name='Depth', pen=pg.mkPen('b', width=2))
        self.plot2.addItem(curve3)
        self.plotItem.addLegend().addItem(curve3, 'Depth')
        self.plotItem.getViewBox().autoRange()

        if save:
            exporter = ImageExporter(self.plotItem)
            exporter.parameters()['width'] = 580
            exporter.parameters()['height'] = 450
            buffer = QBuffer()
            buffer.open(QIODevice.ReadWrite)
            exporter.export(toBytes=True).save(buffer, "PNG")
            buffer.seek(0)
            fig = BytesIO(buffer.readAll())
            return fig
        else:
            if times is not None:
                self.button1.setVisible(True)
                self.button2.setVisible(True)
                vLines1, vLines2 = [self.makeInfLine(time, 0) for time in times[0]], \
                                   [self.makeInfLine(time, 1) for time in times[1]]
                for line in chain(vLines1, vLines2):
                    self.plotItem.addItem(line)
                    line.infLineSignal.connect(self.emittingToMain)

    def emittingToMain(self, line, start, stop):
        self.lineSignalToMain.emit(line, start.x(), stop)

    def makeInfLine(self, time, polka):
        return PolkaInfLine(pos=time.to_pydatetime().timestamp(), polka=polka)

    def updateViews(self):
        self.plot2.setGeometry(self.plotItem.getViewBox().sceneBoundingRect())
        self.proxy2.setGeometry(QtCore.QRectF(self.plotItem.geometry().getCoords()[2] - 197, 0, 153, 23))


class PolkaInfLine(pg.InfiniteLine):
    infLineSignal = pyqtSignal(object, object, object)

    def __init__(self, pos, polka, parent=None):
        super(PolkaInfLine, self).__init__(parent)
        self.setMovable(True)
        self.setPos(pos)
        self.polka = polka
        self.sigPositionChangeFinished.connect(self.emitting)

    def emitting(self):
        self.infLineSignal.emit(self, self.startPosition, self.getXPos())


class PlotWidget(QWidget):
    def __init__(self, parent=None):
        super(PlotWidget, self).__init__(parent)
        self.initUi()

    def initUi(self):
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.plotLayout = QVBoxLayout(self)
        self.plotLayout.addWidget(self.canvas)

    def plot(self, dataToPlot, times=None, save=False):
        x1 = dataToPlot.iloc[:, 0]
        y1 = dataToPlot.iloc[:, 1]
        y2 = dataToPlot.iloc[:, 2]
        x2 = dataToPlot.iloc[:, 3]
        y3 = dataToPlot.iloc[:, 4]
        for g in (x1, y1, y2, x2, y3):
            g.dropna(inplace=True)
        self.figure.clear()
        vis = self.figure.subplots()
        vis.plot(x1, y1, "g", label='Pressure')
        vis.plot(x1, y2, "r", label='Temperature')
        ax2 = vis.axes.twinx()
        ax2.plot(x2, y3, label='Depth')
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
                vis.vlines(times[0], y1.min(), y1.max())
                vis.vlines(times[1], y1.min(), y1.max())
            self.canvas.draw()
