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
        self.plot2 = pg.ViewBox()
        self.button1.clicked.connect(lambda: self.btnSpuskPressedSignal.emit())
        self.button2.clicked.connect(lambda: self.btnPodemPressedSignal.emit())
        self.sigRangeChanged.connect(self.updatePlot2YRange)
        self.data = None
        self.maxYLeft, self.minYLeft = None, None
        self.minYRight, self.maxYRight = None, None

        self.plotItem.setAxisItems(self.axisItem)
        self.plotItem.getAxis('left').setLabel('Pressure, at., Temperature, °C')
        self.plotItem.addLegend()
        self.plotItem.showAxis('right')
        self.plotItem.getAxis('right').setLabel('Depth, m.')


    def updatePlot2YRange(self, widget, newRange):
        if self.maxYLeft is None or self.minYLeft is None: return
        leftInterval = self.maxYLeft - self.minYLeft
        rightInterval = self.maxYRight - self.minYRight
        currentMinYLeft, currentMaxYLeft = newRange[1][0], newRange[1][1]
        newMinYRight = currentMinYLeft/leftInterval * rightInterval + self.minYRight
        newMaxYRight = currentMaxYLeft/leftInterval * rightInterval + self.minYRight
        self.plot2.setYRange(newMinYRight, newMaxYRight)



    def plot(self, dataToPlot, times=None, save=False):
        if not dataToPlot.equals(self.data):
            self.minYLeft, self.maxYLeft, self.minYRight, self.maxYRight = None, None, None, None
        self.data = dataToPlot
        self.data.to_clipboard()
        self.plotItem.clear()
        try:
            self.plotItem.scene().removeItem(self.plot2)
        except AttributeError:
            pass

        x1 = self.data.iloc[:, 0]
        y1 = self.data.iloc[:, 1]
        y2 = self.data.iloc[:, 2]
        x2 = self.data.iloc[:, 3]
        y3 = self.data.iloc[:, 4]

        #print(max(y1.max(), y2.max()), y3.max())
        #print(min(y1.min(), y2.min()), y3.min())
        for g in (x1, y1, y2, x2, y3):
            g.dropna(inplace=True)
        x1 = [i.to_pydatetime().timestamp() for i in x1.to_list()]
        x2 = [i.to_pydatetime().timestamp() for i in x2.to_list()]
        self.plotItem.addLegend().clear()
        self.plotItem.plot(x1, y1, pen=pg.mkPen('g', width=3), name='Pressure')
        self.plotItem.plot(x1, y2, pen=pg.mkPen('r', width=3), name="Temperature")
        self.plot2.clear()
        self.plotItem.scene().addItem(self.plot2)
        self.plotItem.getAxis('right').linkToView(self.plot2)
        self.plot2.setXLink(self.plotItem)
        self.updateViews()
        self.plotItem.vb.sigResized.connect(self.updateViews)
        curve3 = pg.PlotDataItem(x2, y3, name='Depth', pen=pg.mkPen('b', width=3))
        self.plot2.addItem(curve3)
        self.plotItem.addLegend().addItem(curve3, 'Depth')
        self.plotItem.getViewBox().autoRange()
        range = self.plotItem.getViewBox().viewRange()[0]
        minXFirstLeft, maxXFirstLeft = range[0], range[1]
        self.plot2.autoRange()
        range = self.plot2.viewRange()[0]
        minXFirstRight, maxXFirstRight = range[0], range[1]
        self.plot2.setXRange(min(minXFirstRight, minXFirstLeft), max(maxXFirstRight, maxXFirstLeft))
        if (self.minYLeft, self.maxYLeft, self.minYRight, self.maxYRight) == (None, None, None, None):
            range = self.plotItem.getViewBox().viewRange()[1]
            self.minYLeft, self.maxYLeft = range[0], range[1]
            range = self.plot2.viewRange()[1]
            self.minYRight, self.maxYRight = range[0], range[1]
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
