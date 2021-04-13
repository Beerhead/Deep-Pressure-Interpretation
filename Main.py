import sys
import pathlib
import warnings
from reportlab.pdfgen import canvas
from Other import *
from Interpretation import *
from Plot import *
from Report import *
import pyqtgraph as pg
import joblib

warnings.filterwarnings('ignore')


class MainWindow(QMainWindow):
    """Main Window of application"""
    def __init__(self):
        super(MainWindow, self).__init__()
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        self.researchsList = []
        self.initUI()
        self.connectUI()
        self.interpreted = False

    def initUI(self):
        self.testList = QTableView()
        self.testList.setFixedWidth(450)
        self.testList.rowHeight(25)
        self.centralWidget = QWidget(self)
        self.HL = QHBoxLayout(self.centralWidget)  # основной горизонтальный лэйаут
        self.VL1 = QVBoxLayout()  # вертикал лэй №1
        self.VL2 = QVBoxLayout()  # вертикал лэй №2
        self.plotWidget = PlotWidget()
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        self.plotWidget2 = PlotWidget2()
        self.RB1 = QRadioButton("По спуску")
        self.RB2 = QRadioButton("По подъему")
        self.HLTables = QHBoxLayout()  # лэй для двух таблиц
        self.calcTable1 = QTableView()
        self.calcTable2 = QTableView()
        self.calcTable1.setMaximumWidth(540)
        self.calcTable2.setMaximumWidth(540)
        self.calcTable1.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.calcTable2.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.resLabel = QLabel('Данные по скважине')
        self.resLabel.setAlignment(Qt.AlignCenter)
        self.HLTables.addWidget(self.calcTable1)
        self.HLTables.addWidget(self.calcTable2)
        self.labelChBoxLay = QVBoxLayout()  # лэй для лэйбла и чекбоксов
        self.chBoxLay = QHBoxLayout()
        self.chBoxLay.addWidget(self.RB1)
        self.chBoxLay.addWidget(self.RB2)
        self.chBoxLay.setAlignment(Qt.AlignCenter)
        self.deltaEdit = QLineEdit("100")
        self.deltaEdit.setAlignment(Qt.AlignCenter)
        self.deltaEdit.setMaximumWidth(60)
        self.deltaEdit.setValidator(QIntValidator(1, 1000))
        self.labelChBoxLay.addWidget(self.resLabel)
        self.labelChBoxLay.addLayout(self.chBoxLay)
        self.labelChBoxLay.setAlignment(Qt.AlignCenter)
        self.HLTables.addLayout(self.labelChBoxLay)
        self.GL = QGridLayout()  # лэй для кнопок
        self.btn1 = QPushButton('Интерпретация 1')
        self.btn2 = QPushButton('Замеры ОТС')
        self.btn3 = QPushButton('Интерпретация 2')
        self.btn4 = QPushButton('Выгрузить отчеты')
        self.btn4.setDisabled(True)
        self.GL.addWidget(self.btn1, 1, 1)
        self.GL.addWidget(self.btn2, 1, 2)
        self.GL.addWidget(self.btn3, 2, 1)
        self.GL.addWidget(self.btn4, 2, 2)
        self.measureWidget = MeasurementsWidget(self)
        self.gifDialog = GifDialog()
        # добавка 2 графика
        self.hlGraphLayout = QHBoxLayout()
        self.hlGraphLayout.addWidget(self.plotWidget2)
        self.VL1.addLayout(self.HLTables)
        self.VL1.addLayout(self.hlGraphLayout)
        self.VL2.addWidget(self.testList)
        self.VL2.addLayout(self.GL)
        self.HL.addLayout(self.VL1)
        self.HL.addLayout(self.VL2)
        self.setCentralWidget(self.centralWidget)


    def connectUI(self):

        self.btn1.clicked.connect(lambda: self.interpretateResearches(method=1))
        self.btn2.clicked.connect(lambda: self.measureWidget.show())
        self.btn3.clicked.connect(lambda: self.interpretateResearches(method=2))
        self.btn4.clicked.connect(self.reports)
        self.testList.clicked.connect(self.graf)
        self.calcTable1.clicked.connect(self.graf)
        self.calcTable2.clicked.connect(self.graf)
        self.RB1.clicked.connect(self.RB1Clicked)
        self.RB2.clicked.connect(self.RB2Clicked)
        self.plotWidget2.btnSpuskPressedSignal.connect(lambda: self.addInfLine(0))
        self.plotWidget2.btnPodemPressedSignal.connect(lambda: self.addInfLine(1))
        self.plotWidget2.lineSignalToMain.connect(self.infLineMoved)

    def receiveList(self, modelToSet, listOfMeasuarements):

        self.listModel = modelToSet
        self.testList.setModel(self.listModel)
        delegate = IntDelegate()
        delegate.createEditor(self.testList, QStyleOptionViewItem(), QtCore.QModelIndex())
        self.testList.setItemDelegateForColumn(2, delegate)
        self.researchsList = listOfMeasuarements
        self.btn3.setDisabled(False)



    def addInfLine(self, polka):
        tempInd = self.testList.selectedIndexes()[0].row()
        mediumTime = self.researchsList[tempInd].resMid
        time, *_ = self.timeBindInfLine(mediumTime)
        line = self.plotWidget2.makeInfLine(time, polka)
        self.plotWidget2.plotItem.addItem(line)
        line.infLineSignal.connect(self.plotWidget2.emittingToMain)


    def infLineMoved(self, line, start, stop):
        tempInd = self.testList.selectedIndexes()[0].row()
        stop = pd.Timestamp.fromtimestamp(stop)
        start = pd.Timestamp.fromtimestamp(start).round(freq='1S')
        supTimes = self.researchsList[tempInd].timesList[line.polka]
        trueFalseList = [start == i.round(freq='1S') for i in supTimes]
        time, depth, elong, pres, temper = self.timeBindInfLine(stop)
        try:
            index = trueFalseList.index(True)
            self.researchsList[tempInd].supTimes[line.polka][index] = time
            self.researchsList[tempInd].dataFromSupTimes[line.polka][index] = [depth, elong, pres, temper]
        except:
            self.researchsList[tempInd].supTimes[line.polka].append(time)
            self.researchsList[tempInd].dataFromSupTimes[line.polka].append([depth, elong, pres, temper])

        print(self.researchsList[tempInd].dataFromSupTimes[line.polka])

        data = sorted(self.researchsList[tempInd].dataFromSupTimes[line.polka], key=lambda x: x[0])
        self.researchsList[tempInd].tableModels[line.polka], *_ = self.researchsList[tempInd].makeModel(data)
        # for row in range(1, model.rowCount()):
        #     newModel.item(row, 6).setCheckState(0)
        #     if model.item(row, 6).checkState() == 2:
        #         newModel.item(row, 6).setCheckState(2)

        self.graf()


    def timeBindInfLine(self, time):
        tempInd = self.testList.selectedIndexes()[0].row()
        # по манометру
        timeSeries = self.researchsList[tempInd].finalData.iloc[:, 0]
        timeSeries.dropna(inplace=True)
        ind = timeSeries.searchsorted(time)
        cond1 = (timeSeries[ind].timestamp() - time.timestamp()) / timeSeries[ind].timestamp()
        cond2 = (timeSeries[ind + 1].timestamp() - time.timestamp()) / timeSeries[ind].timestamp()
        if cond1 < cond2:
            finalInd = ind
        else:
            finalInd = ind + 1
        time = timeSeries[finalInd]
        pres = self.researchsList[tempInd].finalData.iloc[finalInd, 1]
        temper = self.researchsList[tempInd].finalData.iloc[finalInd, 2]
        # по СПС
        timeSeries = self.researchsList[tempInd].finalData.iloc[:, 3]
        timeSeries.dropna(inplace=True)
        ind = timeSeries.searchsorted(time)
        cond1 = (timeSeries[ind].timestamp() - time.timestamp()) / timeSeries[ind].timestamp()
        cond2 = (timeSeries[ind + 1].timestamp() - time.timestamp()) / timeSeries[ind].timestamp()
        if cond1 < cond2:
            finalInd = ind
        else:
            finalInd = ind + 1
        depth = self.researchsList[tempInd].finalData.iloc[finalInd, 4]
        elong = searchAndInterpolate(self.researchsList[tempInd].incl, depth)
        return time, round(depth, 0), round(elong, 2), round(pres, 2), round(temper, 2)


    def RB1Clicked(self):
        tempInd = self.testList.selectedIndexes()[0].row()
        self.researchsList[tempInd].tableInd = 0
        self.graf()


    def RB2Clicked(self):
        tempInd = self.testList.selectedIndexes()[0].row()
        self.researchsList[tempInd].tableInd = 1
        self.graf()

    def keyPressEvent(self, e):
        if e.key() == QtCore.Qt.Key_Delete and len(self.testList.selectedIndexes()) > 0 and self.testList.hasFocus():
            for i in self.testList.selectedIndexes()[::-1]:
                self.listModel.removeRow(i.row())
                self.researchsList.pop(i.row())
        if e.key() == QtCore.Qt.Key_F10:
            print("ТЫЩЩ")
            for res in self.researchsList:
                print(res.resID)
                self.insertResearchDataToDB(res)
        if e.key() == QtCore.Qt.Key_F9:
            print("ТЫЩЩ")
            for res in self.researchsList:
                print(res.wellName)


    def magicShow(self):
        self.gifDialog.show()
        self.gifDialog.movie.start()

    def interpretateResearches(self, method):
        if len(self.researchsList) < 1: return
        self.now = datetime.datetime.now()
        pplIntervalDict = {}
        self.magicShow()
        for i, res in enumerate(self.researchsList):
            res.warning = False
            interval = int(self.testList.model().item(i, 2).text())
            pplIntervalDict[res] = interval
        self.thread = InterpretationThread(self, pplIntervalDict, method=method)
        self.thread.finishSignal.connect(self.stopGif)
        self.thread.start()


    def stopGif(self, newResList):
        self.gifDialog.movie.stop()
        self.gifDialog.hide()
        self.thread.end()
        self.btn4.setDisabled(False)
        self.interpreted = True
        self.researchsList = newResList
        for i, res in enumerate(self.researchsList):
            if res.warning:
                self.listModel.item(i, 0).setIcon(self.style().standardIcon(10))
            else:
                self.listModel.item(i, 0).setIcon(self.style().standardIcon(45))
                if res.interpreted:
                    self.listModel.item(i, 3).setCheckState(2)
        print(datetime.datetime.now() - self.now)


    def insertResearchDataToDB(self, res):
        if res.tableModels is None: return
        insertDataToSosresearchTable(res)
        insertDataToSosresearchmeasurementTable(res)
        insertDataToSosresearchgraphTable(res)
        insertDataToSosresearchmarkermeasurementTable(res)
        insertDataToSosresearchwellTable(res)
        insertDataToSosresearchvalidationTable(res)
        insertDataToSosresearchlayerinputTable(res)
        insertDataToSosresearchperforationTable(res)
        insertDataToSosresearchresultTable(res)
        insertDataToSosresearchmarkerlayerTable(res)

    def graf(self):
        tempInd = self.testList.selectedIndexes()[0].row()
        layers = self.researchsList[tempInd].layer
        layers = ', '.join(layers)
        vdp = self.researchsList[tempInd].vdp
        elongation = self.researchsList[tempInd].vdpElong
        if self.researchsList[tempInd].interpreted:
            if self.researchsList[tempInd].tableInd == 0:
                self.RB1.setChecked(True)
            else:
                self.RB2.setChecked(True)
            ros = []
            ppls = []
            self.calcTable1.setModel(self.researchsList[tempInd].tableModels[0])
            self.calcTable2.setModel(self.researchsList[tempInd].tableModels[1])
            for i in range(7):
                self.calcTable2.setColumnWidth(i, 75)
                self.calcTable1.setColumnWidth(i, 75)

            checkedSupportTimes = []
            for num, m in enumerate(self.researchsList[tempInd].tableModels):
                halfData = []
                numDots = 0
                ro = 0
                for i in range(1, m.rowCount()):
                    if i == m.rowCount() - 1:
                        manAbsDepth = float(m.item(i, 0).text()) - float(m.item(i, 1).text())
                        vdpAbsDepth = self.researchsList[tempInd].vdp - self.researchsList[tempInd].vdpElong
                        delta = vdpAbsDepth - manAbsDepth
                    if m.item(i, 6).checkState() == 2:
                        halfData.append(self.researchsList[tempInd].timesList[num][i])
                        ro += float(m.item(i, 4).text())
                        numDots += 1
                ros.append(round(ro / numDots, 3))
                ppls.append(round(float(m.item(m.rowCount() - 1, 2).text()) + delta * ros[-1] / 10, 3))
                checkedSupportTimes.append(halfData)

            if (self.focusWidget() == self.calcTable1 and self.RB1.isChecked()) or \
                (self.focusWidget() == self.plotWidget2 and self.RB1.isChecked()) or (self.focusWidget() == self.RB1):
                self.researchsList[tempInd].ro = ros[0]
                self.researchsList[tempInd].ppl = ppls[0]
            if self.focusWidget() == self.calcTable2 and self.RB2.isChecked() or\
                (self.focusWidget() == self.plotWidget2 and self.RB2.isChecked()) or (self.focusWidget() == self.RB2):
                self.researchsList[tempInd].ro = ros[1]
                self.researchsList[tempInd].ppl = ppls[1]

            self.plotWidget.plot(self.researchsList[tempInd].finalData, checkedSupportTimes)
            self.resLabel.setText('Данные по скважине' + '\n' +
                                  'Глубина ВДП - ' + str(vdp) + '\n' +
                                  'Удлинение на ВДП - ' + str(elongation) + '\n' +
                                  'Пласт(ы) - ' + str(layers) + '\n' +
                                  'Плотность на спуске - ' + str(ros[0]) + '\n' +
                                  'Плотность на подъеме - ' + str(ros[1]) + '\n' +
                                  'Пластовое давление на ВДП по спуску - ' + str(ppls[0]) + '\n' +
                                  'Пластовое давление на ВДП по подъему - ' + str(ppls[1]))
            self.plotWidget2.plot(self.researchsList[tempInd].finalData, checkedSupportTimes)
            self.researchsList[tempInd].calcAvgTempGradient()
            self.researchsList[tempInd].calcPhaseBorders()
            self.researchsList[tempInd].determineStaticLevel()
        else:
            self.plotWidget.plot(self.researchsList[tempInd].data)
            self.resLabel.setText('Данные по скважине' + '\n' +
                                  'Глубина ВДП - ' + str(vdp) + '\n' +
                                  'Удлинение на ВДП - ' + str(elongation) + '\n' +
                                  'Пласт(ы) - ' + str(layers) + '\n')
            self.plotWidget2.plot(self.researchsList[tempInd].data)


    def reports(self):
        toDeleteList=[]
        for i, res in enumerate(self.researchsList):
            if not res.interpreted or self.listModel.item(i,3).checkState() != 2: continue
            self.insertResearchDataToDB(res)
            fileName = str(res.field) + " " + str(res.wellName) + '.pdf'
            filePath = (pathlib.Path('D:/') / 'Interpretator 9000' / 'Else' / 'Reports' / fileName).__str__()
            cvs = canvas.Canvas(filePath)
            fig = self.researchsList[i].finalFig
            fig.seek(0)
            model = self.researchsList[i].tableModels[self.researchsList[i].tableInd]
            tableData = []
            checked = []
            columns = model.columnCount()
            rows = model.rowCount()
            for r in range(rows):
                rowToList = []
                for c in range(columns - 1):
                    if not (r == 0 and (c == 4 or c == 5)):
                        rowToList.append(model.item(r, c).text())
                if r != 0:
                    if model.item(r, 6).checkState() == 2:
                        checked.append(r)
                tableData.append(rowToList)
            well = self.researchsList[i].wellName
            field = self.researchsList[i].field
            layer = ''.join(self.researchsList[i].layer)
            researchDate = self.researchsList[i].researchDate
            vdp = self.researchsList[i].vdp
            vdpElongation = self.researchsList[i].vdpElong
            ro = self.researchsList[i].ro
            ppl = self.researchsList[i].ppl
            pdf(cvs, well, field, layer, researchDate, fig, vdp, vdpElongation, ro, ppl, tableData, checked)
            toDeleteList.append(i)
        for i in toDeleteList[::-1]:
            self.listModel.removeRow(i)
            self.researchsList.pop(i)


class GifDialog(QWidget):
    def __init__(self):
        super(GifDialog, self).__init__()
        self.setFixedSize(252, 252)
        mLabel = QLabel(self)
        self.movie = QMovie()
        self.movie.setFileName(os.getcwd() + r"\mgc.gif")
        mLabel.setMovie(self.movie)
        mLabel.setFixedSize(252, 252)
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)



class InterpretationThread(QtCore.QThread):
    finishSignal = pyqtSignal(object)

    def __init__(self, parent=None, pplIntervalDict=None, withML=True, method=1):
        QtCore.QThread.__init__(self, parent)
        self.pplIntervalDict = pplIntervalDict
        self.ML = withML
        self.method = method
        self.pModel = None
        self.dModel = None

    def run(self):
        newResearchList = []
        if self.ML:
            self.pModel = joblib.load('rfc_model_pres.pkl')
            self.dModel = joblib.load('rfc_model_depths.pkl')
        for res in list(self.pplIntervalDict.keys()):
            interval = self.pplIntervalDict[res]
            try:
                newResearchList.append(res.interpret(interval, self.pModel, self.dModel, self.method))
            except:
                newResearchList.append(res)
                res.setWarning()
        self.finishSignal.emit(newResearchList)

    def end(self):
        if self.ML:
            del(self.pModel)
            del(self.dModel)



app = QApplication(sys.argv)
MW = MainWindow()
MW.setWindowTitle("Интерпретатор 9000")
MW.show()
sys.exit(app.exec_())
