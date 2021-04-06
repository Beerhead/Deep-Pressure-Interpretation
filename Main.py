import sys
import pathlib
import warnings
from PyQt5 import QtWidgets
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
        self.initUi()
        self.connectUi()
        self.interpreted = False

    def initUi(self):
        self.testList = QTableView()
        self.testList.setFixedWidth(315)
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
        self.calctTable1 = QTableView()
        self.calcTable2 = QTableView()
        self.calctTable1.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.calcTable2.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.resLabel = QLabel('Данные по скважине')
        self.resLabel.setAlignment(Qt.AlignCenter)
        self.HLTables.addWidget(self.calctTable1)
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
        self.labelChBoxLay.addLayout(self.chBoxLay)
        self.labelChBoxLay.addWidget(self.resLabel)
        self.labelChBoxLay.setAlignment(Qt.AlignCenter)
        self.HLTables.addLayout(self.labelChBoxLay)
        self.GL = QGridLayout()  # лэй для кнопок
        self.btn1 = QPushButton('Открыть файлы')
        self.btn2 = QPushButton('Замеры ОТС')
        self.btn3 = QPushButton('Интерпретация')
        self.btn3.setDisabled(True)
        self.btn4 = QPushButton('Выгрузить отчеты')
        self.btn4.setDisabled(True)
        self.GL.addWidget(self.btn1, 1, 1)
        self.GL.addWidget(self.btn2, 1, 2)
        self.GL.addWidget(self.btn3, 2, 1)
        self.GL.addWidget(self.btn4, 2, 2)
        self.measureWidget = MeasurementsWidget(self)
        self.gifDialog = gifDialog()
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


    def connectUi(self):

        self.btn2.clicked.connect(lambda: self.measureWidget.show())
        self.btn3.clicked.connect(self.interpretateResearches)
        self.btn4.clicked.connect(self.reports)
        self.testList.clicked.connect(self.graf)
        self.calctTable1.clicked.connect(self.graf)
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
        delta = (self.researchsList[tempInd].sup_times[1][-1] - self.researchsList[tempInd].sup_times[0][-1]) / 2
        mediumTime = self.researchsList[tempInd].sup_times[0][-1] + delta
        time, *_ = self.timeBindInfLine(mediumTime)
        line = self.plotWidget2.makeInfLine(time, polka)
        self.plotWidget2.plotItem.addItem(line)
        line.infLineSignal.connect(self.plotWidget2.emittingToMain)


    def infLineMoved(self, line, start, stop):
        tempInd = self.testList.selectedIndexes()[0].row()
        stop = pd.Timestamp.fromtimestamp(stop)
        start = pd.Timestamp.fromtimestamp(start).round(freq='1S')
        supTimes = self.researchsList[tempInd].sup_times[line.polka]
        trueFalseList = [start == i.round(freq='1S') for i in supTimes]
        time, depth, elong, pres, temper = self.timeBindInfLine(stop)
        try:
            index = trueFalseList.index(True)
            self.researchsList[tempInd].supTimes[line.polka][index] = time
            self.researchsList[tempInd].dataFromSupTimes[line.polka][index] = [depth, elong, pres, temper]
        except:
            self.researchsList[tempInd].sup_times[line.polka].append(time)
            self.researchsList[tempInd].data_from_sup_times[line.polka].append([depth, elong, pres, temper])
        self.researchsList[tempInd].data_from_sup_times[line.polka].sort(key=lambda x: x[0])
        newModel, *_ = self.researchsList[tempInd].makeModel(self.researchsList[tempInd].dataFromSupTimes[line.polka], tempInd)
        # for row in range(1, model.rowCount()):
        #     newModel.item(row, 6).setCheckState(0)
        #     if model.item(row, 6).checkState() == 2:
        #         newModel.item(row, 6).setCheckState(2)
        self.researchsList[tempInd].tableModels[line.polka] = newModel
        self.graf()

    def timeBindInfLine(self, time):
        tempInd = self.testList.selectedIndexes()[0].row()
        # по манометру
        timeSeries = self.researchsList[tempInd].final_data.iloc[:, 0]
        timeSeries.dropna(inplace=True)
        ind = timeSeries.searchsorted(time)
        cond1 = (timeSeries[ind].timestamp() - time.timestamp()) / timeSeries[ind].timestamp()
        cond2 = (timeSeries[ind + 1].timestamp() - time.timestamp()) / timeSeries[ind].timestamp()
        if cond1 < cond2:
            finalInd = ind
        else:
            finalInd = ind + 1
        time = timeSeries[finalInd]
        pres = self.researchsList[tempInd].final_data.iloc[finalInd, 1]
        temper = self.researchsList[tempInd].final_data.iloc[finalInd, 2]
        # по СПС
        timeSeries = self.researchsList[tempInd].final_data.iloc[:, 3]
        timeSeries.dropna(inplace=True)
        ind = timeSeries.searchsorted(time)
        cond1 = (timeSeries[ind].timestamp() - time.timestamp()) / timeSeries[ind].timestamp()
        cond2 = (timeSeries[ind + 1].timestamp() - time.timestamp()) / timeSeries[ind].timestamp()
        if cond1 < cond2:
            finalInd = ind
        else:
            finalInd = ind + 1
        depth = self.researchsList[tempInd].final_data.iloc[finalInd, 4]
        elong = searchAndInterpolate(self.researchsList[tempInd].incl, depth)
        return time, round(depth, 0), round(elong, 2), round(pres, 2), round(temper, 2)


    def RB1Clicked(self):
        temp_ind = self.testList.selectedIndexes()[0].row()
        self.researchsList[temp_ind].table_ind = 0
        self.graf()


    def RB2Clicked(self):
        temp_ind = self.testList.selectedIndexes()[0].row()
        self.researchsList[temp_ind].table_ind = 1
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

    def magicShow(self):
        self.gifDialog.show()
        self.gifDialog.movie.start()

    def interpretateResearches(self):
        if len(self.researchsList) < 1: return
        pplIntervalDict = {}
        self.magicShow()
        for i, res in enumerate(self.researchsList):
            interval = int(self.testList.model().item(i, 2).text())
            pplIntervalDict[res] = interval
        self.thread = interpretationThread(self, pplIntervalDict)
        self.thread.finishSignal.connect(self.stopGif)
        self.thread.start()


    def stopGif(self, newResList):
        self.gifDialog.movie.stop()
        self.gifDialog.hide()
        self.btn4.setDisabled(False)
        self.interpreted = True
        self.researchsList = newResList
        for i, res in enumerate(self.researchsList):
            if res.warning:
                self.listModel.item(i, 0).setIcon(self.style().standardIcon(10))
            else:
                self.listModel.item(i, 0).setIcon(self.style().standardIcon(45))

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
        print('begin', self.researchsList[tempInd].ro, self.researchsList[tempInd].ppl)
        layers = self.researchsList[tempInd].layer
        layers = ', '.join(layers)
        vdp = self.researchsList[tempInd].vdp
        elongation = self.researchsList[tempInd].vdpElong
        if self.interpreted:
            if self.researchsList[tempInd].tableInd == 0:
                self.RB1.setChecked(True)
            else:
                self.RB2.setChecked(True)
            ros = []
            ppls = []
            self.calctTable1.setModel(self.researchsList[tempInd].tableModels[0])
            self.calcTable2.setModel(self.researchsList[tempInd].tableModels[1])
            for i in range(7):
                self.calcTable2.setColumnWidth(i, 75)
                self.calctTable1.setColumnWidth(i, 75)

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

            if (self.focusWidget() == self.calctTable1 and self.RB1.isChecked()) or \
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
        for i, res in enumerate(self.researchsList):
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
            for res in self.researchsList:
                print(res.resID)
                self.insertResearchDataToDB(res)

    # def make_model(self, data_polka, j):
    #     model = QtGui.QStandardItemModel(len(data_polka), 7)
    #     model.setHorizontalHeaderLabels(['Depth', 'Elongation', 'Pressure', 'Temperature', 'Density', 'Fluid type', ''])
    #     densities = [0]
    #     types = ['']
    #     for row in range(len(data_polka)):
    #         for col in range(7):
    #             if col in range(4):
    #                 item = QtGui.QStandardItem(str(data_polka[row][col]))
    #             elif col == 4:
    #                 if row != 0:
    #                     ro = round((data_polka[row][2] - data_polka[row - 1][2]) /
    #                                (data_polka[row][0] - data_polka[row][1] - data_polka[row - 1][0] +
    #                                 data_polka[row - 1][
    #                                     1]) * 10, 3)
    #                     densities.append(ro)
    #                     item = QtGui.QStandardItem(str(ro))
    #
    #             elif col == 5:
    #                 if row != 0:
    #                     if ro < 0.7:
    #                         item = QtGui.QStandardItem("Gas")
    #                     elif 0.7 < ro < 0.98:
    #                         item = QtGui.QStandardItem("Oil")
    #                     else:
    #                         item = QtGui.QStandardItem("Water")
    #                     types.append(item.text())
    #             elif col == 6:
    #                 if row != 0:
    #                     item = QtGui.QStandardItem()
    #                     item.setCheckable(True)
    #             item.setTextAlignment(QtCore.Qt.AlignCenter)
    #             model.setItem(row, col, item)
    #     kt = round(len(data_polka) / 3) + 1
    #     calc_list = densities[-kt:]
    #     types_list = types[-kt:]
    #     ref_type = None
    #     if types_list[-1] == types_list[-2] and 0.7 < calc_list[-1] < 1.25 and 0.7 < calc_list[-2] < 1.25:
    #         ref_type = types_list[-1]
    #     elif types_list[-1] != types_list[-2] and 0.7 < calc_list[-1] < 1.25 and 0.7 < calc_list[-2] < 1.25:
    #         if types_list[-2] == types_list[-3] and 0.7 < calc_list[-2] < 1.25 and 0.7 < calc_list[-3] < 1.25:
    #             ref_type = types_list[-2]
    #         elif types_list[-2] != types_list[-3] and 0.7 < calc_list[-2] < 1.25 and 0.7 < calc_list[-3] < 1.25:
    #             if types_list[-3] == types_list[-4] and 0.7 < calc_list[-3] < 1.25 and 0.7 < calc_list[-4] < 1.25:
    #                 ref_type = types_list[-3]
    #         else:
    #             max_num = 0
    #             types_set = set(types_list)
    #             for fluid_type in types_set:
    #                 num = types_list.count(fluid_type)
    #                 if num > max_num:
    #                     max_num = num
    #                     ref_type = fluid_type
    #     else:
    #         max_num = 0
    #         types_set = set(types_list)
    #         for fluid_type in types_set:
    #             num = types_list.count(fluid_type)
    #             if num > max_num:
    #                 max_num = num
    #                 ref_type = fluid_type
    #     if ref_type == "Water":
    #         target = 1.16
    #         calc_list = [ro for ro in calc_list if ro >= 0.98]
    #     else:
    #         target = 0.88
    #         calc_list = [ro for ro in calc_list if 0.7 < ro < 0.98]
    #     if len(calc_list) % 2 == 0:
    #         med_ro2 = median(calc_list)
    #         indic = False
    #         for i in range(len(calc_list) - 1):
    #             if calc_list[i] < med_ro2 < calc_list[i + 1]:
    #                 med_ro1 = calc_list[i]
    #                 med_ro3 = calc_list[i + 1]
    #                 indic = True
    #             if i == len(calc_list) - 2 and indic == False:
    #                 med_ro1 = med_ro2
    #                 med_ro3 = med_ro2
    #         cond1 = (max(target, med_ro1) - min(target, med_ro1)) / max(target, med_ro1)
    #         cond2 = (max(target, med_ro2) - min(target, med_ro2)) / max(target, med_ro2)
    #         cond3 = (max(target, med_ro3) - min(target, med_ro3)) / max(target, med_ro3)
    #         if cond1 == min(cond1, cond2, cond3):
    #             med_ro = med_ro1
    #         elif cond2 == min(cond1, cond2, cond3):
    #             med_ro = med_ro2
    #         else:
    #             med_ro = med_ro3
    #     else:
    #         med_ro = median(calc_list)
    #     final_ro = []
    #     for row in range(len(data_polka) - kt, len(data_polka)):
    #         m1 = max(float(model.index(row, 4).data()), med_ro)
    #         m2 = min(float(model.index(row, 4).data()), med_ro)
    #         if model.index(row, 5).data() == ref_type and (m1 - m2) / m1 < 0.08:
    #             model.item(row, 6).setCheckState(2)
    #             final_ro.append(float(model.item(row, 4).text()))
    #     if 0.8 > mean(final_ro) or 1.2 < mean(final_ro) or 0.8 > mean(final_ro) or 1.2 < mean(final_ro):
    #         self.listmodel.item(j, 0).setIcon(self.style().standardIcon(10))
    #     else:
    #         self.listmodel.item(j, 0).setIcon(self.style().standardIcon(45))
    #     man_abs_depth = float(model.item(len(data_polka) - 1, 0).text()) - float(
    #         model.item(len(data_polka) - 1, 1).text())
    #     vdp_abs_depth = self.ResearchsList[j].vdp - self.ResearchsList[j].vdpElong
    #     delta = vdp_abs_depth - man_abs_depth
    #     ppl = round(float(model.item(len(data_polka) - 1, 2).text()) + delta * mean(final_ro) / 10, 3)
    #     return model, round(mean(final_ro), 3), ppl, ref_type, len(final_ro)



    # def interpretation(self):
    #     self.now = datetime.datetime.now()
    #     self.movie.start()
    #     self.gifdialog.show()
    #     self.thread = GifThread(self, alt=True)
    #     self.thread.data = [res.data for res in self.ResearchsList]
    #     self.thread.incl = [res.incl for res in self.ResearchsList]
    #     self.thread.delta = int(self.deltaEdit.text())
    #     self.thread.finishSignal.connect(self.stop_gif)
    #     self.thread.start()


    # def get_files_paths_and_read_files(self):
    #     self.interpreted = False
    #     self.listmodel = QtGui.QStandardItemModel()
    #     self.listmodel.setColumnCount(1)
    #     paths, _ = QtWidgets.QFileDialog.getOpenFileNames(parent=self,
    #                                                       filter="EXCEL (*.xls *.xlsx);;All (*)",
    #                                                       caption="Выберите EXCEL с замером манометра и датчика спуска",
    #                                                       directory=QtCore.QDir.currentPath())
    #     self.ResearchsList = []
    #     self.listmodel.clear()
    #     for i, path in enumerate(paths):
    #         data = pd.read_excel(path).iloc[:, 0:7]
    #         fw = path[path.rfind("/") + 1:path.rfind(".")]
    #         field, well = fw.split(" ")
    #         exec("res_{}=ppl(\"{}\", \"{}\", {})".format(i, field, well, "data"))
    #         exec("self.ResearchsList.append(res_{})".format(i))
    #         item = QtGui.QStandardItem(fw)
    #         self.listmodel.appendRow(item)
    #     self.testlist.setModel(self.listmodel)
    #     self.btn2.setDisabled(False)
    #     self.btn3.setDisabled(False)



    # def stop_gif(self, dwd, st, td, cd=None):
    #     self.dots_with_data = dwd
    #     self.support_times = st
    #     temp_dfs, centralDots = td, cd
    #     for i, data in enumerate(self.dots_with_data):
    #         model_pair = []
    #         ro_pair = []
    #         ppl_pair = []
    #         f_type_pair = []
    #         checks_pair = []
    #         for polka in data:
    #             temp_model, ro, ppl, f_type, checks = self.make_model(polka, i)
    #             model_pair.append(temp_model)
    #             ro_pair.append(ro)
    #             ppl_pair.append(ppl)
    #             f_type_pair.append(f_type)
    #             checks_pair.append(checks)
    #         self.ResearchsList[i].dividerPoints = cd[i]
    #         self.ResearchsList[i].table_models = model_pair
    #         self.ResearchsList[i].checks = checks_pair
    #         self.ResearchsList[i].supTimes = self.support_times[i]
    #         self.ResearchsList[i].dataFromSupTimes = data
    #         if f_type_pair[0] == "Water":
    #             target = 1.16
    #         else:
    #             target = 0.88
    #         cond1 = (max(target, ro_pair[0]) - min(target, ro_pair[0])) / max(target, ro_pair[0])
    #         cond2 = (max(target, ro_pair[1]) - min(target, ro_pair[1])) / max(target, ro_pair[1])
    #         if self.ResearchsList[i].checks[0] > self.ResearchsList[i].checks[1]:
    #             self.ResearchsList[i].table_ind = 0
    #             self.ResearchsList[i].ro = ro_pair[0]
    #             self.ResearchsList[i].ppl = ppl_pair[0]
    #         elif self.ResearchsList[i].checks[0] < self.ResearchsList[i].checks[1]:
    #             self.ResearchsList[i].table_ind = 1
    #             self.ResearchsList[i].ro = ro_pair[1]
    #             self.ResearchsList[i].ppl = ppl_pair[1]
    #         else:
    #             if cond1 > cond2:
    #                 self.ResearchsList[i].table_ind = 0
    #                 self.ResearchsList[i].ro = ro_pair[0]
    #                 self.ResearchsList[i].ppl = ppl_pair[0]
    #             else:
    #                 self.ResearchsList[i].table_ind = 1
    #                 self.ResearchsList[i].ro = ro_pair[1]
    #                 self.ResearchsList[i].ppl = ppl_pair[1]
    #
    #         if f_type_pair[0] != f_type_pair[1]:
    #             self.listmodel.item(i, 0).setIcon(self.style().standardIcon(10))
    #
    #     for i, d in enumerate(temp_dfs):
    #         t1 = d[0].iloc[:, [0, 1, 2]]
    #         t2 = d[1].iloc[:, [0, 1, 2]]
    #         if t1.isnull().sum().sum() > 0:
    #             t1.dropna(inplace=True, how='all')
    #         if t2.isnull().sum().sum() > 0:
    #             t2.dropna(inplace=True, how='all')
    #         pres = pd.concat([t1, t2], axis=0)
    #         pres.reset_index(inplace=True, drop=True)
    #         t3 = d[0].iloc[:, [3, 4]]
    #         t4 = d[1].iloc[:, [3, 4]]
    #         if t3.isnull().sum().sum() > 0:
    #             t3.dropna(inplace=True, how='all')
    #         if t4.isnull().sum().sum() > 0:
    #             t4.dropna(inplace=True, how='all')
    #         dep = pd.concat([t3, t4], axis=0)
    #         dep.reset_index(inplace=True, drop=True)
    #         temp = pd.concat([pres, dep], axis=1)
    #         self.ResearchsList[i].final_data = temp
    #         temp_pw = PlotWidget2()
    #         self.ResearchsList[i].final_fig = temp_pw.plot(temp, save=True)
    #     for res in self.ResearchsList:
    #         res.calc_avg_temp_gradient()
    #         res.calc_phase_borders()
    #         res.determine_static_level()
    #     self.interpreted = True
    #     self.movie.stop()
    #     self.gifdialog.hide()
    #     self.btn4.setDisabled(False)
    #     print(datetime.datetime.now() - self.now)

class gifDialog(QWidget):
    def __init__(self):
        super(gifDialog, self).__init__()
        self.setFixedSize(252, 252)
        mLabel = QLabel(self)
        self.movie = QMovie()
        self.movie.setFileName(os.getcwd() + r"\mgc.gif")
        mLabel.setMovie(self.movie)
        mLabel.setFixedSize(252, 252)
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)


# class gifThread(QtCore.QThread):
#     def __init__(self, parent=None):
#         QtCore.QThread.__init__(self, parent)
#         print("INIT")
#
#         self.gifDialog = gifDialog()
#
#     def run(self):
#         print("SHOW1")
#         self.gifDialog.show()
#         self.gifDialog.movie.start()
#         print("SHOW2")


class interpretationThread(QtCore.QThread):
    finishSignal = pyqtSignal(object)

    def __init__(self, parent=None, pplIntervalDict=None):
        QtCore.QThread.__init__(self, parent)
        self.pplIntervalDict = pplIntervalDict


    def run(self):
        newResearchList = []
        self.pModel = joblib.load('rfc_model_pres.pkl')
        self.dModel = joblib.load('rfc_model_depths.pkl')
        for res in list(self.pplIntervalDict.keys()):
            interval = self.pplIntervalDict[res]
            newResearchList.append(res.interpret(interval, self.pModel, self.dModel))
        self.finishSignal.emit(newResearchList)


# class GifThread(QtCore.QThread):
#     finishSignal = pyqtSignal(object, object, object, object)
#
#     def __init__(self, parent=None,incl=None, alt=False, delta = 100):
#         QtCore.QThread.__init__(self, parent)
#         self.data = None
#         self.alt = alt
#         self.incl = incl
#         self.delta = delta
#
#     def run(self):
#         if self.alt:
#             ai = AutoInterpretation(self.data,incl=self.incl, alt=True, delta=self.delta)
#         else:
#             ai = AutoInterpretation(self.data,incl=self.incl)
#         dwd, st = ai.zips()
#         td, cd = ai.bias_and_splitting()
#         self.finishSignal.emit(dwd, st, td, cd)


app = QApplication(sys.argv)
MW = MainWindow()
MW.setWindowTitle("Интерпретатор 9000")
MW.show()
sys.exit(app.exec_())
