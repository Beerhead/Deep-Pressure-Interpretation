import sys
import warnings
import datetime
from PyQt5 import QtWidgets
from statistics import median, mean
from reportlab.pdfgen import canvas

from Other import *
from Interpretation import *
from Plot import *
from Report import *
warnings.filterwarnings('ignore')

#Конкретная скважина
#фильтр по месторождениям

class MainWindow(QMainWindow):

    def __init__(self):

        super(MainWindow, self).__init__()
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        self.ResearchsList = []
        self.initUi()
        self.connectUi()
        self.interpreted = False
        self.dots_with_data = None
        self.support_times = None

    def initUi(self):
        self.testlist = QListView()
        self.testlist.setMaximumWidth(250)
        self.testlist.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.centralWidget = QWidget(self)
        self.HL = QHBoxLayout(self.centralWidget)  # основной горизонтальный лэйаут
        self.VL1 = QVBoxLayout()  # вертикал лэй №1
        self.VL2 = QVBoxLayout()  # вертикал лэй №2
        self.plotWidget = PlotWidget()
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        self.plotWidget2 = PlotWidget2()
        self.rb1 = QRadioButton("По спуску")
        self.rb2 = QRadioButton("По подъему")
        self.hltables = QHBoxLayout()  # лэй для двух таблиц
        self.calctable1 = QTableView()
        self.calctable2 = QTableView()
        self.calctable1.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.calctable2.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.reslabel = QLabel('Данные по скважине')
        self.hltables.addWidget(self.calctable1)
        self.hltables.addWidget(self.calctable2)
        self.label_ch_box_lay = QVBoxLayout()  # лэй для лэйбла и чекбоксов
        self.ch_box_lay = QHBoxLayout()
        self.ch_box_lay.addWidget(self.rb1)
        self.ch_box_lay.addWidget(self.rb2)
        self.ch_box_lay.setAlignment(Qt.AlignCenter)
        self.label_ch_box_lay.addLayout(self.ch_box_lay)
        self.label_ch_box_lay.addWidget(self.reslabel)
        self.hltables.addLayout(self.label_ch_box_lay)
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

        # добавка 2 графика

        self.hl_graph_layout = QHBoxLayout()
        #self.hl_graph_layout.addWidget(self.plotWidget)
        self.hl_graph_layout.addWidget(self.plotWidget2)

        self.VL1.addLayout(self.hltables)
        self.VL1.addLayout(self.hl_graph_layout)

        self.VL2.addWidget(self.testlist)
        self.VL2.addLayout(self.GL)
        self.HL.addLayout(self.VL1)
        self.HL.addLayout(self.VL2)
        self.setCentralWidget(self.centralWidget)

        self.gifdialog = QtWidgets.QDialog()
        self.gifdialog.setFixedSize(252, 252)
        self.m_label = QLabel(self.gifdialog)
        self.movie = QMovie()
        self.movie.setFileName(os.getcwd() + r"\mgc.gif")
        self.m_label.setMovie(self.movie)
        self.gifdialog.setWindowFlag(QtCore.Qt.SplashScreen)

    def connectUi(self):

        self.btn1.clicked.connect(self.get_files_paths_and_read_files)
        self.btn2.clicked.connect(self.choose_researches)

        self.btn3.clicked.connect(self.interpretation)
        self.btn4.clicked.connect(self.reports)
        self.testlist.clicked.connect(self.graf)
        self.calctable1.clicked.connect(self.graf)
        self.calctable2.clicked.connect(self.graf)
        self.rb1.clicked.connect(self.rb1_clicked)
        self.rb2.clicked.connect(self.rb2_clicked)
        self.plotWidget2.btn_spusk_pressed_signal.connect(lambda: self.add_inf_line(0))
        self.plotWidget2.btn_podem_pressed_signal.connect(lambda: self.add_inf_line(1))
        self.plotWidget2.line_signal_to_main.connect(self.inf_line_moved)

    def receive_list(self, model_to_set, list_of_measuarements):
        print(model_to_set)
        print(list_of_measuarements)
        self.listmodel = model_to_set
        self.testlist.setModel(self.listmodel)
        self.ResearchsList = list_of_measuarements
        print(self.ResearchsList)
        self.btn3.setDisabled(False)


    def choose_researches(self):
        self.measurewidget = MeasurementsWidget(self)
        self.measurewidget.show()

    def add_inf_line(self, polka):
        temp_ind = self.testlist.selectedIndexes()[0].row()
        delta = (self.ResearchsList[temp_ind].sup_times[1][-1] - self.ResearchsList[temp_ind].sup_times[0][-1]) / 2
        medium_time = self.ResearchsList[temp_ind].sup_times[0][-1] + delta
        time, *_ = self.time_bind_inf_line(medium_time)
        line = self.plotWidget2.make_inf_line(time, polka)
        self.plotWidget2.plotItem.addItem(line)
        line.inf_line_signal.connect(self.plotWidget2.emitting_to_main)

    def inf_line_moved(self, line, start, stop):
        temp_ind = self.testlist.selectedIndexes()[0].row()
        stop = pd.Timestamp.fromtimestamp(stop)
        start = pd.Timestamp.fromtimestamp(start).round(freq='1S')
        sup_times = self.ResearchsList[temp_ind].sup_times[line.polka]
        true_false_list = [start == i.round(freq='1S') for i in sup_times]
        time, depth, elong, pres, temper = self.time_bind_inf_line(stop)
        try:
            index = true_false_list.index(True)
            self.ResearchsList[temp_ind].sup_times[line.polka][index] = time
            self.ResearchsList[temp_ind].data_from_sup_times[line.polka][index] = [depth, elong, pres, temper]
        except:
            self.ResearchsList[temp_ind].sup_times[line.polka].append(time)
            self.ResearchsList[temp_ind].data_from_sup_times[line.polka].append([depth, elong, pres, temper])
        self.ResearchsList[temp_ind].data_from_sup_times[line.polka].sort(key=lambda x: x[0])
        model = self.ResearchsList[temp_ind].table_models[line.polka]
        new_model, *_ = self.make_model(self.ResearchsList[temp_ind].data_from_sup_times[line.polka], temp_ind)
        # for row in range(1, model.rowCount()):
        #     new_model.item(row, 6).setCheckState(0)
        #     if model.item(row, 6).checkState() == 2:
        #         new_model.item(row, 6).setCheckState(2)
        self.ResearchsList[temp_ind].table_models[line.polka] = new_model
        self.graf()

    def time_bind_inf_line(self, time):
        temp_ind = self.testlist.selectedIndexes()[0].row()
        # по манометру
        time_series = self.ResearchsList[temp_ind].final_data.iloc[:, 0]
        time_series.dropna(inplace=True)
        ind = time_series.searchsorted(time)
        cond1 = (time_series[ind].timestamp() - time.timestamp()) / time_series[ind].timestamp()
        cond2 = (time_series[ind + 1].timestamp() - time.timestamp()) / time_series[ind].timestamp()
        if cond1 < cond2:
            final_ind = ind
        else:
            final_ind = ind + 1
        time = time_series[final_ind]
        pres = self.ResearchsList[temp_ind].final_data.iloc[final_ind, 1]
        temper = self.ResearchsList[temp_ind].final_data.iloc[final_ind, 2]
        # по СПС
        time_series = self.ResearchsList[temp_ind].final_data.iloc[:, 3]
        time_series.dropna(inplace=True)
        ind = time_series.searchsorted(time)
        cond1 = (time_series[ind].timestamp() - time.timestamp()) / time_series[ind].timestamp()
        cond2 = (time_series[ind + 1].timestamp() - time.timestamp()) / time_series[ind].timestamp()
        if cond1 < cond2:
            final_ind = ind
        else:
            final_ind = ind + 1
        depth = self.ResearchsList[temp_ind].final_data.iloc[final_ind, 4]
        elong = search_and_interpolate(self.ResearchsList[temp_ind].incl, depth)
        return time, round(depth, 0), round(elong, 2), round(pres, 2), round(temper, 2)

    def rb1_clicked(self):
        temp_ind = self.testlist.selectedIndexes()[0].row()
        self.ResearchsList[temp_ind].table_ind = 0

    def rb2_clicked(self):
        temp_ind = self.testlist.selectedIndexes()[0].row()
        self.ResearchsList[temp_ind].table_ind = 1

    def keyPressEvent(self, e):
        if e.key() == QtCore.Qt.Key_Delete and len(self.testlist.selectedIndexes()) > 0 and self.testlist.hasFocus():
            for i in self.testlist.selectedIndexes()[::-1]:
                self.listmodel.removeRow(i.row())
                self.ResearchsList.pop(i.row())

    def stop_gif(self, dwd, st, td):
        self.dots_with_data = dwd
        self.support_times = st
        temp_dfs = td
        for i, data in enumerate(self.dots_with_data):
            model_pair = []
            ro_pair = []
            ppl_pair = []
            f_type_pair = []
            checks_pair = []
            for polka in data:
                temp_model, ro, ppl, f_type, checks = self.make_model(polka, i)
                model_pair.append(temp_model)
                ro_pair.append(ro)
                ppl_pair.append(ppl)
                f_type_pair.append(f_type)
                checks_pair.append(checks)
            self.ResearchsList[i].table_models = model_pair
            self.ResearchsList[i].checks = checks_pair
            self.ResearchsList[i].sup_times = self.support_times[i]
            self.ResearchsList[i].data_from_sup_times = data
            if f_type_pair[0] == "Water":
                target = 1.16
            else:
                target = 0.88
            cond1 = (max(target, ro_pair[0]) - min(target, ro_pair[0])) / max(target, ro_pair[0])
            cond2 = (max(target, ro_pair[1]) - min(target, ro_pair[1])) / max(target, ro_pair[1])
            if self.ResearchsList[i].checks[0] > self.ResearchsList[i].checks[1]:
                self.ResearchsList[i].table_ind = 0
                self.ResearchsList[i].ro = ro_pair[0]
                self.ResearchsList[i].ppl = ppl_pair[0]
            elif self.ResearchsList[i].checks[0] < self.ResearchsList[i].checks[1]:
                self.ResearchsList[i].table_ind = 1
                self.ResearchsList[i].ro = ro_pair[1]
                self.ResearchsList[i].ppl = ppl_pair[1]
            else:
                if cond1 > cond2:
                    self.ResearchsList[i].table_ind = 0
                    self.ResearchsList[i].ro = ro_pair[0]
                    self.ResearchsList[i].ppl = ppl_pair[0]
                else:
                    self.ResearchsList[i].table_ind = 1
                    self.ResearchsList[i].ro = ro_pair[1]
                    self.ResearchsList[i].ppl = ppl_pair[1]

            if f_type_pair[0] != f_type_pair[1]:
                self.listmodel.item(i, 0).setIcon(self.style().standardIcon(10))

        for i, d in enumerate(temp_dfs):
            t1 = d[0].iloc[:, [0, 1, 2]]
            t2 = d[1].iloc[:, [0, 1, 2]]
            if t1.isnull().sum().sum() > 0:
                t1.dropna(inplace=True, how='all')
            if t2.isnull().sum().sum() > 0:
                t2.dropna(inplace=True, how='all')
            pres = pd.concat([t1, t2], axis=0)
            pres.reset_index(inplace=True, drop=True)
            t3 = d[0].iloc[:, [3, 4]]
            t4 = d[1].iloc[:, [3, 4]]
            if t3.isnull().sum().sum() > 0:
                t3.dropna(inplace=True, how='all')
            if t4.isnull().sum().sum() > 0:
                t4.dropna(inplace=True, how='all')
            dep = pd.concat([t3, t4], axis=0)
            dep.reset_index(inplace=True, drop=True)
            temp = pd.concat([pres, dep], axis=1)
            self.ResearchsList[i].final_data = temp
            temp_pw = PlotWidget()
            self.ResearchsList[i].final_fig = temp_pw.plot(temp, save=True)
        self.interpreted = True
        self.movie.stop()
        self.gifdialog.hide()
        self.btn4.setDisabled(False)
        print(datetime.datetime.now() - self.now)

    def graf(self):
        temp_ind = self.testlist.selectedIndexes()[0].row()
        layers = self.ResearchsList[temp_ind].layer
        layers = ''.join(layers)
        vdp = self.ResearchsList[temp_ind].vdp
        elongation = self.ResearchsList[temp_ind].vdp_elong
        if self.interpreted:
            if self.ResearchsList[temp_ind].table_ind == 0:
                self.rb1.setChecked(True)
            else:
                self.rb2.setChecked(True)
            ros = []
            ppls = []
            self.calctable1.setModel(self.ResearchsList[temp_ind].table_models[0])
            self.calctable2.setModel(self.ResearchsList[temp_ind].table_models[1])
            self.calctable1.setColumnWidth(0, 75)
            self.calctable1.setColumnWidth(1, 75)
            self.calctable1.setColumnWidth(2, 75)
            self.calctable1.setColumnWidth(3, 75)
            self.calctable1.setColumnWidth(4, 75)
            self.calctable1.setColumnWidth(5, 75)
            self.calctable1.setColumnWidth(6, 75)
            self.calctable2.setColumnWidth(0, 75)
            self.calctable2.setColumnWidth(1, 75)
            self.calctable2.setColumnWidth(2, 75)
            self.calctable2.setColumnWidth(3, 75)
            self.calctable2.setColumnWidth(4, 75)
            self.calctable2.setColumnWidth(5, 75)
            self.calctable2.setColumnWidth(6, 75)
            checked_sup_times = []
            for num, m in enumerate(self.ResearchsList[temp_ind].table_models):
                half_cst = []
                kt = 0
                ro = 0
                for i in range(1, m.rowCount()):
                    if i == m.rowCount() - 1:
                        man_abs_depth = float(m.item(i, 0).text()) - float(m.item(i, 1).text())
                        vdp_abs_depth = self.ResearchsList[temp_ind].vdp - self.ResearchsList[temp_ind].vdp_elong
                        delta = vdp_abs_depth - man_abs_depth
                    if m.item(i, 6).checkState() == 2:
                        half_cst.append(self.support_times[temp_ind][num][i])
                        ro += float(m.item(i, 4).text())
                        kt += 1
                ros.append(round(ro / kt, 3))
                ppls.append(round(float(m.item(m.rowCount() - 1, 2).text()) + delta * ros[-1] / 10, 3))
                checked_sup_times.append(half_cst)
            if 0.8 > ros[0] or 1.2 < ros[0] or 0.8 > ros[1] or 1.2 < ros[1]:
                self.listmodel.item(temp_ind, 0).setIcon(self.style().standardIcon(10))
            else:
                self.listmodel.item(temp_ind, 0).setIcon(self.style().standardIcon(45))
            if mean(ros) > 0.98:
                target = 1.16
            else:
                target = 0.88
            cond1 = (max(target, ros[0]) - min(target, ros[0])) / max(target, ros[0])
            cond2 = (max(target, ros[1]) - min(target, ros[1])) / max(target, ros[1])
            if len(checked_sup_times[0]) > len(checked_sup_times[1]):
                self.ResearchsList[temp_ind].table_ind = 0
                self.ResearchsList[temp_ind].ro = ros[0]
                self.ResearchsList[temp_ind].ppl = ppls[0]
            elif len(checked_sup_times[0]) < len(checked_sup_times[1]):
                self.ResearchsList[temp_ind].table_ind = 1
                self.ResearchsList[temp_ind].ro = ros[1]
                self.ResearchsList[temp_ind].ppl = ppls[1]
            else:
                if cond1 == min(cond1, cond2):
                    self.ResearchsList[temp_ind].table_ind = 0
                    self.ResearchsList[temp_ind].ro = ros[0]
                    self.ResearchsList[temp_ind].ppl = ppls[0]
                else:
                    self.ResearchsList[temp_ind].table_ind = 1
                    self.ResearchsList[temp_ind].ro = ros[1]
                    self.ResearchsList[temp_ind].ppl = ppls[1]
            self.plotWidget.plot(self.ResearchsList[temp_ind].final_data, checked_sup_times)
            self.reslabel.setText('Данные по скважине' + '\n' +
                                  'Глубина ВДП - ' + str(vdp) + '\n' +
                                  'Удлинение на ВДП - ' + str(elongation) + '\n' +
                                  'Пласт(ы) - ' + str(layers) + '\n' +
                                  'Плотность на спуске - ' + str(ros[0]) + '\n' +
                                  'Плотность на подъеме - ' + str(ros[1]) + '\n' +
                                  'Пластовое давление на ВДП по спуску - ' + str(ppls[0]) + '\n' +
                                  'Пластовое давление на ВДП по подъему - ' + str(ppls[1]))
            self.plotWidget2.plot(self.ResearchsList[temp_ind].final_data, checked_sup_times)
        else:
            self.plotWidget.plot(self.ResearchsList[temp_ind].data)
            self.reslabel.setText('Данные по скважине' + '\n' +
                                  'Глубина ВДП - ' + str(vdp) + '\n' +
                                  'Удлинение на ВДП - ' + str(elongation) + '\n' +
                                  'Пласт(ы) - ' + str(layers) + '\n')
            self.plotWidget2.plot(self.ResearchsList[temp_ind].data)

    def reports(self):

        fold = "\Reports\\"
        for i, res in enumerate(self.ResearchsList):
            filename = str(res.field) + " " + str(res.well_name) + '.pdf'
            filepath = os.getcwd() + fold + filename
            print(filepath)
            cvs = canvas.Canvas(filepath)
            fig = self.ResearchsList[i].final_fig
            fig.seek(0)
            model = self.ResearchsList[i].table_models[self.ResearchsList[i].table_ind]
            table_data = []
            checked = []
            columns = model.columnCount()
            rows = model.rowCount()
            for r in range(rows):
                row_to_list = []
                for c in range(columns - 1):
                    if not (r == 0 and (c == 4 or c == 5)):
                        row_to_list.append(model.item(r, c).text())
                if r != 0:
                    if model.item(r, 6).checkState() == 2:
                        checked.append(r)
                table_data.append(row_to_list)
            well = self.ResearchsList[i].well_name
            field = self.ResearchsList[i].field
            layer = ''.join(self.ResearchsList[i].layer)
            res_date = self.ResearchsList[i].research_date
            vdp = self.ResearchsList[i].vdp
            vdp_elongation = self.ResearchsList[i].vdp_elong
            ro = self.ResearchsList[i].ro
            ppl = self.ResearchsList[i].ppl
            pdf(cvs, well, field, layer, res_date, fig, vdp, vdp_elongation, ro, ppl, table_data, checked)

    def make_model(self, data_polka, j):
        model = QtGui.QStandardItemModel(len(data_polka), 7)
        model.setHorizontalHeaderLabels(['Depth', 'Elongation', 'Pressure', 'Temperature', 'Density', 'Fluid type', ''])
        densities = [0]
        types = ['']
        for row in range(len(data_polka)):
            for col in range(7):
                if col in range(4):
                    item = QtGui.QStandardItem(str(data_polka[row][col]))
                elif col == 4:
                    if row != 0:
                        ro = round((data_polka[row][2] - data_polka[row - 1][2]) /
                                   (data_polka[row][0] - data_polka[row][1] - data_polka[row - 1][0] +
                                    data_polka[row - 1][
                                        1]) * 10, 3)
                        densities.append(ro)
                        item = QtGui.QStandardItem(str(ro))

                elif col == 5:
                    if row != 0:
                        if ro < 0.7:
                            item = QtGui.QStandardItem("Gas")
                        elif 0.7 < ro < 0.98:
                            item = QtGui.QStandardItem("Oil")
                        else:
                            item = QtGui.QStandardItem("Water")
                        types.append(item.text())
                elif col == 6:
                    if row != 0:
                        item = QtGui.QStandardItem()
                        item.setCheckable(True)
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                model.setItem(row, col, item)
        kt = round(len(data_polka) / 3) + 1
        calc_list = densities[-kt:]
        types_list = types[-kt:]
        ref_type = None
        if types_list[-1] == types_list[-2] and 0.7 < calc_list[-1] < 1.25 and 0.7 < calc_list[-2] < 1.25:
            ref_type = types_list[-1]
        elif types_list[-1] != types_list[-2] and 0.7 < calc_list[-1] < 1.25 and 0.7 < calc_list[-2] < 1.25:
            if types_list[-2] == types_list[-3] and 0.7 < calc_list[-2] < 1.25 and 0.7 < calc_list[-3] < 1.25:
                ref_type = types_list[-2]
            elif types_list[-2] != types_list[-3] and 0.7 < calc_list[-2] < 1.25 and 0.7 < calc_list[-3] < 1.25:
                if types_list[-3] == types_list[-4] and 0.7 < calc_list[-3] < 1.25 and 0.7 < calc_list[-4] < 1.25:
                    ref_type = types_list[-3]
            else:
                max_num = 0
                types_set = set(types_list)
                for fluid_type in types_set:
                    num = types_list.count(fluid_type)
                    if num > max_num:
                        max_num = num
                        ref_type = fluid_type
        else:
            max_num = 0
            types_set = set(types_list)
            for fluid_type in types_set:
                num = types_list.count(fluid_type)
                if num > max_num:
                    max_num = num
                    ref_type = fluid_type
        if ref_type == "Water":
            target = 1.16
            calc_list = [ro for ro in calc_list if ro >= 0.98]
        else:
            target = 0.88
            calc_list = [ro for ro in calc_list if 0.7 < ro < 0.98]
        if len(calc_list) % 2 == 0:
            med_ro2 = median(calc_list)
            indic = False
            for i in range(len(calc_list) - 1):
                if calc_list[i] < med_ro2 < calc_list[i + 1]:
                    med_ro1 = calc_list[i]
                    med_ro3 = calc_list[i + 1]
                    indic = True
                if i == len(calc_list) - 2 and indic == False:
                    med_ro1 = med_ro2
                    med_ro3 = med_ro2
            cond1 = (max(target, med_ro1) - min(target, med_ro1)) / max(target, med_ro1)
            cond2 = (max(target, med_ro2) - min(target, med_ro2)) / max(target, med_ro2)
            cond3 = (max(target, med_ro3) - min(target, med_ro3)) / max(target, med_ro3)
            if cond1 == min(cond1, cond2, cond3):
                med_ro = med_ro1
            elif cond2 == min(cond1, cond2, cond3):
                med_ro = med_ro2
            else:
                med_ro = med_ro3
        else:
            med_ro = median(calc_list)
        final_ro = []
        for row in range(len(data_polka) - kt, len(data_polka)):
            m1 = max(float(model.index(row, 4).data()), med_ro)
            m2 = min(float(model.index(row, 4).data()), med_ro)
            if model.index(row, 5).data() == ref_type and (m1 - m2) / m1 < 0.08:
                model.item(row, 6).setCheckState(2)
                final_ro.append(float(model.item(row, 4).text()))
        if 0.8 > mean(final_ro) or 1.2 < mean(final_ro) or 0.8 > mean(final_ro) or 1.2 < mean(final_ro):
            self.listmodel.item(j, 0).setIcon(self.style().standardIcon(10))
        else:
            self.listmodel.item(j, 0).setIcon(self.style().standardIcon(45))
        man_abs_depth = float(model.item(len(data_polka) - 1, 0).text()) - float(
            model.item(len(data_polka) - 1, 1).text())
        vdp_abs_depth = self.ResearchsList[j].vdp - self.ResearchsList[j].vdp_elong
        delta = vdp_abs_depth - man_abs_depth
        ppl = round(float(model.item(len(data_polka) - 1, 2).text()) + delta * mean(final_ro) / 10, 3)
        return model, round(mean(final_ro), 3), ppl, ref_type, len(final_ro)

    def interpretation(self):
        self.now = datetime.datetime.now()
        self.movie.start()
        self.gifdialog.show()
        self.thread = GifThread(self, alt=True)
        self.thread.data = [res.data for res in self.ResearchsList]
        self.thread.incl = [res.incl for res in self.ResearchsList]
        self.thread.finish_signal.connect(self.stop_gif)
        self.thread.start()

    def get_files_paths_and_read_files(self):
        self.interpreted = False
        self.listmodel = QtGui.QStandardItemModel()
        self.listmodel.setColumnCount(1)
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(parent=self,
                                                          filter="EXCEL (*.xls *.xlsx);;All (*)",
                                                          caption="Выберите EXCEL с замером манометра и датчика спуска",
                                                          directory=QtCore.QDir.currentPath())
        self.ResearchsList = []
        self.listmodel.clear()
        for i, path in enumerate(paths):
            data = pd.read_excel(path).iloc[:, 0:7]
            fw = path[path.rfind("/") + 1:path.rfind(".")]
            field, well = fw.split(" ")
            exec("res_{}=Ppl(\"{}\", \"{}\", {})".format(i, field, well, "data"))
            exec("self.ResearchsList.append(res_{})".format(i))
            item = QtGui.QStandardItem(fw)
            self.listmodel.appendRow(item)
        self.testlist.setModel(self.listmodel)
        self.btn2.setDisabled(False)
        self.btn3.setDisabled(False)


class GifThread(QtCore.QThread):
    finish_signal = pyqtSignal(object, object, object)

    def __init__(self, parent=None,incl=None, alt=False):
        QtCore.QThread.__init__(self, parent)
        self.data = None
        self.alt = alt
        self.incl = incl

    def run(self):
        if self.alt:
            ai = AutoInterpretation(self.data,incl=self.incl, alt=True)
        else:
            ai = AutoInterpretation(self.data,incl=self.incl)
        dwd, st = ai.zips()
        td = ai.bias_and_splitting()
        self.finish_signal.emit(dwd, st, td)


app = QApplication(sys.argv)
MW = MainWindow()
MW.setWindowTitle("Интерпретация Рпл")
MW.show()
sys.exit(app.exec_())
