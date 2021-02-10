from contextlib import contextmanager
from PyQt5 import QtSql, QtGui
from PyQt5.Qt import *
from Plot import PlotWidget2
import cx_Oracle
import base64

from itertools import zip_longest

class MeasurementsWidget(QWidget):

    def __init__(self):
        super(MeasurementsWidget, self).__init__()
        self.initUi()
        self.connectUi()

    def initUi(self):
        self.HL = QHBoxLayout(self)
        self.measurelistview = QListView()
        self.date_and_button_layout = QVBoxLayout()
        self.searchbtn = QPushButton('Найти замеры')
        self.addbtn = QPushButton('->')
        self.rmvbtn = QPushButton('<-')
        self.to_analysis_btn = QPushButton('Добавить в анализ выбранное')
        self.startdate = QDateEdit()
        self.startdate.setCalendarPopup(True)
        self.startdate.setDate(QDate.currentDate().addDays(-1))
        self.enddate = QDateEdit()
        self.enddate.setCalendarPopup(True)
        self.enddate.setDate(QDate.currentDate())
        self.fieldcombobox = QComboBox()
        self.fieldcombobox.addItems(self.get_field_names())
        self.wellcombobox = QComboBox()
        self.wellcombobox.addItem('Все скважины')
        self.date_and_button_layout.addWidget(self.fieldcombobox)
        self.date_and_button_layout.addWidget(self.wellcombobox)
        self.date_and_button_layout.addWidget(self.startdate)
        self.date_and_button_layout.addWidget(self.enddate)
        self.date_and_button_layout.addWidget(self.searchbtn)
        self.date_and_button_layout.addWidget(self.addbtn)
        self.date_and_button_layout.addWidget(self.rmvbtn)
        self.chosenmeasurelistview = QListView()
        self.plot = PlotWidget2()
        self.HL.addWidget(self.measurelistview)
        self.HL.addLayout(self.date_and_button_layout)
        self.HL.addWidget(self.chosenmeasurelistview)
        self.HL.addWidget(self.plot)
        self.measurelistview.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.chosenmeasurelistview.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list_model_1 = QtGui.QStandardItemModel()
        self.list_model_1.setColumnCount(1)
        self.list_model_2 = QtGui.QStandardItemModel()
        self.list_model_2.setColumnCount(1)
        self.measurelist = []
        self.chosenmeasurelist = []
        self.currentfield = None
        self.currentfieldid = None
        self.currentwell = None
        self.currentwellid = None

    def connectUi(self):
        self.searchbtn.clicked.connect(self.search_for_measurements_from_db)
        self.addbtn.clicked.connect(self.add_measurement_to_analysis)
        self.rmvbtn.clicked.connect(self.remove_measurement_from_analysis)
        self.fieldcombobox.currentIndexChanged.connect(self.get_well_names)
        self.wellcombobox.currentIndexChanged.connect(self.set_current_well)


    def set_current_well(self):
        self.currentwell = self.wellcombobox.currentText()
        self.currentwellid = None

    def get_field_names(self):
        with sql_query() as connection:
            cursor = connection.cursor()
            cursor.execute('''SELECT ots_bn.sosfield.fldname from  ots_bn.sosfield''')
            rows = [field[0] for field in cursor.fetchall()]
            rows.insert(0, 'Все площади')
            return rows

    def get_well_names(self):
        self.wellcombobox.clear()
        if self.fieldcombobox.currentIndex() != 0:
            with sql_query() as connection:
                cursor = connection.cursor()
                self.currentfield = self.fieldcombobox.currentText()
                cursor.execute('''SELECT ots_bn.sosfield.fldid from  ots_bn.sosfield
                                WHERE ots_bn.sosfield.fldname = :field''', field = self.currentfield)
                self.currentfieldid = cursor.fetchone()[0]
                cursor.execute('''SELECT ots_bn.soswell.welname from ots_bn.soswell 
                                WHERE ots_bn.soswell.welfieldid = :fieldid''', fieldid = self.currentfieldid)
                rows = [field[0] for field in cursor.fetchall()]
                rows.insert(0, 'Все скважины')
                self.wellcombobox.addItems(rows)
        else:
            self.wellcombobox.addItem('Все скважины')
            self.currentwell = None
            self.currentwellid = None

    def search_for_measurements_from_db(self):
        if self.fieldcombobox.currentIndex() == 0:
            with sql_query() as connection:
                cursor = connection.cursor()
                if  self.startdate.date() != self.enddate.date():
                    cursor.execute(''' SELECT ots_bn.sosmeasurement.meswellid,
                                    ots_bn.sosmeasurementmtmeter.mtinterval,
                                    ots_bn.sosmeasurementmtmeter.mtdate,
                                    ots_bn.sosmeasurementmtmeter.mtpressure,
                                    ots_bn.sosmeasurementmtmeter.mttemperature,
                                    ots_bn.sosmeasurementmtmeter.mtcount,
                                    ots_bn.sosmeasurementmtmeter.mtdepthstartdate,
                                    ots_bn.sosmeasurementmtmeter.mtdepthinterval,
                                    ots_bn.sosmeasurementmtmeter.mtdepthdate,
                                    ots_bn.sosmeasurementmtmeter.mtdepth
                                    from ots_bn.sosmeasurementmtmeter
                                    INNER JOIN ots_bn.sosmeasurement
                                    ON ots_bn.sosmeasurementmtmeter.mtmeasurementid = ots_bn.sosmeasurement.mesid
                                    WHERE (ots_bn.sosmeasurement.messtartdate >= :startdate) and
                                          (ots_bn.sosmeasurement.messtartdate <= :enddate)''',
                                   startdate=self.startdate.date().toString(Qt.LocalDate),
                                   enddate=self.enddate.date().toString(Qt.LocalDate))
                else:
                    cursor.execute(''' SELECT ots_bn.sosmeasurement.meswellid,
                                    ots_bn.sosmeasurementmtmeter.mtinterval,
                                    ots_bn.sosmeasurementmtmeter.mtdate,
                                    ots_bn.sosmeasurementmtmeter.mtpressure,
                                    ots_bn.sosmeasurementmtmeter.mttemperature,
                                    ots_bn.sosmeasurementmtmeter.mtcount,
                                    ots_bn.sosmeasurementmtmeter.mtdepthstartdate,
                                    ots_bn.sosmeasurementmtmeter.mtdepthinterval,
                                    ots_bn.sosmeasurementmtmeter.mtdepthdate,
                                    ots_bn.sosmeasurementmtmeter.mtdepth
                                    from ots_bn.sosmeasurementmtmeter
                                    INNER JOIN ots_bn.sosmeasurement
                                    ON ots_bn.sosmeasurementmtmeter.mtmeasurementid = ots_bn.sosmeasurement.mesid
                                    WHERE (ots_bn.sosmeasurement.messtartdate >= :startdate) and
                                          (ots_bn.sosmeasurement.messtartdate < :enddate)''',
                                    startdate = self.startdate.date().toString(Qt.LocalDate),
                                    enddate = self.enddate.date().addDays(1).toString(Qt.LocalDate))
                rows = cursor.fetchall()

                print(rows[0])
                temp = None
                print('!!!!!')
                print(rows[0][2].read())
                print('!!!!!')
                print(base64.b64decode(rows[0][2].read()))
                print('!!!!!')
                print(base64.b64decode(rows[0][2].read()).decode('cp1251'))

                # print(rows[0][2].size())
                # print(rows[0][3].size())
                # print(rows[0][4].size())
                # print('!!!!!!!!!!!!!')
                # for i, j, k in zip_longest(rows[0][2].read(), rows[0][3].read(), rows[0][4].read(), fillvalue= '!'):
                #     print(i, end=" : ")
                #     print(j, end=" : ")
                #     print(k)




    def add_measurement_to_analysis(self):
        pass

    def remove_measurement_from_analysis(self):
        pass


def search_and_interpolate(searching_array, x, xleft=True):
    if xleft:
        column1 = searching_array.iloc[:, 0].to_list()
        column2 = searching_array.iloc[:, 1].to_list()
    else:
        column2 = searching_array.iloc[:, 0].to_list()
        column1 = searching_array.iloc[:, 1].to_list()
    ind = 0
    percent = 0
    indicator = False
    if x in column1:
        indicator = True
        if column1.count(x) == 1:
            ind = column1.index(x)
        else:
            same_indexes = []
            for i in range(0, len(column1)):
                if column1[i] == x:
                    same_indexes.append(i)
            ind = int(sum(same_indexes) / len(same_indexes))

    else:
        if column1[-1] > column1[0]:
            for i in range(0, len(column1) - 1):
                if column1[i] < x < column1[i + 1]:
                    ind = i
                    percent = (x - column1[i]) / (column1[i + 1] - column1[i])
        else:
            for i in range(1, len(column1)):
                if column1[i - 1] > x > column1[i]:
                    ind = i - 1
                    percent = (x - column1[i - 1]) / (column1[i - 1] - column1[i])

    if indicator:
        return column2[ind]
    else:
        return column2[ind] + abs((column2[ind] - column2[ind + 1]) * percent)


@contextmanager
def sql_query():
    username = 'Shusharin'
    userpwd = 'Shusharin555'
    ip = 'oilteamsrv.bashneft.ru'
    port = 1521
    service_name = 'OTS'
    connection = cx_Oracle.connect(user=username,
                                   password=userpwd,
                                   dsn=cx_Oracle.makedsn(ip, port, service_name=service_name))
    yield connection
    connection.close()


@contextmanager
def sql_query_old(db_name):
    con = QtSql.QSqlDatabase.addDatabase('QSQLITE')
    con.setDatabaseName(db_name)
    con.open()
    query = QtSql.QSqlQuery()
    msg = ''
    try:
        yield query
    except Exception:
        msg = "Something gone wrong"
    finally:
        if msg:
            print(msg)
        con.close()

