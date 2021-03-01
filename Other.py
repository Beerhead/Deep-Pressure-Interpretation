from contextlib import contextmanager
from PyQt5 import QtSql, QtGui
from PyQt5.Qt import *
from Plot import PlotWidget2
import cx_Oracle
import pylzma
from sys import byteorder
import numpy as np
import pandas as pd
import datetime

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
                                WHERE ots_bn.sosfield.fldname = :field''', field=self.currentfield)
                self.currentfieldid = cursor.fetchone()[0]
                cursor.execute('''SELECT ots_bn.soswell.welname from ots_bn.soswell 
                                WHERE ots_bn.soswell.welfieldid = :fieldid''', fieldid=self.currentfieldid)
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
                startdate = self.startdate.date().toString(Qt.LocalDate)
                if self.startdate.date() != self.enddate.date():
                   enddate=self.enddate.date().toString(Qt.LocalDate)
                else:
                   enddate=self.enddate.date().addDays(1).toString(Qt.LocalDate)
                cursor.execute(''' SELECT ots_bn.sosmeasurement.meswellid,
                                                    ots_bn.sosmeasurementmtmeter.mtinterval,
                                                    ots_bn.sosmeasurementmtmeter.mtdate,
                                                    ots_bn.sosmeasurementmtmeter.mtpressure,
                                                    ots_bn.sosmeasurementmtmeter.mttemperature,
                                                    ots_bn.sosmeasurementmtmeter.mtcount,
                                                    ots_bn.sosmeasurementmtmeter.mtdepthstartdate,
                                                    ots_bn.sosmeasurementmtmeter.mtdepthinterval,
                                                    ots_bn.sosmeasurementmtmeter.mtdepthdate,
                                                    ots_bn.sosmeasurementmtmeter.mtdepth,
                                                    ots_bn.sosmeasurement.messtartdate
                                                    from ots_bn.sosmeasurementmtmeter
                                                    INNER JOIN ots_bn.sosmeasurement
                                                    ON ots_bn.sosmeasurementmtmeter.mtmeasurementid = ots_bn.sosmeasurement.mesid
                                                    WHERE (ots_bn.sosmeasurement.messtartdate >= :startdate) and
                                                          (ots_bn.sosmeasurement.messtartdate <= :enddate)''',
                               startdate=startdate,
                               enddate=enddate)
                rows = cursor.fetchall()
                print(rows[0])
                well_name, field_name = self.get_well_field_and_number(rows[0][0]) # Скважина

                pressure_BLOB = rows[0][3].read() # Давление
                bytes_len = int.from_bytes(pressure_BLOB[3:7], byteorder=byteorder)
                pressure = pd.Series(np.frombuffer(pylzma.decompress(pressure_BLOB[7:],
                                                                     maxlength=bytes_len), dtype=np.dtype('f')))

                temperature_BLOB = rows[0][4].read() #Температура
                bytes_len = int.from_bytes(temperature_BLOB[3:7], byteorder=byteorder)
                temperature = pd.Series(np.frombuffer(pylzma.decompress(temperature_BLOB[7:],
                                                                        maxlength=bytes_len), dtype=np.dtype('f')))

                depth_BLOB = rows[0][9].read() #Глубина
                bytes_len = int.from_bytes(depth_BLOB[3:7], byteorder=byteorder)
                depth = pd.Series(np.frombuffer(pylzma.decompress(depth_BLOB[7:],
                                                                  maxlength=bytes_len), dtype=np.dtype('f')))

                mtStartDateTime = pd.to_datetime(rows[0][10], dayfirst=True) # Даты манометра
                BD_mt_delta=rows[0][1]
                mtTimeDelta = pd.Timedelta(minutes=BD_mt_delta.minute, seconds=BD_mt_delta.second)
                mt_dates = pd.Series((mtStartDateTime+mtTimeDelta*i for i in range(rows[0][5])))

                date_depth_BLOB = rows[0][8].read() #даты глубин
                bytes_len = int.from_bytes(date_depth_BLOB[3:7], byteorder=byteorder)
                dates_depth = np.frombuffer(pylzma.decompress(date_depth_BLOB[7:], maxlength=bytes_len))
                dates_depth = pd.Series(pd.to_datetime(dates_depth, unit = 'd',
                                                       dayfirst=True, origin=pd.Timestamp('1900-01-01')))

                data = 
                # print(well_name, field_name)
                # print(mt_dates)
                # print(pressure)
                # print(temperature)
                # print(dates_depth)
                # print(depth)



    def add_measurement_to_analysis(self):
        pass

    def remove_measurement_from_analysis(self):
        pass

    def get_well_field_and_number(self, id):
        with sql_query() as connection:
            cursor = connection.cursor()
            cursor.execute(''' SELECT ots_bn.soswell.welname, ots_bn.soswell.welcupola 
                            FROM ots_bn.soswell WHERE welid = :wellid''', wellid=id)
            rows = cursor.fetchone()
        return rows[0], rows[1]

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


