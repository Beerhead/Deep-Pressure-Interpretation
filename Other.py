from contextlib import contextmanager
from PyQt5 import QtSql, QtGui
from PyQt5.Qt import *
from Plot import PlotWidget2
import Interpretation
import cx_Oracle
import pylzma
from sys import byteorder
import numpy as np
import pandas as pd
import joblib
import bisect

class MeasurementsWidget(QWidget):

    def __init__(self):
        super(MeasurementsWidget, self).__init__()
        self.initUi()
        self.connectUi()

    def initUi(self):
        self.setWindowTitle("Выбор замеров")
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
        self.leftmodel = QtGui.QStandardItemModel()
        self.leftmodel.setColumnCount(1)
        self.rightmodel = QtGui.QStandardItemModel()
        self.rightmodel.setColumnCount(1)
        self.notPpllist = []
        self.Ppllist = []
        self.currentfield = None
        self.currentfieldid = None
        self.currentwell = None
        self.currentwellid = None
        self.currentWellIdList = []

    def connectUi(self):
        self.searchbtn.clicked.connect(self.search_for_measurements_from_db)
        self.addbtn.clicked.connect(self.add_measurement_to_analysis)
        self.rmvbtn.clicked.connect(self.remove_measurement_from_analysis)
        self.fieldcombobox.currentIndexChanged.connect(self.get_well_names)
        self.wellcombobox.currentIndexChanged.connect(self.set_current_well)
        self.measurelistview.clicked.connect(lambda: self.draw_graph(0))
        self.chosenmeasurelistview.clicked.connect(lambda:self.draw_graph(1))


    def draw_graph(self, list_id):
        if not list_id:
            index = self.measurelistview.selectedIndexes()[0].row()
            self.plot.plot(self.notPpllist[index].data)
        else:
            index = self.chosenmeasurelistview.selectedIndexes()[0].row()
            self.plot.plot(self.Ppllist[index].data)

    def set_current_well(self):
        try:
            if self.wellcombobox.currentIndex() != 0:
                self.currentwell = self.wellcombobox.currentText()
                with sql_query() as connection:
                    cursor = connection.cursor()
                    cursor.execute('''SELECT ots_bn.soswell.welid from ots_bn.soswell
                                    WHERE (ots_bn.soswell.welname = :name) and
                                            (ots_bn.soswell.welfieldid = :fieldid) ''',
                                   fieldid=self.currentfieldid,
                                   name = self.currentwell)
                    self.currentwellid = cursor.fetchone()[0]
                    print(self.currentwellid)
            else:
                self.currentwell = None
                self.currentwellid = None
        except:
            pass


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
        self.fullMeasurementList = []
        only_field = False
        with sql_query() as connection:
            cursor = connection.cursor()
            startdate = self.startdate.date().toString(Qt.LocalDate)
            if self.startdate.date() != self.enddate.date():
               enddate=self.enddate.date().toString(Qt.LocalDate)
            else:
               enddate=self.enddate.date().addDays(1).toString(Qt.LocalDate)
            Sql_Q = ''' SELECT ots_bn.sosmeasurement.meswellid,
                                            ots_bn.sosmeasurementmtmeter.mtinterval,
                                            ots_bn.sosmeasurementmtmeter.mtdate,
                                            ots_bn.sosmeasurementmtmeter.mtpressure,
                                            ots_bn.sosmeasurementmtmeter.mttemperature,
                                            ots_bn.sosmeasurementmtmeter.mtcount,
                                            ots_bn.sosmeasurementmtmeter.mtdepthstartdate,
                                            ots_bn.sosmeasurementmtmeter.mtdepthinterval,
                                            ots_bn.sosmeasurementmtmeter.mtdepthdate,
                                            ots_bn.sosmeasurementmtmeter.mtdepth,
                                            ots_bn.sosmeasurement.messtartdate,
                                            ots_bn.sosmeasurement.mesid
                                            from ots_bn.sosmeasurementmtmeter
                                            INNER JOIN ots_bn.sosmeasurement
                                            ON ots_bn.sosmeasurementmtmeter.mtmeasurementid = ots_bn.sosmeasurement.mesid
                                            WHERE (ots_bn.sosmeasurement.mesoriginalid IS NULL) and
                                                  (ots_bn.sosmeasurement.messtartdate >= :startdate) and
                                                  (ots_bn.sosmeasurement.messtartdate <= :enddate) and
                                                  (ots_bn.sosmeasurement.mesdeviceid != 'SP2UXDfKjUuyuV/EDzyNEA')'''
            if self.fieldcombobox.currentIndex() != 0:
                if self.wellcombobox.currentIndex() != 0:
                    Sql_Q += 'and (ots_bn.sosmeasurement.meswellid = :wellid)'
                    cursor.execute(Sql_Q, startdate=startdate, enddate=enddate, wellid=self.currentwellid)
                else:
                    only_field = True
                    cursor.execute(Sql_Q, startdate=startdate, enddate=enddate)
            else:
                cursor.execute(Sql_Q, startdate=startdate, enddate=enddate)

            rows = cursor.fetchall()

            for row in rows:
                    print(row)
                    well_name, field_name = self.get_well_field_and_number(row[0]) # Скважина
                    if (not only_field or field_name == self.currentfield) and (row[9] is not None) and (row[3] is not None):

                        pressure_BLOB = row[3].read() # Давление
                        if pressure_BLOB[2] == 1:
                            bytes_len = int.from_bytes(pressure_BLOB[3:7], byteorder=byteorder)
                            pressure = pd.Series(np.frombuffer(pylzma.decompress(pressure_BLOB[7:],
                                                                                 maxlength=bytes_len), dtype=np.dtype('f')))
                        else:
                            pressure = pd.Series(np.frombuffer(pressure_BLOB[3:], dtype=np.dtype('f')))
                        print('pressure')
                        print(pressure)
                        pressure.dropna(inplace=True, how='all')
                        if row[4] is not None: #Температура
                            temperature_BLOB = row[4].read()
                            if temperature_BLOB[2] == 1:
                                bytes_len = int.from_bytes(temperature_BLOB[3:7], byteorder=byteorder)
                                temperature = pd.Series(np.frombuffer(pylzma.decompress(temperature_BLOB[7:],
                                                                                        maxlength=bytes_len),
                                                                                        dtype=np.dtype('f')))
                            else:
                                temperature = pd.Series(np.frombuffer(temperature_BLOB[3:], dtype=np.dtype('f')))
                        else:
                            temperature = pd.Series((None for i in range(pressure.size)))
                        print('temperature')
                        print(temperature)
                        temperature.dropna(inplace=True, how='all')
                        depth_BLOB = row[9].read() #Глубина
                        bytes_len = int.from_bytes(depth_BLOB[3:7], byteorder=byteorder)
                        depth = pd.Series(np.frombuffer(pylzma.decompress(depth_BLOB[7:],
                                                                          maxlength=bytes_len), dtype=np.dtype('f')))

                        print('depth')
                        print(depth)
                        if row[1] is not None: # Даты манометра
                            print('VAR 1')
                            mtStartDateTime = pd.to_datetime(row[10], dayfirst=True)
                            BD_mt_delta=row[1]
                            mtTimeDelta = pd.Timedelta(minutes=BD_mt_delta.minute, seconds=BD_mt_delta.second)
                            mt_dates = pd.Series((mtStartDateTime+mtTimeDelta*i for i in range(len(pressure))))
                        else:
                            print('VAR 2')
                            date_mt_BLOB = row[2].read()
                            bytes_len = int.from_bytes(date_mt_BLOB[3:7], byteorder=byteorder)
                            mt_dates = np.frombuffer(pylzma.decompress(date_mt_BLOB[7:], maxlength=bytes_len))
                            time = pd.Timestamp(row[10])
                            mt_dates = pd.to_datetime(mt_dates, unit='d', dayfirst=True, origin=pd.Timestamp(row[10]))
                            mt_dates = mt_dates.to_series(index=None)
                            mt_dates.reset_index(inplace=True, drop=True)
                            mt_dates = mt_dates.apply(lambda  x: x + pd.Timedelta(hours =time.hour, minutes=time.minute,
                                                                                  seconds=time.second))
                        print('mt_dates')
                        print(mt_dates)
                        if row[8] is not None: #даты глубин
                            date_depth_BLOB = row[8].read()
                            bytes_len = int.from_bytes(date_depth_BLOB[3:7], byteorder=byteorder)
                            dates_depth = np.frombuffer(pylzma.decompress(date_depth_BLOB[7:], maxlength=bytes_len))
                            dates_depth = pd.Series(pd.to_datetime(dates_depth, unit = 'd',
                                                                dayfirst=True, origin=pd.Timestamp('1899-12-30')))
                        else:
                            dpStartDateTime = pd.to_datetime(row[6], dayfirst=True)
                            BD_dp_delta=row[7]
                            mtTimeDelta = pd.Timedelta(minutes=BD_dp_delta.minute, seconds=BD_dp_delta.second)
                            dates_depth = pd.Series((dpStartDateTime+mtTimeDelta*i for i in range(len(depth))))
                        print('dates_depth')
                        print(dates_depth)
                        data = pd.concat([mt_dates, pressure, temperature,dates_depth,depth], axis = 1)
                        print(field_name, well_name)
                        print(data)
                        data.to_clipboard()
                        self.fullMeasurementList.append(Interpretation.Ppl_fabric(field_name, well_name, data))
        if len(self.fullMeasurementList)>0:
            self.bi_divide()

    def bi_divide(self):
        model = joblib.load('bin_divider_etc.pkl')
        self.leftmodel.clear()
        self.rightmodel.clear()
        for measuarement in self.fullMeasurementList:
            kt_pres = measuarement.data.iloc[:, 1].count()
            #print(measuarement.data)
            points300 = to_300_points(measuarement.data.iloc[:kt_pres,1])
            time_length = (measuarement.data.iloc[kt_pres-1,0]-measuarement.data.iloc[0,0]).total_seconds()/86400
            if time_length > 2: time_length = 2
            points300 = points300.append(pd.Series(time_length))
            #print(points300)
            points300 = points300.to_numpy().reshape(1, -1)
            prediction = model.predict(points300)
            name = QtGui.QStandardItem(measuarement.field + ' ' + measuarement.well_name)
            if prediction[0] == 1:
                self.Ppllist.append(measuarement)
                self.rightmodel.appendRow(name)
            else:
                self.notPpllist.append(measuarement)
                self.leftmodel.appendRow(name)
        self.measurelistview.setModel(self.leftmodel)
        self.chosenmeasurelistview.setModel(self.rightmodel)
        print(len(self.Ppllist))
        print(len(self.notPpllist))





    def add_measurement_to_analysis(self):
        pass

    def remove_measurement_from_analysis(self):
        pass

    def get_well_field_and_number(self, id):
        with sql_query() as connection:
            cursor = connection.cursor()
            cursor.execute(''' SELECT ots_bn.soswell.welname, ots_bn.soswell.welfieldid 
                            FROM ots_bn.soswell WHERE welid = :wellid''', wellid=id)
            rows = cursor.fetchone()
            wname = rows[0]
            cursor.execute(''' SELECT ots_bn.sosfield.fldname FROM ots_bn.sosfield
                            WHERE fldid = :fieldid''', fieldid=rows[1])
            rows = cursor.fetchone()
        return wname, rows[0]


def search_and_interpolate(searching_array, x, interpolate=True):
    if searching_array.iloc[-1,0] < searching_array.iloc[0,0]:
        searching_array = searching_array.iloc[::-1]
        searching_array.reset_index(inplace=True, drop=True)

    column1 = searching_array.iloc[:, 0]
    column2 = searching_array.iloc[:, 1]

    length = column1.size
    if (length != column2.size) or length == 0: return
    if length == 1: return column2.iloc[0]

    if x < column1.iloc[0]: return column2.iloc[0]
    if x > column1.iloc[-1]: return column2.iloc[-1]
    left = bisect.bisect_left(column1.to_list(), x)
    right = bisect.bisect_right(column1.to_list(), x)
    if left != right:
        return column2.iloc[left]
    else:
        if interpolate:
            percent = (x - column1.iloc[left-1]) / (column1.iloc[right] - column1.iloc[left-1])
            return column2.iloc[left-1] + abs((column2.iloc[right] - column2.iloc[left-1]) * percent)
        else:
            return column2.iloc[left]



def search_and_interpolate_old(searching_array, x, xleft=True, interpolate=True):
    if xleft:
        column1 = searching_array.iloc[:, 0].to_list()
        column2 = searching_array.iloc[:, 1].to_list()
    else:
        column2 = searching_array.iloc[:, 0].to_list()
        column1 = searching_array.iloc[:, 1].to_list()
    if x in column1:
        if column1.count(x) == 1:
            ind = column1.index(x)
        else:
            same_indexes = []
            for i in range(0, len(column1)):
                if column1[i] == x:
                    same_indexes.append(i)
            ind = int(sum(same_indexes) / len(same_indexes))
        return column2[ind]
    else:
        if column1[-1] > column1[0]:
            for i in range(0, len(column1) - 1):
                if column1[i] < x < column1[i + 1]:
                    ind = i
                    if interpolate:
                        percent = (x - column1[i]) / (column1[i + 1] - column1[i])
                        return column2[ind] + abs((column2[ind] - column2[ind + 1]) * percent)
                    return column2[ind]
        else:
            for i in range(1, len(column1)):
                if column1[i - 1] > x > column1[i]:
                    ind = i - 1
                    if interpolate:
                        percent = (x - column1[i]) / (column1[i + 1] - column1[i])
                        return column2[ind] + abs((column2[ind] - column2[ind + 1]) * percent)
                    return column2[ind]
    return 0

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


def to_300_points(sample):
    def final_300(sample):
        n = len(sample) // 300
        if n > 2:
            sample = sample.iloc[0::n]
            sample.reset_index(drop=True, inplace=True)
        new_index = pd.Series(((i + 1) / len(sample) * 300 for i in range(len(sample))))
        data = pd.concat((new_index, sample), axis=1)
        return pd.Series((search_and_interpolate(data, x + 1, interpolate=False) for x in range(300)))

    def inserting_nans(sample):
        old_array = sample.values
        new_array = np.array((old_array[0], None))
        for i in range(1, len(old_array)):
            new_array = np.append(new_array, (old_array[i], None))
        new_array = np.delete(new_array, -1)
        return pd.Series(new_array)

    if len(sample) == 300:
        return sample
    elif len(sample) < 300:
        while len(sample) < 300:
            sample = inserting_nans(sample)
            sample = pd.to_numeric(sample)
            sample = sample.interpolate()
        return final_300(sample)
    else:
        return final_300(sample)