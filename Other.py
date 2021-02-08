from contextlib import contextmanager
from PyQt5 import QtSql, QtGui
from PyQt5.Qt import *
from Plot import PlotWidget2
import cx_Oracle


class MeasurementsWidget(QWidget):

    def __init__(self):
        super(MeasurementsWidget, self).__init__()
        self.initUi()
        self.connectUi()

    def initUi(self):
        cx_Oracle.init_oracle_client()
        self.HL = QHBoxLayout(self)
        self.measurelistview = QListView()
        self.date_and_button_layout = QVBoxLayout()
        self.searchbtn = QPushButton('Найти замеры')
        self.addbtn = QPushButton('->')
        self.rmvbtn = QPushButton('<-')
        self.to_analysis_btn = QPushButton('Добавить в анализ выбранное')
        self.startdate = QDateEdit()
        self.startdate.setCalendarPopup(True)
        self.startdate.setDate(QDate.currentDate())
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


    def connectUi(self):
        self.searchbtn.clicked.connect(self.search_for_measurements_from_db)
        self.addbtn.clicked.connect(self.add_measurement_to_analysis)
        self.rmvbtn.clicked.connect(self.remove_measurement_from_analysis)
        self.fieldcombobox.currentIndexChanged.connect(self.get_well_names)

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
                currentfield = self.fieldcombobox.currentText
                cursor.execute('''SELECT ots_bn.sosfield.fldid from  ots_bn.sosfield
                                WHERE ots_bn.sosfield.fldname = :field''', field = currentfield)
                currentfieldid = cursor.fetchone()[0]
                cursor.execute('''SELECT ots_bn.soswell.welname from ots_bn.soswell 
                                WHERE ots_bn.soswell.welfieldid = :fieldid''', fieldid = currentfieldid)
                rows = [field[0] for field in cursor.fetchall()]
                self.wellcombobox.addItems(rows)
        else:
            self.wellcombobox.addItem('Все скважины')

    def search_for_measurements_from_db(self):
        with sql_query() as connection:
            cursor = connection.cursor()
            cursor.execute('''SELECT ots_bn.sosmeasurementmtmeter.mtinterval, ots_bn.sosmeasurement.meswellid
                            FROM OTS_BN.Sosmeasurementmtmeter
                            INNER JOIN OTS_BN.sosmeasurement 
                            ON ots_bn.sosmeasurement.MESID = ots_bn.sosmeasurementmtmeter.mtmeasurementid
                            WHERE (ots_bn.sosmeasurement.messtartdate > '25.01.2021') and
                            (ots_bn.sosmeasurement.mestypeid = 'mtmeter')''')
            rows = cursor.fetchall()

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
    msg = None
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
