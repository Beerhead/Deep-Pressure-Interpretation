from contextlib import contextmanager
from PyQt5 import QtSql, QtGui, QtCore
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
import uuid, base64
import datetime
from Plot import PlotWidget


class MeasurementsWidget(QWidget):

    def __init__(self, parent):
        super(MeasurementsWidget, self).__init__()
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.initUi()
        self.connectUi()
        self.parent = parent

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
        #self.startdate.setDate(QDate.currentDate().addDays(-1))
        self.startdate.setDate(QDate(2021, 6, 15))
        self.enddate = QDateEdit()
        self.enddate.setCalendarPopup(True)
        #self.enddate.setDate(QDate.currentDate())
        self.enddate.setDate(QDate(2021, 6, 16))
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
        self.date_and_button_layout.addWidget(self.to_analysis_btn)
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
        self.measurelistview.setMaximumWidth(300)
        self.chosenmeasurelistview.setMaximumWidth(300)

    def connectUi(self):
        self.searchbtn.clicked.connect(self.search_for_measurements_from_db)
        self.addbtn.clicked.connect(
            lambda: self.throw_measurement_between_lists(self.measurelistview, self.chosenmeasurelistview,
                                                         self.notPpllist, self.Ppllist))
        self.rmvbtn.clicked.connect(
            lambda: self.throw_measurement_between_lists(self.chosenmeasurelistview, self.measurelistview, self.Ppllist,
                                                         self.notPpllist))
        self.fieldcombobox.currentIndexChanged.connect(self.get_well_names)
        self.wellcombobox.currentIndexChanged.connect(self.set_current_well)
        self.measurelistview.clicked.connect(lambda: self.draw_graph(0))
        self.chosenmeasurelistview.clicked.connect(lambda: self.draw_graph(1))
        self.to_analysis_btn.clicked.connect(self.throw_measurements_to_main)

    def throw_measurements_to_main(self):
        if len(self.Ppllist) == 0: return
        for ppl in self.Ppllist: ppl.get_well_params()
        print(self.chosenmeasurelistview.model())
        print(self.Ppllist)
        self.parent.receive_list(self.chosenmeasurelistview.model(), self.Ppllist)
        self.close()

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
                                   name=self.currentwell)
                    self.currentwellid = cursor.fetchone()[0]
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
            enddate = self.enddate.date().addDays(1).toString(Qt.LocalDate)
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
                well_name, field_name = self.get_well_field_and_number(row[0])  # Скважина
                print(well_name, field_name)
                if (not only_field or field_name == self.currentfield) and (row[9] is not None) and (
                        row[3] is not None):
                    pressure_BLOB = row[3].read()  # Давление
                    if pressure_BLOB[2] == 1:
                        bytes_len = int.from_bytes(pressure_BLOB[3:7], byteorder=byteorder)
                        pressure = pd.Series(np.frombuffer(pylzma.decompress(pressure_BLOB[7:],
                                                                             maxlength=bytes_len), dtype=np.dtype('f')))
                    else:
                        pressure = pd.Series(np.frombuffer(pressure_BLOB[3:], dtype=np.dtype('f')))

                    pressure.dropna(inplace=True, how='all')
                    if row[4] is not None:  # Температура
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

                    temperature.dropna(inplace=True, how='all')
                    depth_BLOB = row[9].read()  # Глубина
                    bytes_len = int.from_bytes(depth_BLOB[3:7], byteorder=byteorder)
                    depth = pd.Series(np.frombuffer(pylzma.decompress(depth_BLOB[7:],
                                                                      maxlength=bytes_len), dtype=np.dtype('f')))

                    if row[1] is not None:  # Даты манометра
                        mtStartDateTime = pd.to_datetime(row[10], dayfirst=True)
                        BD_mt_delta = row[1]
                        mtTimeDelta = pd.Timedelta(minutes=BD_mt_delta.minute, seconds=BD_mt_delta.second)
                        mt_dates = pd.Series((mtStartDateTime + mtTimeDelta * i for i in range(len(pressure))))
                    else:
                        date_mt_BLOB = row[2].read()
                        bytes_len = int.from_bytes(date_mt_BLOB[3:7], byteorder=byteorder)
                        mt_dates = np.frombuffer(pylzma.decompress(date_mt_BLOB[7:], maxlength=bytes_len))
                        time = pd.Timestamp(row[10])
                        mt_dates = pd.to_datetime(mt_dates, unit='d', dayfirst=True, origin=pd.Timestamp(row[10]))
                        mt_dates = mt_dates.to_series(index=None)
                        mt_dates.reset_index(inplace=True, drop=True)
                        mt_dates = mt_dates.apply(lambda x: x + pd.Timedelta(hours=time.hour, minutes=time.minute,
                                                                             seconds=time.second))

                    if row[8] is not None:  # даты глубин
                        date_depth_BLOB = row[8].read()
                        bytes_len = int.from_bytes(date_depth_BLOB[3:7], byteorder=byteorder)
                        dates_depth = np.frombuffer(pylzma.decompress(date_depth_BLOB[7:], maxlength=bytes_len))
                        dates_depth = pd.Series(pd.to_datetime(dates_depth, unit='d',
                                                               dayfirst=True, origin=pd.Timestamp('1899-12-30')))
                    else:
                        dpStartDateTime = pd.to_datetime(row[6], dayfirst=True)
                        mtTimeDelta = pd.Timedelta(minutes=row[7].minute, seconds=row[7].second)
                        dates_depth = pd.Series((dpStartDateTime + mtTimeDelta * i for i in range(len(depth))))

                    data = pd.concat([mt_dates, pressure, temperature, dates_depth, depth], axis=1)
                    Ppl = Interpretation.Ppl_fabric(field_name, well_name, data)
                    Ppl.OTS_Well_ID = row[0]
                    Ppl.OTS_Mes_ID = row[11]
                    self.fullMeasurementList.append(Ppl)

        print(self.fullMeasurementList)
        if len(self.fullMeasurementList) > 0:
            self.bi_divide()

    def bi_divide(self):
        model = joblib.load('bin_divider_etc.pkl')
        self.leftmodel.clear()
        self.rightmodel.clear()
        for measuarement in self.fullMeasurementList:
            try:
                kt_pres = measuarement.data.iloc[:, 1].count()
                # print(measuarement.data)
                measuarement.data.to_clipboard()
                points300 = to_300_points(measuarement.data.iloc[:kt_pres, 1])
                time_length = (measuarement.data.iloc[kt_pres - 1, 0] - measuarement.data.iloc[
                    0, 0]).total_seconds() / 86400
                if time_length > 2: time_length = 2
                points300 = points300 / points300.max()
                points300 = points300.append(pd.Series(time_length))
                points300 = points300.to_numpy().reshape(1, -1)

                prediction = model.predict(points300)
                name = QtGui.QStandardItem(measuarement.field + ' ' + measuarement.well_name)

                if prediction[0] == 1:
                    self.Ppllist.append(measuarement)
                    self.rightmodel.appendRow(name)
                else:
                    self.notPpllist.append(measuarement)
                    self.leftmodel.appendRow(name)
            except:
                print("ИСКЛЮЧЕНА:", measuarement.field, measuarement.well_name)
        self.measurelistview.setModel(self.leftmodel)
        self.chosenmeasurelistview.setModel(self.rightmodel)
        # print(len(self.Ppllist))
        # print(len(self.notPpllist))

    def throw_measurement_between_lists(self, listview_from, listview_to, list_from, list_to):
        index_list = listview_from.selectedIndexes()
        if len(index_list) > 0:
            for index in index_list[::-1]:
                item_text = listview_from.model().item(index.row()).text()
                listview_from.model().removeRow(index.row())
                new_item = QtGui.QStandardItem(item_text)
                listview_to.model().appendRow(new_item)
                list_to.append(list_from.pop(index.row()))

    def get_well_field_and_number(self, ID):
        with sql_query() as connection:
            cursor = connection.cursor()
            cursor.execute(''' SELECT ots_bn.soswell.welname, ots_bn.soswell.welfieldid 
                            FROM ots_bn.soswell WHERE welid = :wellid''', wellid=ID)
            rows = cursor.fetchone()
            wname = rows[0]
            cursor.execute(''' SELECT ots_bn.sosfield.fldname FROM ots_bn.sosfield
                            WHERE fldid = :fieldid''', fieldid=rows[1])
            rows = cursor.fetchone()
        return wname, rows[0]




def insert_data_to_sosresearch_table(Ppl):
    if Ppl.table_models is None:return
    resid = Ppl.resid
    RESCREATEDATETIME = datetime.datetime.now()#.strftime('%d.%m.%Y %H:%M:%S')
    resmoduletypeid = 'none'
    resinterpretatorid = 'U5kpGq6aT42GXQopVH7PTA'
    RESSUBORGANIZATIONID = Ppl.TP_id
    RESGEOLOGISTORGANIZATIONID = Ppl.tzeh_id
    resresearchtypeid = 281
    resfieldid = Ppl.OTS_Field_ID
    reswellid = Ppl.OTS_Well_ID
    rescreatorid = 'U5kpGq6aT42GXQopVH7PTA'
    resfirstmeasurementstamp = Ppl.first_measure_datetime.to_pydatetime()
    resgoal = 'Рпласт'
    resstatusid = 20
    reslastmeasurementstamp = Ppl.last_measure_datetime.to_pydatetime()
    resinbrief = 1  # ???
    with sql_query() as connection:
        cursor = connection.cursor()
        Sql_Q = ''' INSERT INTO ots_bn.sosresearch
                    (ots_bn.sosresearch.resid,
                    ots_bn.sosresearch.RESCREATEDATETIME,
                    ots_bn.sosresearch.resmoduletypeid,
                    ots_bn.sosresearch.resinterpretatorid,
                    ots_bn.sosresearch.RESSUBORGANIZATIONID,
                    ots_bn.sosresearch.RESGEOLOGISTORGANIZATIONID,
                    ots_bn.sosresearch.resresearchtypeid,
                    ots_bn.sosresearch.resfieldid,
                    ots_bn.sosresearch.reswellid,
                    ots_bn.sosresearch.rescreatorid,
                    ots_bn.sosresearch.resfirstmeasurementstamp,
                    ots_bn.sosresearch.resgoal,
                    ots_bn.sosresearch.resstatusid,
                    ots_bn.sosresearch.reslastmeasurementstamp,
                    ots_bn.sosresearch.resinbrief) VALUES
                    (:resid,
                    :RESCREATEDATETIME,
                    :resmoduletypeid,
                    :resinterpretatorid,
                    :RESSUBORGANIZATIONID,
                    :RESGEOLOGISTORGANIZATIONID,
                    :resreserachtypeid,
                    :resfieldid,
                    :reswellid,
                    :rescreatorid,
                    :resfirstmeasarementstamp,
                    :resgoal,
                    :resstatusid,
                    :reslastmeasarementstamp,
                    :resinbrief)'''


        print(resid,
        RESCREATEDATETIME,
        resmoduletypeid,
        resinterpretatorid,
        RESSUBORGANIZATIONID,
        RESGEOLOGISTORGANIZATIONID,
        resresearchtypeid,
        resfieldid,
        reswellid,
        rescreatorid,
        resfirstmeasurementstamp,
        resgoal,
        resstatusid,
        reslastmeasurementstamp,
        resinbrief)
        cursor.execute(Sql_Q,
                    resid=resid,
                    RESCREATEDATETIME=RESCREATEDATETIME,
                    resmoduletypeid=resmoduletypeid,
                    resinterpretatorid=resinterpretatorid,
                    RESSUBORGANIZATIONID=RESSUBORGANIZATIONID,
                    RESGEOLOGISTORGANIZATIONID=RESGEOLOGISTORGANIZATIONID,
                    resreserachtypeid=resresearchtypeid,
                    resfieldid=resfieldid,
                    reswellid=reswellid,
                    rescreatorid=rescreatorid,
                    resfirstmeasarementstamp=resfirstmeasurementstamp,
                    resgoal=resgoal,
                    resstatusid=resstatusid,
                    reslastmeasarementstamp=reslastmeasurementstamp,
                    resinbrief=resinbrief)
        connection.commit()

def insert_data_to_sosresearchresult_table(Ppl):
    rsrrresearchid = Ppl.resid
    rsrcriticalvolumegascontent = 20
    rsrreckonspeedloss = 0
    rsrpiperoughness = 0.0254
    rsrdensityliquid = Ppl.ro
    rsrresulmeasurementid = Ppl.OTS_Mes_ID
    rsrinflowtechhaltime = 180  # ????????????????
    rsrextractzaboyleveltypeid = 1  # ?????????



def insert_data_to_sosresearchmeasurement_table(Ppl):
    double_measure_in_sosmeasurement(Ppl)
    rsrresearchid = Ppl.resid
    rmsmeasurementid = Ppl.OTS_New_Mes_ID
    if Ppl.table_ind == 0:
        rmsaslgorithmtype = 1
        rmsdescentdensityliquid = Ppl.ro
    else:
        rmsaslgorithmtype = 2
        rmsascentdensityliquid = Ppl.ro
    rmsavgtgradient = Ppl.avg_temp_gradient
    rmsgasoilboundary = Ppl.GOB
    rmswateroilboundary = Ppl.OWB
    rmsgaswaterboundary = Ppl.GWB
    rmsstaticlevel = Ppl.static_level
    if rmsgasoilboundary is not None:
        rmsverticalgasoilboundary = search_and_interpolate(Ppl.incl, rmsgasoilboundary)
    if rmswateroilboundary is not None:
        rmsverticalwateroilboundary = search_and_interpolate(Ppl.incl, rmswateroilboundary)
    if rmsgaswaterboundary is not None:
        rmsverticalgaswaterboundary = search_and_interpolate(Ppl.incl, rmsgaswaterboundary)
    if Ppl.table_ind == 0:
        rmsascentshelfs = make_shelfs_blob(Ppl)
        rmsdescentshelfs = None
    else:
        rmsascentshelfs = None
        rmsdescentshelfs = make_shelfs_blob(Ppl)



def double_measure_in_sosmeasurement(Ppl):
    with sql_query() as connection:
        cursor = connection.cursor()
        cursor.execute(''' SELECT * from ots_bn.sosmeasurement
                    WHERE ots_bn.sosmeasurement.mesid = :mesid''',
                       mesid=Ppl.OTS_Mes_ID)
        row = list(cursor.fetchone())
        original_id = row[0]
        row[0] = make_id()
        Ppl.OTS_New_Mes_ID = row[0]
        row[1] = Ppl.resid
        row[15] = original_id
        row[16] = 'U5kpGq6aT42GXQopVH7PTA'
        row[17] = None
        row[18] = None
        row[22] = None



def make_shelfs_blob(Ppl):
    if Ppl.final_data is None or Ppl.table_models is None:
        print('no data')
        return
    model = Ppl.table_models[Ppl.table_ind]

    blob = b'\x01'
    for i in range(model.rowCount()):
        float_series = pd.Series(dtype='f')
        depth = float(model.item(i, 0).text())
        elongation = float(model.item(i, 1).text())
        vert_depth = depth - elongation
        pressure = float(model.item(i, 2).text())
        temperature = float(model.item(i, 3).text())
        try:
            if model.item(i, 6).checkState() == 2:
                blob += (b'\x01' + b'\x01')
            else:
                blob += (b'\x00' + b'\x01')
            ro = float(model.item(i, 4).text())
            interval = float(model.item(i, 0).text()) - float(model.item(i-1, 0).text())
            delta_pres = pressure - float(model.item(i-1, 2).text())
            delta_vert_depth = vert_depth - (float(model.item(i - 1, 0).text()) - float(model.item(i - 1, 1).text()))
            temp_grad = (temperature - float(model.item(i - 1, 3).text()))/delta_vert_depth

        except: # первая строка
            ro = 0.
            delta_vert_depth = 0.
            interval = 0.
            delta_pres = 0.
            temp_grad = 0.
            blob += (b'\x00' + b'\x00')
        if Ppl.table_ind == 0:
            datetime = search_and_interpolate(Ppl.final_data.iloc[:int(Ppl.divider_points[1]), [4, 3]], depth)
            blob += pd.Series((datetime - pd.Timestamp('1899-12-30')).total_seconds()/86400).to_numpy().tobytes()
        else:
            datetime = search_and_interpolate(Ppl.final_data.iloc[int(Ppl.divider_points[1]):, [4, 3]], depth)
            blob += pd.Series((datetime - pd.Timestamp('1899-12-30')).total_seconds()/86400).to_numpy().tobytes()
        temp_series = pd.Series([depth, vert_depth, pressure, temperature, interval, elongation, delta_pres,
                                       delta_vert_depth, ro, temp_grad, pressure, delta_pres, ro], dtype='f')
        float_series=float_series.append(temp_series, ignore_index=True)
        blob += float_series.to_numpy().tobytes()
    return blob
    #print(Ppl.well_name, len(blob), blob)
    #blob_what_is_what(blob)

def insert_data_to_sosresearchgraph_table(Ppl):
    rgrid = make_id()
    rgrresearchid = Ppl.resid
    rgrgraphtypeid = 64
    rgrcoordtypeid = 32
    rgrorder = '1'
    rgrfilename = 'khOCt7N/xkmfK+XrNUl3hw'
    plot = PlotWidget()
    fig = plot.plot(Ppl.final_data, save=True)
    rgrpicture = fig.getvalue()

def insert_data_to_sosresearchmarkermeasurement_table(Ppl):

    rmmresearchid = Ppl.resid
    rmmmeasurementid = Ppl.OTS_New_Mes_ID
    rmmtypeid = 5
    rmmdate = datetime.datetime.now().strftime('%d.%m.%Y %H:%M:%S')
    rmmp = Ppl.final_data.iloc[:, 1].max()
    rmmt = Ppl.final_data.iloc[:, 2].max()


def make_id():
    return base64.b64encode(uuid.uuid4().bytes)[:-2].decode()


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


def blob_what_is_what(bytes_in):

        a = bytes_in[1:]

        num_shelfs = int((len(a))/62)
        lists=[list() for i in range(16)]
        for shelf in range(num_shelfs):
            lists[0].append(a[shelf*62]) #bool расчет ср.пл.
            lists[1].append(a[1+shelf * 62]) #bool расчет ср.темп.
            lists[2].append(np.frombuffer(a[ 2 + shelf * 62:10 + shelf * 62]).data[0]) # datetime
            lists[3].append(np.frombuffer(a[10 + shelf * 62:14 + shelf * 62], dtype=np.dtype('f')).data[0]) # глубина(м)
            lists[4].append(np.frombuffer(a[14 + shelf * 62:18 + shelf * 62], dtype=np.dtype('f')).data[0]) # глубина с учетом удлинения
            lists[5].append(np.frombuffer(a[18 + shelf * 62:22 + shelf * 62], dtype=np.dtype('f')).data[0])  # давление
            lists[6].append(np.frombuffer(a[22 + shelf * 62:26 + shelf * 62], dtype=np.dtype('f')).data[0])  # температура
            lists[7].append(np.frombuffer(a[26 + shelf * 62:30 + shelf * 62], dtype=np.dtype('f')).data[0])  # интервал
            lists[8].append(np.frombuffer(a[30 + shelf * 62:34 + shelf * 62], dtype=np.dtype('f')).data[0])  # удлинение
            lists[9].append(np.frombuffer(a[34 + shelf * 62:38 + shelf * 62], dtype=np.dtype('f')).data[0])  # разность давлений между полками
            lists[10].append(np.frombuffer(a[38 + shelf * 62:42 + shelf * 62], dtype=np.dtype('f')).data[0])  # разность глубин с учетом удлинения
            lists[11].append(np.frombuffer(a[42 + shelf * 62:46 + shelf * 62], dtype=np.dtype('f')).data[0])  # расчетная плотность
            lists[12].append(np.frombuffer(a[46 + shelf * 62:50 + shelf * 62], dtype=np.dtype('f')).data[0])  # градиент температуры
            lists[13].append(np.frombuffer(a[50 + shelf * 62:54 + shelf * 62], dtype=np.dtype('f')).data[0])  # давление 2
            lists[14].append(np.frombuffer(a[54 + shelf * 62:58 + shelf * 62], dtype=np.dtype('f')).data[0])  # разность давлений 2
            lists[15].append(np.frombuffer(a[58 + shelf * 62:62 + shelf * 62], dtype=np.dtype('f')).data[0])  # расчетная плотность 2

        for l in lists:
            print(l, end ='\n\n')

def search_and_interpolate(searching_array, x, interpolate=True):
    if searching_array.iloc[-1, 0] < searching_array.iloc[0, 0]:
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
            percent = (x - column1.iloc[left - 1]) / (column1.iloc[right] - column1.iloc[left - 1])
            return column2.iloc[left - 1] + abs((column2.iloc[right] - column2.iloc[left - 1]) * percent)
        else:
            return column2.iloc[left]
