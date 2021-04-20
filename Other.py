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
from math import ceil, floor

class MeasurementsWidget(QWidget):
    """Widget to choose measurements to interpret"""
    def __init__(self, parent):
        super(MeasurementsWidget, self).__init__()
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.initUI()
        self.connectUI()
        self.parent = parent
        self.setAttribute(Qt.WA_DeleteOnClose, False)

    def initUI(self):
        self.setWindowTitle("Выбор замеров")
        self.HL = QHBoxLayout(self)
        self.measureListView = QListView()
        self.dateAndButtonLayout = QVBoxLayout()
        self.searchBtn = QPushButton('Найти замеры')
        self.addBtn = QPushButton('->')
        self.rmvBtn = QPushButton('<-')
        self.toAnalysisBtn = QPushButton('Добавить в анализ выбранное')
        self.startDate = QDateEdit()
        self.startDate.setCalendarPopup(True)
        self.startDate.setDate(QDate.currentDate().addDays(-1))
        self.startDate.setDate(QDate(2021, 6, 15))
        self.endDate = QDateEdit()
        self.endDate.setCalendarPopup(True)
        self.endDate.setDate(QDate.currentDate())
        self.endDate.setDate(QDate(2021, 6, 16))
        self.fieldCombobox = QComboBox()
        self.fieldCombobox.addItems(self.getFieldNames())
        self.wellCombobox = QComboBox()
        self.wellCombobox.addItem('Все скважины')
        self.dateAndButtonLayout.addWidget(self.fieldCombobox)
        self.dateAndButtonLayout.addWidget(self.wellCombobox)
        self.dateAndButtonLayout.addWidget(self.startDate)
        self.dateAndButtonLayout.addWidget(self.endDate)
        self.dateAndButtonLayout.addWidget(self.searchBtn)
        self.dateAndButtonLayout.addWidget(self.addBtn)
        self.dateAndButtonLayout.addWidget(self.rmvBtn)
        self.dateAndButtonLayout.addWidget(self.toAnalysisBtn)
        self.chosenMeasureListView = QListView()
        self.plot = PlotWidget2()
        self.HL.addWidget(self.measureListView)
        self.HL.addLayout(self.dateAndButtonLayout)
        self.HL.addWidget(self.chosenMeasureListView)
        self.HL.addWidget(self.plot)
        self.measureListView.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.chosenMeasureListView.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.leftModel = QtGui.QStandardItemModel()
        self.leftModel.setColumnCount(1)
        self.rightModel = QtGui.QStandardItemModel()
        self.rightModel.setColumnCount(1)
        self.notPpllist = []
        self.Ppllist = []
        self.currentField = None
        self.currentFieldID = None
        self.currentWell = None
        self.currentWellID = None
        self.currentWellIdList = []
        self.measureListView.setMaximumWidth(300)
        self.chosenMeasureListView.setMaximumWidth(300)

    def connectUI(self):
        self.searchBtn.clicked.connect(self.searchForMeasurementsFromDB)
        self.addBtn.clicked.connect(
            lambda: self.throwMeasurementBetweenLists(self.measureListView, self.chosenMeasureListView,
                                                      self.notPpllist, self.Ppllist))
        self.rmvBtn.clicked.connect(
            lambda: self.throwMeasurementBetweenLists(self.chosenMeasureListView, self.measureListView, self.Ppllist,
                                                      self.notPpllist))
        self.fieldCombobox.currentIndexChanged.connect(self.getWellNames)
        self.wellCombobox.currentIndexChanged.connect(self.setCurrentWell)
        self.measureListView.clicked.connect(lambda: self.drawGraph(0))
        self.chosenMeasureListView.clicked.connect(lambda: self.drawGraph(1))
        self.toAnalysisBtn.clicked.connect(self.throwMeasurementsToMain)

    def throwMeasurementsToMain(self):
        if len(self.Ppllist) == 0: return
        for ppl in self.Ppllist: ppl.getWellParams()
        self.parent.receiveList(self.makeModelToReturnToMain(), self.Ppllist)
        self.hide()


    def makeModelToReturnToMain(self):
        model = self.chosenMeasureListView.model()
        newModel = QtGui.QStandardItemModel(len(self.Ppllist), 4)
        newModel.setHorizontalHeaderLabels(['Площадь', 'Скважина', 'Интервал', 'В отчет/БД'])
        for i in range(model.rowCount()):
            try:
                square, wellNum = model.item(i, 0).text().split(" ")
            except:
                *square, wellNum = model.item(i, 0).text().split(" ")
                square = " ".join(square)
            lastDigit = int(str(int(self.Ppllist[i].maxDepth / 50))[-1])
            if self.Ppllist[i].maxDepth <=1000:
                interval = 50
            elif lastDigit >=5:
                interval = ceil(self.Ppllist[i].maxDepth / 500) * 50
            else:
                interval = floor(self.Ppllist[i].maxDepth / 500) * 50
            item = QStandardItem(square)
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            item.setEditable(False)
            newModel.setItem(i, 0, QStandardItem(item))
            item.setText(wellNum)
            item.setEditable(False)
            newModel.setItem(i, 1, QStandardItem(item))
            item.setText(str(interval))
            item.setEditable(True)
            newModel.setItem(i, 2, QStandardItem(item))

            item.setText('')
            item.setEditable(False)
            item.setCheckable(True)
            newModel.setItem(i, 3, QStandardItem(item))

        return newModel

    def drawGraph(self, listID):
        if not listID:
            index = self.measureListView.selectedIndexes()[0].row()
            self.plot.plot(self.notPpllist[index].data)
        else:
            index = self.chosenMeasureListView.selectedIndexes()[0].row()
            self.plot.plot(self.Ppllist[index].data)

    def setCurrentWell(self):
        try:
            if self.wellCombobox.currentIndex() != 0:
                self.currentWell = self.wellCombobox.currentText()
                with sqlQuery() as connection:
                    cursor = connection.cursor()
                    cursor.execute('''SELECT ots_bn.soswell.welid from ots_bn.soswell
                                    WHERE (ots_bn.soswell.welname = :name) and
                                            (ots_bn.soswell.welfieldid = :fieldid) ''',
                                   fieldid=self.currentFieldID,
                                   name=self.currentWell)
                    self.currentWellID = cursor.fetchone()[0]
            else:
                self.currentWell = None
                self.currentWellID = None
        except:
            pass

    def getFieldNames(self):
        with sqlQuery() as connection:
            cursor = connection.cursor()
            cursor.execute('''SELECT ots_bn.sosfield.fldname from  ots_bn.sosfield''')
            rows = [field[0] for field in cursor.fetchall()]
            rows.insert(0, 'Все площади')
            return rows

    def getWellNames(self):
        self.wellCombobox.clear()
        if self.fieldCombobox.currentIndex() != 0:
            with sqlQuery() as connection:
                cursor = connection.cursor()
                self.currentField = self.fieldCombobox.currentText()
                cursor.execute('''SELECT ots_bn.sosfield.fldid from  ots_bn.sosfield
                                WHERE ots_bn.sosfield.fldname = :field''', field=self.currentField)
                self.currentFieldID = cursor.fetchone()[0]
                cursor.execute('''SELECT ots_bn.soswell.welname from ots_bn.soswell 
                                WHERE ots_bn.soswell.welfieldid = :fieldid''', fieldid=self.currentFieldID)
                rows = [field[0] for field in cursor.fetchall()]
                rows.insert(0, 'Все скважины')
                self.wellCombobox.addItems(rows)
        else:
            self.wellCombobox.addItem('Все скважины')
            self.currentWell = None
            self.currentWellID = None

    def searchForMeasurementsFromDB(self):
        self.fullMeasurementList = []
        onlyField = False
        with sqlQuery() as connection:
            cursor = connection.cursor()
            startDate = self.startDate.date().toString(Qt.LocalDate)
            endDate = self.endDate.date().addDays(1).toString(Qt.LocalDate)
            sql_Q = ''' SELECT ots_bn.sosmeasurement.meswellid,
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
                              (ots_bn.sosmeasurement.messtartdate >= :startDate) and
                              (ots_bn.sosmeasurement.messtartdate <= :endDate) and
                              (ots_bn.sosmeasurement.mesdeviceid != 'SP2UXDfKjUuyuV/EDzyNEA')
                              '''
            if self.fieldCombobox.currentIndex() != 0:
                if self.wellCombobox.currentIndex() != 0:
                    sql_Q += 'and (ots_bn.sosmeasurement.meswellid = :wellid)'
                    cursor.execute(sql_Q, startdate=startDate, enddate=endDate, wellid=self.currentWellID)
                else:
                    onlyField = True
                    cursor.execute(sql_Q, startdate=startDate, enddate=endDate)
            else:
                cursor.execute(sql_Q, startdate=startDate, enddate=endDate)
            rows = cursor.fetchall()

            for row in rows:
                print(row)
                wellName, fieldName = self.getWellFieldAndNumber(row[0])  # Скважина
                print(wellName, fieldName)
                if (not onlyField or fieldName == self.currentField) and (row[9] is not None) and (
                        row[3] is not None):
                    pressureBLOB = row[3].read()  # Давление
                    if pressureBLOB[2] == 1:
                        bytesLen = int.from_bytes(pressureBLOB[3:7], byteorder=byteorder)
                        pressure = pd.Series(np.frombuffer(pylzma.decompress(pressureBLOB[7:],
                                                           maxlength=bytesLen),
                                                           dtype=np.dtype('f')))
                    else:
                        pressure = pd.Series(np.frombuffer(pressureBLOB[3:], dtype=np.dtype('f')))

                    pressure.dropna(inplace=True, how='all')
                    pressure.reset_index(inplace=True, drop=True)
                    if row[4] is not None:  # Температура
                        temperatureBLOB = row[4].read()
                        if temperatureBLOB[2] == 1:
                            bytesLen = int.from_bytes(temperatureBLOB[3:7], byteorder=byteorder)
                            temperature = pd.Series(np.frombuffer(pylzma.decompress(temperatureBLOB[7:],
                                                                  maxlength=bytesLen),
                                                                  dtype=np.dtype('f')))
                        else:
                            temperature = pd.Series(np.frombuffer(temperatureBLOB[3:], dtype=np.dtype('f')))
                    else:
                        temperature = pd.Series((None for i in range(pressure.size)))

                    temperature.dropna(inplace=True, how='all')
                    temperature.reset_index(inplace=True, drop=True)

                    depthBLOB = row[9].read()  # Глубина
                    bytesLen = int.from_bytes(depthBLOB[3:7], byteorder=byteorder)
                    depth = pd.Series(np.frombuffer(pylzma.decompress(depthBLOB[7:],
                                                                      maxlength=bytesLen), dtype=np.dtype('f')))
                    depth.reset_index(inplace=True, drop=True)
                    if row[1] is not None:  # Даты манометра
                        mtStartDateTime = pd.to_datetime(row[10], dayfirst=True)
                        BD_mt_delta = row[1]
                        mtTimeDelta = pd.Timedelta(minutes=BD_mt_delta.minute, seconds=BD_mt_delta.second)
                        mtDates = pd.Series((mtStartDateTime + mtTimeDelta * i for i in range(len(pressure))))
                    else:
                        dateMtBLOB = row[2].read()
                        bytesLen = int.from_bytes(dateMtBLOB[3:7], byteorder=byteorder)
                        mtDates = np.frombuffer(pylzma.decompress(dateMtBLOB[7:], maxlength=bytesLen))
                        time = pd.Timestamp(row[10])
                        mtDates = pd.to_datetime(mtDates, unit='d', dayfirst=True, origin=pd.Timestamp(row[10]))
                        mtDates = mtDates.to_series(index=None)
                        mtDates.reset_index(inplace=True, drop=True)
                        mtDates = mtDates.apply(lambda x: x + pd.Timedelta(hours=time.hour, minutes=time.minute,
                                                                           seconds=time.second))
                    mtDates.reset_index(inplace=True, drop=True)
                    if row[8] is not None:  # даты глубин
                        dateDepthBLOB = row[8].read()
                        bytesLen = int.from_bytes(dateDepthBLOB[3:7], byteorder=byteorder)
                        datesDepth = np.frombuffer(pylzma.decompress(dateDepthBLOB[7:], maxlength=bytesLen))
                        datesDepth = pd.Series(pd.to_datetime(datesDepth, unit='d',
                                                              dayfirst=True, origin=pd.Timestamp('1899-12-30')))
                    else:
                        dpStartDateTime = pd.to_datetime(row[6], dayfirst=True)
                        mtTimeDelta = pd.Timedelta(minutes=row[7].minute, seconds=row[7].second)
                        datesDepth = pd.Series((dpStartDateTime + mtTimeDelta * i for i in range(len(depth))))
                    datesDepth.reset_index(inplace=True, drop=True)
                    mtData = pd.concat([mtDates, pressure, temperature], axis=1, ignore_index=True)
                    datesDepth = datesDepth.rename_axis(3)
                    depth = depth.rename_axis(4)
                    spsData = pd.concat([datesDepth, depth], axis=1, ignore_index=True)
                    mtData.dropna(inplace=True)
                    spsData.dropna(inplace=True)
                    data = pd.concat([mtData, spsData], axis=1, ignore_index=True, keys=[0,1,2,3,4])
                    ppl = Interpretation.PplFabric(fieldName, wellName, data)
                    ppl.otsWellID = row[0]
                    ppl.otsMesID = row[11]
                    self.fullMeasurementList.append(ppl)
        if len(self.fullMeasurementList) > 0:
            self.biDivide()

    def biDivide(self):
        model = joblib.load('bin_divider_etc.pkl')
        self.leftModel.clear()
        self.rightModel.clear()
        for measurement in self.fullMeasurementList:
            try:
                numDotsPressure = measurement.data.iloc[:, 1].count()
                measurement.data.to_clipboard()
                points300 = to300Points(measurement.data.iloc[:numDotsPressure, 1])
                timeLength = (measurement.data.iloc[numDotsPressure - 1, 0] - measurement.data.iloc[
                    0, 0]).total_seconds() / 86400
                if timeLength > 2: timeLength = 2
                points300 = points300 / points300.max()
                points300 = points300.append(pd.Series(timeLength))
                points300 = points300.to_numpy().reshape(1, -1)
                prediction = model.predict(points300)
                name = QtGui.QStandardItem(measurement.field + ' ' + measurement.wellName)
                if prediction[0] == 1:
                    self.Ppllist.append(measurement)
                    self.rightModel.appendRow(name)
                else:
                    self.notPpllist.append(measurement)
                    self.leftModel.appendRow(name)
            except:
                print("ИСКЛЮЧЕНА:", measurement.field, measurement.wellName)
        self.measureListView.setModel(self.leftModel)
        self.chosenMeasureListView.setModel(self.rightModel)

    def throwMeasurementBetweenLists(self, listviewFrom, listviewTo, listFrom, listTo):
        indexList = listviewFrom.selectedIndexes()
        try:
            if len(indexList) > 0:
                for index in indexList[::-1]:
                    itemText = listviewFrom.model().item(index.row()).text()
                    listviewFrom.model().removeRow(index.row())
                    newItem = QtGui.QStandardItem(itemText)
                    listviewTo.model().appendRow(newItem)
                    listTo.append(listFrom.pop(index.row()))
        except:
            pass

    def getWellFieldAndNumber(self, ID):
        with sqlQuery() as connection:
            cursor = connection.cursor()
            cursor.execute(''' SELECT ots_bn.soswell.welname, ots_bn.soswell.welfieldid 
                            FROM ots_bn.soswell WHERE welid = :wellid''', wellid=ID)
            rows = cursor.fetchone()
            wName = rows[0]
            cursor.execute(''' SELECT ots_bn.sosfield.fldname FROM ots_bn.sosfield
                            WHERE fldid = :fieldid''', fieldid=rows[1])
            rows = cursor.fetchone()
        return wName, rows[0]

class IntDelegate(QStyledItemDelegate):

    def createEditor(self, parent, option, index):
        lineEdit = QLineEdit(parent)
        validator = QIntValidator(10, 1000, lineEdit)
        lineEdit.setValidator(validator)
        return lineEdit

def insertDataToSosresearchTable(ppl):
    if ppl.tableModels is None:return
    RESID = ppl.resID
    RESCREATEDATETIME = datetime.datetime.now()
    RESMODULETYPEID = 'none'
    RESINTERPRETATORID = 'U5kpGq6aT42GXQopVH7PTA'
    RESSUBORGANIZATIONID = ppl.tpID
    RESGEOLOGISTORGANIZATIONID = ppl.tzehID
    RESRESEARCHTYPEID = 281
    RESFIELDID = ppl.otsFieldID
    RESWELLID = ppl.otsWellID
    RESCREATORID = 'U5kpGq6aT42GXQopVH7PTA'
    RESFIRSTMEASUREMENTSTAMP = ppl.firstMeasureDatetime.to_pydatetime()
    RESGOAL = 'Рпласт'
    RESSTATUSID = 20
    RESLASTMEASUREMENTSTAMP = ppl.lastMeasureDatetime.to_pydatetime()
    RESINBRIEF = 1
    row = [RESID,RESCREATEDATETIME,RESMODULETYPEID,RESINTERPRETATORID,RESSUBORGANIZATIONID,RESGEOLOGISTORGANIZATIONID,
           RESRESEARCHTYPEID,RESFIELDID,RESWELLID,RESCREATORID,RESFIRSTMEASUREMENTSTAMP,RESGOAL,RESSTATUSID,
           RESLASTMEASUREMENTSTAMP,RESINBRIEF]
    with sqlQuery() as connection:
        cursor = connection.cursor()
        sqlQ = ''' INSERT INTO ots_bn.sosresearch
                    (ots_bn.sosresearch.RESID,
                    ots_bn.sosresearch.RESCREATEDATETIME,
                    ots_bn.sosresearch.RESMODULETYPEID,
                    ots_bn.sosresearch.RESINTERPRETATORID,
                    ots_bn.sosresearch.RESSUBORGANIZATIONID,
                    ots_bn.sosresearch.RESGEOLOGISTORGANIZATIONID,
                    ots_bn.sosresearch.RESRESEARCHTYPEID,
                    ots_bn.sosresearch.RESFIELDID,
                    ots_bn.sosresearch.RESWELLID,
                    ots_bn.sosresearch.RESCREATORID,
                    ots_bn.sosresearch.RESFIRSTMEASUREMENTSTAMP,
                    ots_bn.sosresearch.RESGOAL,
                    ots_bn.sosresearch.RESSTATUSID,
                    ots_bn.sosresearch.RESLASTMEASUREMENTSTAMP,
                    ots_bn.sosresearch.RESINBRIEF) VALUES
                    (:1,:2,:3,:4,:5,:6,:7,:8,:9,:10,:11,:12,:13,:14,:15)'''
        cursor.execute(sqlQ, row)
        connection.commit()

def insertDataToSosresearchresultTable(Ppl):
    if Ppl.ro is None:
        print('no ppl.ro')
        return
    RSRRESEARCHID = Ppl.resID
    RSRCRITICALVOLUMEGASCONTENT = 20
    RSRRECKONSPEEDLOSS = 0
    RSRPIPEROUGHNESS = 0.0254
    RSRDENSITYLIQUID = Ppl.ro
    RSRRESULTMEASUREMENTID = Ppl.otsNewMesID
    RSRINFLOWTECHHALTTIME = 180
    RSREXTRACTZABOYLEVELTYPEID = 1
    with sqlQuery() as connection:
        cursor = connection.cursor()
        sqlQ = ''' INSERT INTO ots_bn.sosresearchresult
                    (RSRRESEARCHID,
                     RSRCRITICALVOLUMEGASCONTENT,
                     RSRRECKONSPEEDLOSS,
                     RSRPIPEROUGHNESS,
                     RSRDENSITYLIQUID,
                     RSRRESULTMEASUREMENTID,
                     RSRINFLOWTECHHALTTIME,
                     RSREXTRACTZABOYLEVELTYPEID) VALUES
                    (:RSRRESEARCHID,
                     :RSRCRITICALVOLUMEGASCONTENT,
                     :RSRRECKONSPEEDLOSS,
                     :RSRPIPEROUGHNESS,
                     :RSRDENSITYLIQUID,
                     :RSRRESULTMEASUREMENTID,
                     :RSRINFLOWTECHHALTTIME,
                     :RSREXTRACTZABOYLEVELTYPEID
                    )'''
        cursor.execute(sqlQ,
                       RSRRESEARCHID = RSRRESEARCHID,
                       RSRCRITICALVOLUMEGASCONTENT = RSRCRITICALVOLUMEGASCONTENT,
                       RSRRECKONSPEEDLOSS = RSRRECKONSPEEDLOSS,
                       RSRPIPEROUGHNESS = RSRPIPEROUGHNESS,
                       RSRDENSITYLIQUID = RSRDENSITYLIQUID,
                       RSRRESULTMEASUREMENTID = RSRRESULTMEASUREMENTID,
                       RSRINFLOWTECHHALTTIME =  RSRINFLOWTECHHALTTIME,
                       RSREXTRACTZABOYLEVELTYPEID =  RSREXTRACTZABOYLEVELTYPEID)
        connection.commit()

def insertDataToSosresearchmeasurementTable(ppl):
    if ppl.tableModels is None:
        print('error ppl')
        return
    doubleMeasureInSosmeasurement(ppl)
    RMSRESEARCHID = ppl.resID
    RMSMEASUREMENTID = ppl.otsNewMesID
    if ppl.tableInd == 0:
        RMSALGORITHMTYPE = 1
        RMSDESCENTDENSITYLIQUID = ppl.ro
        RMSASCENTDENSITYLIQUID = None
    else:
        RMSALGORITHMTYPE = 2
        RMSASCENTDENSITYLIQUID = ppl.ro
        RMSDESCENTDENSITYLIQUID = None
    RMSAVGTGRADIENT = ppl.avgTempGradient
    RMSGASOILBOUNDARY = ppl.GOB
    RMSWATEROILBOUNDARY = ppl.OWB
    RMSGASWATERBOUNDARY = ppl.GWB
    RMSSTATICLEVEL = ppl.staticLevel
    if RMSGASOILBOUNDARY is not None:
        RMSVERTICALGASOILBOUNDARY = RMSGASOILBOUNDARY - searchAndInterpolate(ppl.incl, RMSGASOILBOUNDARY)
    else:
        RMSVERTICALGASOILBOUNDARY = None
    if RMSWATEROILBOUNDARY is not None:
        RMSVERTICALWATEROILBOUNDARY = RMSWATEROILBOUNDARY - searchAndInterpolate(ppl.incl, RMSWATEROILBOUNDARY)
    else:
        RMSVERTICALWATEROILBOUNDARY = None
    if RMSGASWATERBOUNDARY is not None:
        RMSVERTICALGASWATERBOUNDARY = RMSGASWATERBOUNDARY - searchAndInterpolate(ppl.incl, RMSGASWATERBOUNDARY)
    else:
        RMSVERTICALGASWATERBOUNDARY = None
    if RMSSTATICLEVEL is not None:
        RMSVERTICALSTATICLEVEL = RMSSTATICLEVEL - searchAndInterpolate(ppl.incl, RMSSTATICLEVEL)
    else:
        RMSVERTICALSTATICLEVEL = None
    if ppl.tableInd == 1:
        RMSASCENTSHELFS = makeShelfsBLOB(ppl)
        RMSDESCENTSHELFS = None
    else:
        RMSASCENTSHELFS = None
        RMSDESCENTSHELFS = makeShelfsBLOB(ppl)

    with sqlQuery() as connection:
        cursor = connection.cursor()
        sqlQ = ''' INSERT INTO ots_bn.sosresearchmeasurement
                    (RMSRESEARCHID,
                    RMSMEASUREMENTID,
                    RMSALGORITHMTYPE,
                    RMSDESCENTDENSITYLIQUID,
                    RMSASCENTDENSITYLIQUID,
                    RMSAVGTGRADIENT,
                    RMSGASOILBOUNDARY,
                    RMSWATEROILBOUNDARY,
                    RMSGASWATERBOUNDARY,
                    RMSSTATICLEVEL,
                    RMSVERTICALGASOILBOUNDARY,
                    RMSVERTICALWATEROILBOUNDARY,
                    RMSVERTICALGASWATERBOUNDARY,
                    RMSVERTICALSTATICLEVEL,
                    RMSASCENTSHELFS,
                    RMSDESCENTSHELFS) VALUES
                    (:RMSRESEARCHID,
                    :RMSMEASUREMENTID,
                    :RMSALGORITHMTYPE,
                    :RMSDESCENTDENSITYLIQUID,
                    :RMSASCENTDENSITYLIQUID,
                    :RMSAVGTGRADIENT,
                    :RMSGASOILBOUNDARY,
                    :RMSWATEROILBOUNDARY,
                    :RMSGASWATERBOUNDARY,
                    :RMSSTATICLEVEL,
                    :RMSVERTICALGASOILBOUNDARY,
                    :RMSVERTICALWATEROILBOUNDARY,
                    :RMSVERTICALGASWATERBOUNDARY,
                    :RMSVERTICALSTATICLEVEL,
                    :RMSASCENTSHELFS,
                    :RMSDESCENTSHELFS
                    )'''
        cursor.execute(sqlQ,
                       RMSRESEARCHID =RMSRESEARCHID,
                       RMSMEASUREMENTID =RMSMEASUREMENTID,
                       RMSALGORITHMTYPE =RMSALGORITHMTYPE,
                       RMSDESCENTDENSITYLIQUID=RMSDESCENTDENSITYLIQUID,
                       RMSASCENTDENSITYLIQUID=RMSASCENTDENSITYLIQUID,
                       RMSAVGTGRADIENT=RMSAVGTGRADIENT,
                       RMSGASOILBOUNDARY=RMSGASOILBOUNDARY,
                       RMSWATEROILBOUNDARY=RMSWATEROILBOUNDARY,
                       RMSGASWATERBOUNDARY=RMSGASWATERBOUNDARY,
                       RMSSTATICLEVEL=RMSSTATICLEVEL,
                       RMSVERTICALGASOILBOUNDARY=RMSVERTICALGASOILBOUNDARY,
                       RMSVERTICALWATEROILBOUNDARY=RMSVERTICALWATEROILBOUNDARY,
                       RMSVERTICALGASWATERBOUNDARY=RMSVERTICALGASWATERBOUNDARY,
                       RMSVERTICALSTATICLEVEL=RMSVERTICALSTATICLEVEL,
                       RMSASCENTSHELFS=RMSASCENTSHELFS,
                       RMSDESCENTSHELFS=RMSDESCENTSHELFS)
        connection.commit()

def doubleMeasureInSosmeasurement(ppl):

    with sqlQuery() as connection:
        cursor = connection.cursor()
        mesID = ppl.otsMesID
        cursor.execute(''' SELECT * from ots_bn.sosmeasurement
                    WHERE ots_bn.sosmeasurement.mesID = :mesID''',
                       mesID=mesID)
        row = list(cursor.fetchone())
        originalID = row[0]
        row[0] = makeID()
        ppl.otsNewMesID = row[0]
        row[1] = ppl.resID
        row[15] = originalID
        row[16] = 'U5kpGq6aT42GXQopVH7PTA'
        row[17] = None
        row[18] = None
        row[22] = None
        row[-1] = '/fk5oAxKyUm2NL+7ZjkYTg'
        sqlQ = ''' INSERT INTO ots_bn.sosmeasurement VALUES 
        (:1,:2,:3,:4,:5,:6,:7,:8,:9,:10,:11,:12,:13,:14,:15,
        :16,:17,:18,:19,:20,:21,:22,:23,:24,:25)'''
        cursor.execute(sqlQ, row)
        connection.commit()
        doubleMeasureMtMeterInSosmeasurementmtmeter(ppl)

def doubleMeasureMtMeterInSosmeasurementmtmeter(ppl):
    with sqlQuery() as connection:
        cursor = connection.cursor()
        cursor.execute(''' SELECT * from ots_bn.sosmeasurementmtmeter
                    WHERE ots_bn.sosmeasurementmtmeter.mtmeasurementid = :mesid''',
                       mesid=ppl.otsMesID)
        row = list(cursor.fetchone())
        row[0] = ppl.otsNewMesID
        sqlQ = ''' INSERT INTO ots_bn.sosmeasurementmtmeter VALUES 
        (:1,:2,:3,:4,:5,:6,:7,:8,:9,:10,:11,:12,:13,:14,:15,:16,:17,:18)'''
        cursor.execute(sqlQ, row)
        connection.commit()


def makeShelfsBLOB(ppl):
    if ppl.finalData is None or ppl.tableModels is None:
        print('no data')
        return
    model = ppl.tableModels[ppl.tableInd]

    blob = b'\x01'
    for i in range(model.rowCount()):
        floatSeries = pd.Series(dtype='f')
        depth = float(model.item(i, 0).text())
        elongation = float(model.item(i, 1).text())
        vertDepth = depth - elongation
        pressure = float(model.item(i, 2).text())
        temperature = float(model.item(i, 3).text())
        try:
            if model.item(i, 6).checkState() == 2:
                blob += (b'\x01' + b'\x01')
            else:
                blob += (b'\x00' + b'\x01')
            ro = float(model.item(i, 4).text())
            interval = float(model.item(i, 0).text()) - float(model.item(i-1, 0).text())
            deltaPressure = pressure - float(model.item(i - 1, 2).text())
            deltaVertDepth = vertDepth - (float(model.item(i - 1, 0).text()) - float(model.item(i - 1, 1).text()))
            tempGradient = ((temperature - float(model.item(i - 1, 3).text())) / deltaVertDepth) * 100

        except: # первая строка
            ro = 0.
            deltaVertDepth = 0.
            interval = 0.
            deltaPressure = 0.
            tempGradient = 0.
            blob += (b'\x00' + b'\x00')
        if ppl.tableInd == 0:
            datetime = searchAndInterpolate(ppl.finalData.iloc[:int(ppl.centralDots[1]), [4, 3]], depth)
            blob += pd.Series((datetime - pd.Timestamp('1899-12-30')).total_seconds()/86400).to_numpy().tobytes()
        else:
            datetime = searchAndInterpolate(ppl.finalData.iloc[int(ppl.centralDots[1]):, [4, 3]], depth)
            blob += pd.Series((datetime - pd.Timestamp('1899-12-30')).total_seconds()/86400).to_numpy().tobytes()
        tempSeries = pd.Series([depth, vertDepth, pressure, temperature, interval, elongation, deltaPressure,
                                deltaVertDepth, ro, tempGradient, pressure, deltaPressure, ro], dtype='f')
        floatSeries=floatSeries.append(tempSeries, ignore_index=True)
        blob += floatSeries.to_numpy().tobytes()
    return blob


def insertDataToSosresearchgraphTable(ppl):
    RGRID = makeID()
    RGRRESEARCHID = ppl.resID
    RGRGRAPHTYPEID = 64
    RGRCOORDTYPEID = 32
    RGRORDER = '1'
    RGRFILENAME = 'khOCt7N/xkmfK+XrNUl3hw'
    plot = PlotWidget()
    fig = plot.plot(ppl.finalData, save=True)
    RGRPICTURE = fig.getvalue()
    with sqlQuery() as connection:
        cursor = connection.cursor()
        sqlQ = ''' INSERT INTO ots_bn.sosresearchgraph
                    (RGRID,
                     RGRRESEARCHID,
                     RGRGRAPHTYPEID,
                     RGRCOORDTYPEID,
                     RGRORDER,
                     RGRFILENAME,
                     RGRPICTURE) VALUES
                    (:RGRID,
                     :RGRRESEARCHID,
                     :RGRGRAPHTYPEID,
                     :RGRCOORDTYPEID,
                     :RGRORDER,
                     :RGRFILENAME,
                     :RGRPICTURE)'''
        cursor.execute(sqlQ,
                       RGRID=RGRID,
                       RGRRESEARCHID=RGRRESEARCHID,
                       RGRGRAPHTYPEID=RGRGRAPHTYPEID,
                       RGRCOORDTYPEID=RGRCOORDTYPEID,
                       RGRORDER=RGRORDER,
                       RGRFILENAME=RGRFILENAME,
                       RGRPICTURE=RGRPICTURE)
        connection.commit()

def insertDataToSosresearchmarkermeasurementTable(ppl):

    RMMRESEARCHID = ppl.resID
    RMMMEASUREMENTID = ppl.otsNewMesID
    RMMTYPEID = 5
    RMMDATE = datetime.datetime.now()
    RMMP = float(ppl.finalData.iloc[:, 1].max())
    RMMT = float(ppl.finalData.iloc[:, 2].max())
    with sqlQuery() as connection:
        cursor = connection.cursor()
        sqlQ = ''' INSERT INTO ots_bn.sosresearchmarkermeasurement
                    (RMMRESEARCHID,
                     RMMMEASUREMENTID,
                     RMMTYPEID,
                     RMMDATE,
                     RMMP,
                     RMMT) VALUES
                    (:RMMRESEARCHID,
                     :RMMMEASUREMENTID,
                     :RMMTYPEID,
                     :RMMDATE,
                     :RMMP,
                     :RMMT)'''
        cursor.execute(sqlQ,
                       RMMRESEARCHID=RMMRESEARCHID,
                       RMMMEASUREMENTID= RMMMEASUREMENTID,
                       RMMTYPEID=RMMTYPEID,
                       RMMDATE=RMMDATE,
                       RMMP=RMMP,
                       RMMT=RMMT)
        connection.commit()

def insertDataToSosresearchwellTable(ppl):
    inds = [None, 12, 11, 69, 66, 72, 75, 73, 76, 60, 77, 78, 79, 57,
            80, 81, 82, 83, None, None, None, 58, 59, 61, None, 15, 5,
            74, 0, None, 84, 56, None, 62, 63, 16, 21, 70, 71, 67, 68,
            85, 86, 6, 7, 8, 9, 10, 26, 36, 27, 39, 28, 37, 38, 29, 40,
            41, 32, 33, 42, 30, 35, 43, 34, None, None, None, 25, None,
            31, 46, 47, 48, 64, 65, 87, None, 112, 113, 88, 89, 90, 91,
            92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105,
            106, 107, 22, 23, 108, 109, None, None, 54, 55, 53, 98, None,
            None, None, None, 111, 116, 117, None, 119, 120, 121]
    with sqlQuery() as connection:
        cursor = connection.cursor()
        sqlQ = '''SELECT * FROM ots_bn.soswell
        WHERE ots_bn.soswell.welid = :welid'''
        cursor.execute(sqlQ, welid = ppl.otsWellID)
        row = cursor.fetchone()
        newRow = rowPermutation(row, inds)
        newRow[0] = makeID()
        ppl.otsResearchWellID = newRow[0]
        newRow[24] = 52
        newRow[29] = ppl.resID
        sqlQ = '''INSERT INTO ots_bn.sosresearchwell
        VALUES ('''
        for i in range(121):
            sqlQ += ':' + str(i + 1) + ','
        sqlQ = sqlQ.replace(':121,', ':121)')
        cursor.execute(sqlQ, newRow)
        connection.commit()


def insertDataToSosresearchvalidationTable(ppl):
    with sqlQuery() as connection:
        cursor = connection.cursor()
        row=[ppl.resID, datetime.datetime.now(), None, 'Интерпретатор 9000', 20]
        cursor.execute('''INSERT INTO ots_bn.sosresearchvalidation
        VALUES (:1,:2,:3,:4,:5)''', row)
        connection.commit()

def insertDataToSosresearchlayerinputTable(ppl):
    inds = [None, 2, 6, 8, 7, 4, 5, 9, 10, 11, 3,
            None, None, None, None, None, 17, 18,
            19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
    with sqlQuery() as connection:
        cursor = connection.cursor()
        sqlQ = '''SELECT * FROM ots_bn.sosbed
        WHERE ots_bn.sosbed.bedid = :bedid'''
        for layerID in ppl.layersIDs:
            cursor.execute(sqlQ, bedid=layerID)
            row = cursor.fetchone()
            newRow = rowPermutation(row, inds)
            newRow[0] = makeID()
            ppl.otsResearchLayerInputIds.append(newRow[0])
            newRow[11] = 1
            newRow[13] = ppl.otsResearchWellID
            newRow[14] = layerID
            cursor.execute('''INSERT INTO ots_bn.sosresearchlayerinput
            VALUES (:1,:2,:3,:4,:5,:6,:7,:8,:9,:10,:11,:12,:13,:14,:15,:16,:17,:18,
            :19,:20,:21,:22,:23,:24,:25,:26,:27,:28,:29)''', newRow)
            connection.commit()

def insertDataToSosresearchperforationTable(ppl):

    with sqlQuery() as connection:
        cursor = connection.cursor()
        for researchLayerID, bedID in zip(ppl.otsResearchLayerInputIds, ppl.layersIDs):
            sqlQ = '''SELECT * FROM ots_bn.sosperforation
                    WHERE (ots_bn.sosperforation.prfbedid = :bedID) AND
                    (ots_bn.sosperforation.prfwellid = :wellid)'''
            cursor.execute(sqlQ, bedid=bedID, wellid=ppl.otsWellID)
            rows = cursor.fetchall()
            minPerf = 9999999999
            for row in rows:
                newRow = [row[1], row[2], row[6], researchLayerID, row[5]]
                sqlQ = '''INSERT INTO ots_bn.sosresearchperforation
                VALUES (:1,:2,:3,:4,:5)'''
                cursor.execute(sqlQ, newRow)
                connection.commit()
                if row[1] < minPerf:
                    perf = row[1]
                    pvdp = ppl.ppl + (perf - searchAndInterpolate(ppl.incl, perf) - ppl.vdp + ppl.vdpElong) * ppl.ro / 10
                    sqlQ = '''SELECT ots_bn.sosbed.BEDWATEROILCONTACT FROM ots_bn.sosbed
                                WHERE (ots_bn.sosbed.bedID = :bedID)'''
                    cursor.execute(sqlQ, bedid=bedID)
                    vnk = cursor.fetchone()[0]
                    try:
                        pvnk = ppl.ppl + (vnk - ppl.vdp + ppl.vdpElong + ppl.altitude) * ppl.ro / 10
                    except:
                        pvnk = None
                    ppl.otsResearchMarkerLayer[researchLayerID] = [pvdp, pvnk]
                    minPerf = row[1]

def insertDataToSosresearchmarkerlayerTable(ppl):
    with sqlQuery() as connection:
        cursor = connection.cursor()
        for research_layer_id in ppl.otsResearchLayerInputIds:
            row = [research_layer_id, 4, datetime.datetime.now(), None,
                   ppl.otsResearchMarkerLayer[research_layer_id][0],
                   ppl.otsResearchMarkerLayer[research_layer_id][1], None, None]
            Sql_Q = '''INSERT INTO ots_bn.sosresearchmarkerlayer
            VALUES (:1, :2, :3, :4, :5, :6, :7, :8)'''
            cursor.execute(Sql_Q, row)
            connection.commit()

def rowPermutation(row, indexes):
    def permutator(List, i):
        try:
            return List[i]
        except:
            return i
    return [permutator(row, i) for i in indexes]


def makeID():
    return base64.b64encode(uuid.uuid4().bytes)[:-2].decode()


def searchAndInterpolateOld(searchingArray, x, xleft=True, interpolate=True):
    if xleft:
        column1 = searchingArray.iloc[:, 0].to_list()
        column2 = searchingArray.iloc[:, 1].to_list()
    else:
        column2 = searchingArray.iloc[:, 0].to_list()
        column1 = searchingArray.iloc[:, 1].to_list()
    if x in column1:
        if column1.count(x) == 1:
            ind = column1.index(x)
        else:
            sameIndexes = []
            for i in range(0, len(column1)):
                if column1[i] == x:
                    sameIndexes.append(i)
            ind = int(sum(sameIndexes) / len(sameIndexes))
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
def sqlQuery():
    userName = 'Shusharin'
    userPwd = 'Shusharin555'
    ip = 'oilteamsrv.bashneft.ru'
    port = 1521
    serviceName = 'OTS'
    connection = cx_Oracle.connect(user=userName,
                                   password=userPwd,
                                   dsn=cx_Oracle.makedsn(ip, port, service_name=serviceName))
    yield connection
    connection.close()


@contextmanager
def sqlQueryOld(db_name):
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


def to300Points(sample):
    def final300(sample):
        n = len(sample) // 300
        if n > 2:
            sample = sample.iloc[0::n]
            sample.reset_index(drop=True, inplace=True)
        new_index = pd.Series(((i + 1) / len(sample) * 300 for i in range(len(sample))))
        data = pd.concat((new_index, sample), axis=1)
        return pd.Series((searchAndInterpolate(data, x + 1, interpolate=False) for x in range(300)))

    def insertingNans(sample):
        oldArray = sample.values
        newArray = np.array((oldArray[0], None))
        for i in range(1, len(oldArray)):
            newArray = np.append(newArray, (oldArray[i], None))
        newArray = np.delete(newArray, -1)
        return pd.Series(newArray)

    if len(sample) == 300:
        return sample
    elif len(sample) < 300:
        while len(sample) < 300:
            sample = insertingNans(sample)
            sample = pd.to_numeric(sample)
            sample = sample.interpolate()
        return final300(sample)
    else:
        return final300(sample)


def searchAndInterpolate(searchingArray, x, interpolate=True):

    if searchingArray.iloc[-1, 0] < searchingArray.iloc[0, 0]:
        searchingArray = searchingArray.iloc[::-1]
        searchingArray.reset_index(inplace=True, drop=True)
    column1 = searchingArray.iloc[:, 0]
    column2 = searchingArray.iloc[:, 1]
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

