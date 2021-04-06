import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from numpy.polynomial import polynomial as Poly
from functools import lru_cache
import pylzma
from sys import byteorder
import Other
from collections import Counter
from Other import PlotWidget2
from PyQt5 import QtGui, QtCore
from statistics import median, mean

index = ['d4+A31plakuShf1sT9STqg', 'KP5sao1H3UK6UUOO5H/D4A',
         '9GJ+Twe4e0mFcQjZkVD2SQ', 'N43A6ZYo0kGxs9l2dXuFGg',
         'BJAEBMmWd0iOs+Do9bZQow', 'FQGY5JRISkiX3zA9vACUcA',
         'XDjnM3vEvkqOdxy7eXGRng', 'oJoVYbYX60qMPvrSPvSOww',
         'P4bX5rQoTUGIsv7IGGgCyA', '4i6xzt9yM02co5LjZPpBlw',
         'rGQSV/1mTU2vonzz8c3+5g', 'lz+Hn3TJ5EmOZJ0SlFsLig',
         'ljV0DivalEqVsTsEaOhf9Q', 's3tu+JZVG0uMZwzkQsbONA',
         'W7DbyFVNQU2O08sXoRhC/w', '8bIFLiWjTU2OYkGvhdPUiA',
         'DE8zAiQJlk+3FVNPVqewDQ', 'bWCOPj0Pm06wKXeqcSZTcw',
         'kKkS9q4SO0W8jYv9aMWCFw', 'sVqSoiaK/kiNqqX/p55B0g',
         '/ttwA0UDHUCh+ewqp86rRQ', '6QTgVIBI2ESPMH9uxXNthA',
         'SSgo1Mk3nUOGqDU5xWrMLA', '9eTuODmz10+c6WOnNVvpWQ',
         '3nHHNXnqzEiBTOmtwTvdSA', 'nWETMqQzFUGyJvxJJD1lpg',
         '8yjD2kzknEyW98zyQVHAtQ', '6G3itilU+EaotAz+b/npQA',
         'WtyowFIw/0KlzYfX2knV+Q', '5YDgbMsjJEChloLMYMq7Zw',
         'zv9zmmLrl0K8rPXPxqq9lA', 'lL9VzbdM9ESD5D+XX0r1gA',
         'AGDfWJdQf06qRL1FxnHK1A', 'Z70Tidoyzk6qPRND3GqnRA',
         'ZOum5njqOk6yCVokTR3uZg', 'DGhwyaj9wkmhawGxQVHajQ',
         'dqU/yoxTQEuSckKxDq0XjA', 'rVW6F7vK0ky4YL9l9vdMRQ',
         'XvFcM69cX0yL+JfS9ssiYg', 'LQYLj9WtHUe6qPUDMLBmbw',
         'boKlkyhA4kSlTEFXWs0ikQ', 'oi5DQYPvoEGJkd2RAQ5OlA',
         'jybjSiKUhEi8UFo9OC/f0Q', 'IJ0CVh3Aak6IIoyDKmgYJg',
         '+AsUpknvTk68g//Q+/U3iw', 'ecG/ppwmakqWH/mNWhYNXA',
         '07+m5sNNR0uVLbU64Ww5JQ', 'ZjLGZWls7USTgUNA5CvTbg',
         'AYQsW06moUy0UblPlG57gQ', 'Up3oBlfaREm1WppiIWSfng',
         'xHuwmEb/nUmHJ0s39aT6nQ', 'tdkIjfJo5UScNvrk8r6c+A',
         'uxtv8dXBCEqSqYyZxJ3CLg', 'cBGI/G/BCEC0ietd0z47/Q',
         'g/Um5pqeqU6x0tejw1SUwg', 'lYZfF4U+DESA4OJ3nZxikA',
         'Up/+cZsrHE6WLgU+HGCrwA', 'UZj9PYU3nUaNYsfanvXi+g',
         'rvRHY5GyZ06PUDqDx+d4SA', 'Wrje3M2mIE+/NPxgS8dL7g',
         'PGfeA4xlZkGF9K+0XoEP/A']
d = ['Rl2eZ/jJOkC5wWER/HenaQ', 'Rl2eZ/jJOkC5wWER/HenaQ',
     'Rl2eZ/jJOkC5wWER/HenaQ', 'Rl2eZ/jJOkC5wWER/HenaQ',
     'Rl2eZ/jJOkC5wWER/HenaQ', 'Rl2eZ/jJOkC5wWER/HenaQ',
     'Rl2eZ/jJOkC5wWER/HenaQ', 'Rl2eZ/jJOkC5wWER/HenaQ',
     'Rl2eZ/jJOkC5wWER/HenaQ', 'Rl2eZ/jJOkC5wWER/HenaQ',
     'Rl2eZ/jJOkC5wWER/HenaQ', 'Rl2eZ/jJOkC5wWER/HenaQ',
     'z2CdP3RuN0GYqifU5kBRog', 'z2CdP3RuN0GYqifU5kBRog',
     'z2CdP3RuN0GYqifU5kBRog', 'z2CdP3RuN0GYqifU5kBRog',
     'z2CdP3RuN0GYqifU5kBRog', 'z2CdP3RuN0GYqifU5kBRog',
     'z2CdP3RuN0GYqifU5kBRog', 'z2CdP3RuN0GYqifU5kBRog',
     'z2CdP3RuN0GYqifU5kBRog', 'z2CdP3RuN0GYqifU5kBRog',
     'z2CdP3RuN0GYqifU5kBRog', 'z2CdP3RuN0GYqifU5kBRog',
     'XX4ix3Zc1U6qlE7z7U2dFA', 'XX4ix3Zc1U6qlE7z7U2dFA',
     'XX4ix3Zc1U6qlE7z7U2dFA', 'XX4ix3Zc1U6qlE7z7U2dFA',
     'XX4ix3Zc1U6qlE7z7U2dFA', 'XX4ix3Zc1U6qlE7z7U2dFA',
     'XX4ix3Zc1U6qlE7z7U2dFA', 'XX4ix3Zc1U6qlE7z7U2dFA',
     'XX4ix3Zc1U6qlE7z7U2dFA', 'QJofzcONpE2zc67YmsJNig',
     'QJofzcONpE2zc67YmsJNig', 'QJofzcONpE2zc67YmsJNig',
     'QJofzcONpE2zc67YmsJNig', 'QJofzcONpE2zc67YmsJNig',
     'QJofzcONpE2zc67YmsJNig', 'QJofzcONpE2zc67YmsJNig',
     'QJofzcONpE2zc67YmsJNig', 'QJofzcONpE2zc67YmsJNig',
     'QJofzcONpE2zc67YmsJNig', 'QJofzcONpE2zc67YmsJNig',
     'QJofzcONpE2zc67YmsJNig', 'Rl2eZ/jJOkC5wWER/HenaQ',
     'Rl2eZ/jJOkC5wWER/HenaQ', 'bz/8lz6ekkGHgMIGKW0DLw',
     'bz/8lz6ekkGHgMIGKW0DLw', 'Rl2eZ/jJOkC5wWER/HenaQ',
     'Rl2eZ/jJOkC5wWER/HenaQ', '1AD7PvB7HUiPgfv/uik6lQ',
     'bz/8lz6ekkGHgMIGKW0DLw', 'bz/8lz6ekkGHgMIGKW0DLw',
     'bz/8lz6ekkGHgMIGKW0DLw', 'bz/8lz6ekkGHgMIGKW0DLw',
     'Rl2eZ/jJOkC5wWER/HenaQ', 'bz/8lz6ekkGHgMIGKW0DLw',
     'Rl2eZ/jJOkC5wWER/HenaQ', 'bz/8lz6ekkGHgMIGKW0DLw',
     'bz/8lz6ekkGHgMIGKW0DLw']

dict_suborg = {i: d for i, d in zip(index, d)}


class Ppl:

    def __init__(self, field, wname, data, new=False):
        self.field = field
        self.wellName = wname
        self.data = data
        self.resID = Other.makeID()
        self.tableModels = None
        self.checks = None
        self.tableInd = None
        self.ro = None
        self.ppl = None
        self.pvnk = None
        self.finalData = None
        self.finalFig = None
        self.supTimes = None
        self.dataFromSupTimes = None
        self.vdp = None
        self.layer = None
        self.otsFieldID = None
        self.otsWellID = None
        self.otsMesID = None
        self.otsNewMesID = None
        self.tpID = None
        self.region = None
        self.tzehID = None
        self.avgTempGradient = None
        self.GOB = None
        self.OWB = None
        self.GWB = None
        self.staticLevel = None
        self.maxDepth = None
        self.depthPressureTemperatureData=None
        self.layersIDs = None
        self.otsResearchWellID = None
        self.otsResearchLayerInputIds = []
        self.otsResearchMarkerLayer = {}
        self.centralDots = []
        self.warning = False
        if not new:
            self.getWellParamsOld()

    def makeDepthPressureTemperatureData(self):
        if self.finalData is None: return
        s1 = self.finalData.iloc[:, 4]
        s1.dropna(inplace=True, how='all')
        s2, s3 = pd.Series(), pd.Series()
        PresDF = self.finalData.iloc[:, [0, 1]]
        TemperDF = self.finalData.iloc[:, [0, 2]]
        for i in range(s1.size):
            s2 = s2.append(pd.Series([Other.searchAndInterpolate(PresDF, self.finalData.iloc[i, 3])]), ignore_index=True)
            s3 = s3.append(pd.Series([Other.searchAndInterpolate(TemperDF, self.finalData.iloc[i, 3])]), ignore_index=True)
        self.depthPressureTemperatureData = pd.concat([s1, s2, s3], axis=1)

    def determineStaticLevel(self):
        if self.GOB:
            self.staticLevel = self.preciseCalcStaticLevel(self.GOB)
            self.GOB = self.staticLevel
            return
        if self.GWB:
            self.staticLevel = self.preciseCalcStaticLevel(self.GWB)
            self.GWB = self.staticLevel

    def preciseCalcStaticLevel(self, referencePoint):
        lowBound = int(max(1, referencePoint - 50))
        highBound = int(min(self.maxDepth, referencePoint + 50))
        tempDF = self.finalData.iloc[:, [4, 3]]
        tempDF1 = tempDF.dropna(how='all')
        tempDF = self.finalData.iloc[:, [0, 1]]
        tempDF2 = tempDF.dropna(how='all')
        prevPres = 9000
        for i in range(lowBound, highBound):
            timeDepth = Other.searchAndInterpolateOld(tempDF1, i)
            newPres = Other.searchAndInterpolate(tempDF2, timeDepth)
            if (newPres - prevPres)*10 > 0.7:
                return i
            else:
                prevPres = newPres
        return referencePoint



    def calcAvgTempGradient(self):
        if self.tableInd is None: return
        model = self.tableModels[self.tableInd]
        dT = float(model.item(model.rowCount() - 1, 3).text()) - float(model.item(1, 3).text())
        dH = float(model.item(model.rowCount() - 1, 0).text()) - float(model.item(model.rowCount() - 1, 1).text()) - \
             float(model.item(1, 0).text()) + float(model.item(1, 1).text())
        self.avgTempGradient = 100 * dT / dH

    def calcPhaseBorders(self):
        if self.tableModels is None: return
        model = self.tableModels[self.tableInd]
        typesList = [model.item(i, 5).text() for i in range(1, model.rowCount())]
        depthList = [float(model.item(i, 0).text()) for i in range(1, model.rowCount())]
        self.GOB, self.OWB, self.GWB = calcBorders(typesList, depthList)



    def getWellParams(self):
        def sqlBed(bedID):
            with Other.sqlQuery() as connection:
                cursor = connection.cursor()
                cursor.execute('''SELECT ots_bn.sosbed.bedname from ots_bn.sosbed
                                WHERE ots_bn.sosbed.bedid = :bedid''',
                               bedid=bedID)
                row = cursor.fetchone()
            return row[0]

        with Other.sqlQuery() as connection:
            cursor = connection.cursor()
            cursor.execute('''SELECT ots_bn.sosperforation.PRFPERFORATEDZONETOP,
                            ots_bn.sosperforation.PRFBEDID
                            from ots_bn.sosperforation
                            WHERE ots_bn.sosperforation.prfwellid = :wellid''',
                           wellid=self.otsWellID)
            rows = cursor.fetchall()
            rows.sort()
            self.vdp = rows[0][0]
            self.layersIDs = [i[1] for i in rows]
            layer = [sqlBed(i[1]) for i in rows]
            self.layer = set(layer)
            cursor.execute('''SELECT ots_bn.soswell.WELANGULARITYTESTDEPTH,
                            ots_bn.soswell.WELANGULARITYTESTELONGATION,
                            ots_bn.soswell.welorganizationid,
                            ots_bn.soswell.welfieldid,
                            ots_bn.soswell.WELALTITUDE
                            from ots_bn.soswell
                            WHERE ots_bn.soswell.welid = :wellid''',
                           wellid=self.otsWellID)
            rows = cursor.fetchone()
            self.tzehID = rows[2]
            self.otsFieldID = rows[3]
            self.altitude = rows[4]
            try:
                bytesLen = int.from_bytes(rows[0].read()[3:7], byteorder=byteorder)
                depth = pd.Series(
                    np.frombuffer(pylzma.decompress(rows[0].read()[7:], maxlength=bytesLen), dtype=np.dtype('f')))
            except:
                depth = pd.Series(np.frombuffer(rows[0].read()[3:], dtype=np.dtype('f')))
            try:
                bytesLen = int.from_bytes(rows[1].read()[3:7], byteorder=byteorder)
                elong = pd.Series(
                    np.frombuffer(pylzma.decompress(rows[1].read()[7:], maxlength=bytesLen), dtype=np.dtype('f')))
            except:
                elong = pd.Series(np.frombuffer(rows[1].read()[3:], dtype=np.dtype('f')))

            self.incl = pd.concat([depth, elong], axis=1)
            self.vdpElong = round(Other.searchAndInterpolate(self.incl, self.vdp), 2)
            tempPressureTimeSeries = self.data.iloc[:, 0]
            tempPressureTimeSeries.dropna(inplace=True)
            self.researchDate = str(tempPressureTimeSeries[len(tempPressureTimeSeries) // 2])[:10]
            self.firstMeasureDatetime = tempPressureTimeSeries.iloc[0]
            self.lastMeasureDatetime = tempPressureTimeSeries.iloc[-1]
            self.tpID = dict_suborg[self.tzehID]
            self.maxDepth = self.data.iloc[:, 4].max()

    def getWellParamsOld(self):
        with Other.sqlQueryOld('FWDB.db') as query:
            query.exec("SELECT Wells.VDP FROM Wells "
                       "INNER JOIN BF2 "
                       "ON BF2.IDField=Wells.Field "
                       "WHERE Wells.Well=" + self.wellName +
                       " AND BF2.Field=" + '"' + self.field + '"')
            query.first()
            self.vdp = query.value("VDP")
            query.exec("SELECT Layers.Layer FROM Layers "
                       "INNER JOIN Well_Layer ON Layers.IDLayer=Well_Layer.IDL "
                       "INNER JOIN Wells ON Wells.IDWell=Well_Layer.IDW "
                       "INNER JOIN BF2 ON BF2.IDField=Wells.Field "
                       "WHERE Wells.Well=" + self.wellName +
                       " AND BF2.Field=" + '"' + self.field + '"')
            query.first()
            self.layer = query.value('Layer')

        tempPressureTimeSeries = self.data.iloc[:, 0]
        tempPressureTimeSeries.dropna(inplace=True)
        self.researchDate = str(tempPressureTimeSeries[len(tempPressureTimeSeries) // 2])[:10]
        if self.data.iloc[:, 0].dtype != 'datetime64[ns]':
            self.data.iloc[:, 0] = pd.to_datetime(self.data.iloc[:, 0], dayfirst=True)
        if self.data.iloc[:, 3].dtype != 'datetime64[ns]':
            self.data.iloc[:, 3] = pd.to_datetime(self.data.iloc[:, 3], dayfirst=True)
        self.incl = self.data.iloc[:, [5, 6]]
        self.incl.dropna(inplace=True, how='all')
        self.vdpElong = Other.searchAndInterpolate(self.incl, self.vdp)


    def interpret(self, delta, pModel, dModel):
        transformedData = self.transformData()
        oneZeroIndexes = self.divideEtImpera(transformedData, pModel, dModel)
        offsetSplittedData = self.OffsetAndSplitting(oneZeroIndexes)
        dataList, self.timesList = self.makeDataForModels(offsetSplittedData, delta)
        self.finalCalc(dataList, self.timesList, offsetSplittedData)
        return self

    def transformData(self):
        numPresDots = self.data.iloc[:, 0].count()
        numDepthDots = self.data.iloc[:, 4].count()
        print('KT = ', numPresDots, " KD = ", numDepthDots)
        tempListP = []
        for p in range(1, numPresDots - 1):
            tempListP.append((self.data.iloc[p + 1, 1] - self.data.iloc[p - 1, 1]) / (self.data.iloc[p + 1, 0] -
                                                                                      self.data.iloc[p - 1, 0]).total_seconds() / 86400)
        tempListP.append(tempListP[-1])
        tempListP.insert(0, tempListP[0])
        tempListTime1 = []
        for p in range(0, numPresDots):
            tempListTime1.append(self.data.iloc[p, 0].day +
                                 self.data.iloc[p, 0].hour / 24 +
                                 self.data.iloc[p, 0].minute / 1440 +
                                 self.data.iloc[p, 0].second / 86400)

        tempListD = []
        for p in range(1, numDepthDots - 1):
            tempListD.append((self.data.iloc[p + 1, 4] - self.data.iloc[p - 1, 4]) / (self.data.iloc[p + 1, 3] -
                                                                                      self.data.iloc[p - 1, 3]).total_seconds() / 86400)
        tempListD.append(tempListD[-1])
        tempListD.insert(0, tempListD[0])
        tempListTime2 = []
        for p in range(0, numDepthDots):
            tempListTime2.append(self.data.iloc[p, 3].day +
                                 self.data.iloc[p, 3].hour / 24 +
                                 self.data.iloc[p, 3].minute / 1440 +
                                 self.data.iloc[p, 3].second / 86400)
        maxP = max((p for p in tempListP if p < 10000))
        minP = min((p for p in tempListP if p > -10000))
        maxD = max((d for d in tempListD if d < 100000))
        midD = min((d for d in tempListD if d > -100000))
        ext1 = pd.Series(tempListTime1)
        ext2 = pd.Series(tempListP)
        ext3 = self.data.iloc[:, 1]
        ext4 = pd.Series(tempListTime2)
        ext5 = pd.Series(tempListD)
        ext6 = self.data.iloc[:, 4]
        ext2.mask(ext2 > 10000, maxP, inplace=True)
        ext2.mask(ext2 < -10000, minP, inplace=True)
        ext5.mask(ext5 > 100000, maxD, inplace=True)
        ext5.mask(ext5 < -100000, midD, inplace=True)
        prescaledSample = pd.concat([ext1, ext2, ext3, ext4, ext5, ext6], axis=1)
        return prescaledSample

    def divideEtImpera(self, receivedData, pModel, dModel):
        def calcATan(i, numDots, data):
            t0 = int(i - numDots) + 1
            t2 = int(i + numDots)
            pres1 = data.iloc[t0 + 1:i + 1, :].values
            pres2 = data.iloc[i:t2, :].values
            x1 = pres1[:, 0]
            y1 = pres1[:, 1]
            c1, stats1 = Poly.polyfit(x1, y1, 1, full=True)
            y_lin1 = Poly.polyval(x1, c1)
            r2 = r2_score(y1, y_lin1)
            k1 = c1[1]
            x2 = pres2[:, 0]
            y2 = pres2[:, 1]
            c2, stats2 = Poly.polyfit(x2, y2, 1, full=True)
            y_lin2 = Poly.polyval(x2, c2)
            r2 += r2_score(y2, y_lin2)
            k2 = c2[1]
            tan = abs((k1 - k2) / (1 + k1 * k2))
            return round(np.arctan(tan), 5)

        data = receivedData
        # maximum = self.data.iloc[:, 1].max()*0.99
        # num98Pressure = self.data.iloc[:, 1].gt(maximum).sum()/2
        # maximum = self.data.iloc[:, 4].max()*0.98
        # depth98Pressure = self.data.iloc[:, 4].gt(maximum).sum()/2
        numPressureDots = data.iloc[:, 0].count()
        numDepthDots = data.iloc[:, 3].count()
        presData = data.iloc[:numPressureDots, 0:3]
        depthData = data.iloc[:numDepthDots, 3:]
        scalerPressure = MinMaxScaler()
        scalerPressure.fit(presData)
        scalerDepth = MinMaxScaler()
        scalerDepth.fit(depthData)
        pipelinePressure = Pipeline([
            ("scaler", scalerPressure),
            ("etc", pModel)
        ])
        pipelineDepth = Pipeline([
            ("scaler", scalerDepth),
            ("etc", dModel)
        ])
        presPredict = pipelinePressure.predict(presData)
        depthPredict = pipelineDepth.predict(depthData)
        presPredict = presPredict.tolist().count(2) / 2
        depthPredict = depthPredict.tolist().count(2) / 2
        numStopDotsPressure = max(min(presPredict, 30), 5)
        numStopDotsDepth = max(min(depthPredict, 30), 5)
        finalIndexes = [None, None, None, None]

        # pressures
        presDataNoDerivative = presData.iloc[:, [0, 2]]
        presDataNoDerivative.iloc[:, 0] = presDataNoDerivative.iloc[:,
                                        0].apply(lambda x: x * 24 * 60 - presDataNoDerivative.iloc[:, 0].min())
        srez = presDataNoDerivative.iloc[:, 1]
        maxPres = srez.max()
        ind98 = []
        for i in range(srez.count()):
            if srez[i] > maxPres * 0.98 and len(ind98) == 0:
                ind98.append(i)
            if srez[i] < maxPres * 0.98 and len(ind98) == 1:
                ind98.append(i)

        min_feature = 0
        for i in range(ind98[0], int((ind98[0] + ind98[1]) / 2)):
            atan = calcATan(i, numStopDotsPressure, presDataNoDerivative)
            if atan > min_feature:
                min_feature = atan
                finalIndexes[0] = i

        min_feature = 0
        for i in range(int((ind98[0] + ind98[1]) / 2), ind98[1]):
            atan = calcATan(i, numStopDotsPressure, presDataNoDerivative)
            if atan > min_feature:
                min_feature = atan
                finalIndexes[1] = i

        # depths
        depthDataNoDeriv = depthData.iloc[:, [0, 2]]
        depthDataNoDeriv.iloc[:, 0] = depthDataNoDeriv.iloc[:, 0].apply(lambda x:
                                                                        x * 24 * 60 - depthDataNoDeriv.iloc[
                                                                                            :,
                                                                                            0].min())

        srez = depthDataNoDeriv.iloc[:, 1]
        maxDepth = srez.max()
        ind98 = []
        for i in range(srez.count()):
            if srez[i] > maxDepth * 0.98 and len(ind98) == 0:
                ind98.append(i)
            if srez[i] < maxDepth * 0.98 and len(ind98) == 1:
                ind98.append(i)
        min_feature = 0

        for i in range(ind98[0], int((ind98[0] + ind98[1]) / 2)):
            atan = calcATan(i, numStopDotsDepth, depthDataNoDeriv)
            if atan > min_feature:
                min_feature = atan
                finalIndexes[2] = i

        min_feature = 0

        for i in range(int((ind98[0] + ind98[1]) / 2), ind98[1]):
            atan = calcATan(i, numStopDotsDepth, depthDataNoDeriv)
            if atan > min_feature:
                min_feature = atan
                finalIndexes[3] = i
        return finalIndexes

    def OffsetAndSplitting(self, receivedIndexes):
        ind = receivedIndexes
        self.centralDots = [(ind[0] + ind[1]) / 2, (ind[2] + ind[3]) / 2]
        delimeterPressure = int((ind[0] + ind[1]) / 2)
        delimeterDepth = int((ind[2] + ind[3]) / 2)
        dt1 = self.data.iloc[ind[0], 0] - self.data.iloc[ind[2], 3]
        dt2 = self.data.iloc[ind[1], 0] - self.data.iloc[ind[3], 3]
        offsetData1 = self.data.copy()
        offsetData2 = self.data.copy()
        target_name = self.data.columns[3]
        offsetData1[target_name] = offsetData1[target_name].apply(lambda x: x + dt1)
        offsetData2[target_name] = offsetData2[target_name].apply(lambda x: x + dt2)
        db1 = pd.concat([offsetData1.iloc[:delimeterPressure, 0],
                         offsetData1.iloc[:delimeterPressure, 1],
                         offsetData1.iloc[:delimeterPressure, 2],
                         offsetData1.iloc[:delimeterDepth, 3],
                         offsetData1.iloc[:delimeterDepth, 4]], axis=1)
        preDB2_1 = pd.concat([offsetData2.iloc[delimeterPressure:, 0],
                               offsetData2.iloc[delimeterPressure:, 1],
                               offsetData2.iloc[delimeterPressure:, 2]], axis=1)
        preDB2_1.reset_index(inplace=True, drop=True)
        preDB2_2 = pd.concat([offsetData2.iloc[delimeterDepth:, 3],
                               offsetData2.iloc[delimeterDepth:, 4]], axis=1)
        preDB2_2.reset_index(inplace=True, drop=True)
        preDB2_2.dropna(inplace=True, how='all')
        db2 = pd.concat([preDB2_1, preDB2_2], axis=1)
        doubleData = [db1, db2]
        return doubleData

    def supportDots(self, data, interval):

        depth = []
        elongation = []
        pressure = []
        temperature = []
        time = []
        for shelf in data:
            tempDepths = []
            tempElongations = []
            tempPressures = []
            tempTemperatures = []
            tempTimes = []
            maxDepth = int(max(shelf.iloc[:, 4]))
            minDepth = int(max(min(shelf.iloc[:, 4]), 0))
            for depthI in range(minDepth, maxDepth):
                if depthI % interval == 0 and (maxDepth - depthI) > 30:
                    tempDepths.append(depthI)  # глубины
                    tempElongations.append(round(Other.searchAndInterpolate(self.incl, depthI), 2))  # удлинения
                    searchArray = shelf.iloc[:, [4, 3]]  # давления
                    searchArray.dropna(inplace=True, how='all')
                    tempTime = Other.searchAndInterpolate(searchArray, depthI)
                    tempTimes.append(tempTime)
                    searchArray = shelf.iloc[:, [0, 1]]
                    searchArray.dropna(inplace=True, how='all')
                    tempPressures.append(round(Other.searchAndInterpolate(searchArray, tempTime), 2))
                    searchArray = shelf.iloc[:, [0, 2]]  # температуры
                    searchArray.dropna(inplace=True, how='all')
                    tempTemperatures.append(round(Other.searchAndInterpolate(searchArray, tempTime), 2))
            tempDepths.append(maxDepth)  # глубины
            depth.append(tempDepths)
            tempElongations.append(round(Other.searchAndInterpolate(self.incl, maxDepth), 2))  # удлинения
            elongation.append(tempElongations)
            # давления
            searchArray = shelf.iloc[:, [4, 3]]
            searchArray.dropna(inplace=True, how='all')
            tempTime = Other.searchAndInterpolate(searchArray, maxDepth)
            searchArray = shelf.iloc[:, [0, 1]]
            searchArray.dropna(inplace=True, how='all')
            tempPressures.append(round(Other.searchAndInterpolate(searchArray, tempTime), 2))
            pressure.append(tempPressures)
            # температуры
            searchArray = shelf.iloc[:, [0, 2]]
            searchArray.dropna(inplace=True, how='all')
            tempTemperatures.append(round(Other.searchAndInterpolate(searchArray, tempTime), 2))
            temperature.append(tempTemperatures)
            tempTimes.append(tempTime)
            time.append(tempTimes)

        return depth, elongation, pressure, temperature, time

    def makeDataForModels(self, splittedData, delta):
        data = splittedData
        depths, elongations, pressures, temperatures, times = self.supportDots(data, delta)
        calcTable = []
        for i in range(0, 2):
            numDots = len(depths[i])
            sample = []
            for j in range(0, numDots):
                if j == 0:
                    ro = None
                else:
                    ro = round((pressures[i][j] - pressures[i][j - 1]) /
                               (depths[i][j] - elongations[i][j] - depths[i][j - 1] + elongations[i][j - 1]) * 10, 3)
                union = [depths[i][j], elongations[i][j], pressures[i][j], temperatures[i][j], ro]
                sample.append(union)
            calcTable.append(sample)


        for numShelf, shelf in enumerate(calcTable):
            a = shelf[-1][4]
            b = shelf[-2][4]
            c = shelf[-3][4]
            meanRo = (a + b + c) / 3
            cond1 = (max(a, meanRo) - a) / max(a, meanRo) > 0.15
            cond2 = (max(b, meanRo) - b) / max(b, meanRo) > 0.15
            cond3 = (max(c, meanRo) - c) / max(c, meanRo) > 0.15
            if cond1 or cond2 or cond3 or len(shelf) < 7:
                depths, elongations, pressures, temperatures, reTiming = self.supportDots(data, delta / 2)
                numDots = len(depths[numShelf])
                sample = []
                timeSample = []
                for j in range(0, numDots):
                    union = [depths[numShelf][j], elongations[numShelf][j], pressures[numShelf][j], temperatures[numShelf][j]]
                    sample.append(union)
                    timeSample.append(reTiming[numShelf][j])
                calcTable[numShelf] = sample
                times[numShelf] = timeSample
            for p in shelf:
                if len(p) == 5:
                    p.pop()

        return calcTable, times

    def finalCalc(self, dataList, timeList, newData):

        modelPair = []
        roPair = []
        pplPair = []
        fTypePair = []
        checksPair = []
        for polka in dataList:
            tempModel, ro, ppl, fType, checks = self.makeModel(polka)
            modelPair.append(tempModel)
            roPair.append(ro)
            pplPair.append(ppl)
            fTypePair.append(fType)
            checksPair.append(checks)
        self.tableModels = modelPair
        self.checks = checksPair
        self.supTimes = timeList
        self.dataFromSupTimes = dataList
        if fTypePair[0] == "Water":
            target = 1.16
        else:
            target = 0.88
        cond1 = (max(target, roPair[0]) - min(target, roPair[0])) / max(target, roPair[0])
        cond2 = (max(target, roPair[1]) - min(target, roPair[1])) / max(target, roPair[1])
        if self.checks[0] > self.checks[1]:
            self.tableInd = 0
            self.ro = roPair[0]
            self.ppl = pplPair[0]
        elif self.checks[0] < self.checks[1]:
            self.tableInd = 1
            self.ro = roPair[1]
            self.ppl = pplPair[1]
        else:
            if cond1 > cond2:
                self.tableInd = 0
                self.ro = roPair[0]
                self.ppl = pplPair[0]
            else:
                self.tableInd = 1
                self.ro = roPair[1]
                self.ppl = pplPair[1]

        if fTypePair[0] != fTypePair[1]:
            self.warning = True

        t1 = newData[0].iloc[:, [0, 1, 2]]
        t2 = newData[1].iloc[:, [0, 1, 2]]
        if t1.isnull().sum().sum() > 0:
            t1.dropna(inplace=True, how='all')
        if t2.isnull().sum().sum() > 0:
            t2.dropna(inplace=True, how='all')
        pres = pd.concat([t1, t2], axis=0)
        pres.reset_index(inplace=True, drop=True)
        t3 = newData[0].iloc[:, [3, 4]]
        t4 = newData[1].iloc[:, [3, 4]]
        if t3.isnull().sum().sum() > 0:
            t3.dropna(inplace=True, how='all')
        if t4.isnull().sum().sum() > 0:
            t4.dropna(inplace=True, how='all')
        dep = pd.concat([t3, t4], axis=0)
        dep.reset_index(inplace=True, drop=True)
        temp = pd.concat([pres, dep], axis=1)
        self.finalData = temp
        tempPlotWidget = PlotWidget2()
        self.finalFig = tempPlotWidget.plot(temp, save=True)
        self.calcAvgTempGradient()
        self.calcPhaseBorders()
        self.determineStaticLevel()

    def makeModel(self, halfData):
        model = QtGui.QStandardItemModel(len(halfData), 7)
        model.setHorizontalHeaderLabels(['Depth', 'Elongation', 'Pressure', 'Temperature', 'Density', 'Fluid type', ''])
        densities = [0]
        types = ['']
        for row in range(len(halfData)):
            for col in range(7):
                if col in range(4):
                    item = QtGui.QStandardItem(str(halfData[row][col]))
                elif col == 4:
                    if row != 0:
                        ro = round((halfData[row][2] - halfData[row - 1][2]) /
                                   (halfData[row][0] - halfData[row][1] - halfData[row - 1][0] +
                                    halfData[row - 1][
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
        numDots = round(len(halfData) / 3) + 1
        calcList = densities[-numDots:]
        typesList = types[-numDots:]
        refType = None
        if typesList[-1] == typesList[-2] and 0.7 < calcList[-1] < 1.25 and 0.7 < calcList[-2] < 1.25:
            refType = typesList[-1]
        elif typesList[-1] != typesList[-2] and 0.7 < calcList[-1] < 1.25 and 0.7 < calcList[-2] < 1.25:
            if typesList[-2] == typesList[-3] and 0.7 < calcList[-2] < 1.25 and 0.7 < calcList[-3] < 1.25:
                refType = typesList[-2]
            elif typesList[-2] != typesList[-3] and 0.7 < calcList[-2] < 1.25 and 0.7 < calcList[-3] < 1.25:
                if typesList[-3] == typesList[-4] and 0.7 < calcList[-3] < 1.25 and 0.7 < calcList[-4] < 1.25:
                    refType = typesList[-3]
            else:
                maxNum = 0
                typesSet = set(typesList)
                for fluidType in typesSet:
                    num = typesList.count(fluidType)
                    if num > maxNum:
                        maxNum = num
                        refType = fluidType
        else:
            maxNum = 0
            typesSet = set(typesList)
            for fluidType in typesSet:
                num = typesList.count(fluidType)
                if num > maxNum:
                    maxNum = num
                    refType = fluidType
        if refType == "Water":
            target = 1.16
            calcList = [ro for ro in calcList if ro >= 0.98]
        else:
            target = 0.88
            calcList = [ro for ro in calcList if 0.7 < ro < 0.98]
        if len(calcList) % 2 == 0:
            medianRo2 = median(calcList)
            indicator = False
            for i in range(len(calcList) - 1):
                if calcList[i] < medianRo2 < calcList[i + 1]:
                    medianRo1 = calcList[i]
                    medianRo3 = calcList[i + 1]
                    indicator = True
                if i == len(calcList) - 2 and indicator == False:
                    medianRo1 = medianRo2
                    medianRo3 = medianRo2
            cond1 = (max(target, medianRo1) - min(target, medianRo1)) / max(target, medianRo1)
            cond2 = (max(target, medianRo2) - min(target, medianRo2)) / max(target, medianRo2)
            cond3 = (max(target, medianRo3) - min(target, medianRo3)) / max(target, medianRo3)
            if cond1 == min(cond1, cond2, cond3):
                medianRo = medianRo1
            elif cond2 == min(cond1, cond2, cond3):
                medianRo = medianRo2
            else:
                medianRo = medianRo3
        else:
            medianRo = median(calcList)
        finalRo = []
        for row in range(len(halfData) - numDots, len(halfData)):
            m1 = max(float(model.index(row, 4).data()), medianRo)
            m2 = min(float(model.index(row, 4).data()), medianRo)
            if model.index(row, 5).data() == refType and (m1 - m2) / m1 < 0.08:
                model.item(row, 6).setCheckState(2)
                finalRo.append(float(model.item(row, 4).text()))
        if 0.8 > mean(finalRo) or 1.2 < mean(finalRo) or 0.8 > mean(finalRo) or 1.2 < mean(finalRo):
            self.warning = True
        mtmeterAbsDepth = float(model.item(len(halfData) - 1, 0).text()) - float(
            model.item(len(halfData) - 1, 1).text())
        vdpAbsDepth = self.vdp - self.vdpElong
        delta = vdpAbsDepth - mtmeterAbsDepth
        ppl = round(float(model.item(len(halfData) - 1, 2).text()) + delta * mean(finalRo) / 10, 3)
        return model, round(mean(finalRo), 3), ppl, refType, len(finalRo)

# class AutoInterpretation:
#
#     def __init__(self, vhoddata, incl=None, alt=False, delta=100):
#         self.data = vhoddata
#         self.alt = alt
#         self.delta = delta
#         if incl:
#             for d, i, index in zip(self.data, incl, range(len(self.data))):
#                 self.data[index] = pd.concat([d, i], axis=1)
#
#     def transform_data(self):
#         self.fed_data = []
#         for d in self.data:
#             kt_pres = d.iloc[:, 0].count()
#             kt_dep = d.iloc[:, 4].count()
#             print('KT = ', kt_pres, " KD = ", kt_dep)
#             templistp = []
#             for p in range(1, kt_pres - 1):
#                 templistp.append((d.iloc[p + 1, 1] - d.iloc[p - 1, 1]) / (d.iloc[p + 1, 0] -
#                                                                           d.iloc[p - 1, 0]).total_seconds() / 86400)
#             templistp.append(templistp[-1])
#             templistp.insert(0, templistp[0])
#             templisttime1 = []
#             for p in range(0, kt_pres):
#                 templisttime1.append(d.iloc[p, 0].day +
#                                      d.iloc[p, 0].hour / 24 +
#                                      d.iloc[p, 0].minute / 1440 +
#                                      d.iloc[p, 0].second / 86400)
#
#             templistd = []
#             for p in range(1, kt_dep - 1):
#                 templistd.append((d.iloc[p + 1, 4] - d.iloc[p - 1, 4]) / (d.iloc[p + 1, 3] -
#                                                                           d.iloc[p - 1, 3]).total_seconds() / 86400)
#             templistd.append(templistd[-1])
#             templistd.insert(0, templistd[0])
#             templisttime2 = []
#             for p in range(0, kt_dep):
#                 templisttime2.append(d.iloc[p, 3].day +
#                                      d.iloc[p, 3].hour / 24 +
#                                      d.iloc[p, 3].minute / 1440 +
#                                      d.iloc[p, 3].second / 86400)
#             p_max = max((p for p in templistp if p < 10000))
#             p_min = min((p for p in templistp if p > -10000))
#             d_max = max((d for d in templistd if d < 100000))
#             d_min = min((d for d in templistd if d > -100000))
#             ext1 = pd.Series(templisttime1)
#             ext2 = pd.Series(templistp)
#             ext3 = d.iloc[:, 1]
#             ext4 = pd.Series(templisttime2)
#             ext5 = pd.Series(templistd)
#             ext6 = d.iloc[:, 4]
#             ext2.mask(ext2 > 10000, p_max, inplace=True)
#             ext2.mask(ext2 < -10000, p_min, inplace=True)
#             ext5.mask(ext5 > 100000, d_max, inplace=True)
#             ext5.mask(ext5 < -100000, d_min, inplace=True)
#             prescaled_sample = pd.concat([ext1, ext2, ext3, ext4, ext5, ext6], axis=1)
#             self.fed_data.append(prescaled_sample)
#             prescaled_sample.to_clipboard()
#         return self.fed_data
#
#     def divide_et_impera(self):
#         data = self.transform_data()
#         modelp = joblib.load('rfc_model_pres.pkl')
#         modeld = joblib.load('rfc_model_depths.pkl')
#         finalindexes = []
#         for sample in data:
#             sampleindexes = []
#             kt_pres = sample.iloc[:, 0].count()
#             kt_dep = sample.iloc[:, 3].count()
#             pres_data = sample.iloc[:kt_pres, 0:3]
#             depth_data = sample.iloc[:kt_dep, 3:]
#             scalerp = MinMaxScaler()
#             scalerp.fit(pres_data)
#             scalerd = MinMaxScaler()
#             scalerd.fit(depth_data)
#             pipeline_pres = Pipeline([
#                 ("scaler", scalerp),
#                 ("etc", modelp)
#             ])
#             pipeline_depth = Pipeline([
#                 ("scaler", scalerd),
#                 ("etc", modeld)
#             ])
#             pres_pred = pipeline_pres.predict(pres_data)
#             depth_pred = pipeline_depth.predict(depth_data)
#             pres_pred = pres_pred.tolist()
#             depth_pred = depth_pred.tolist()
#             if self.alt:
#                 quan_pred = min(pres_pred.count(2) / 2, 30)
#                 quan_depth = min(depth_pred.count(2) / 2, 30)
#                 quan_pred = max(quan_pred, 5)
#                 quan_depth = max(quan_depth, 5)
#                 finalindexes2 = [None, None, None, None]
#
#                 # pressures
#                 pres_data_no_deriv = pres_data.iloc[:, [0, 2]]
#                 pres_data_no_deriv.iloc[:, 0] = pres_data_no_deriv.iloc[:,
#                                                 0].apply(lambda x: x * 24 * 60 - pres_data_no_deriv.iloc[:, 0].min())
#
#                 srez = pres_data_no_deriv.iloc[:, 1]
#                 max_pres = srez.max()
#                 ind98 = []
#                 for i in range(srez.count()):
#                     if srez[i] > max_pres * 0.98 and len(ind98) == 0:
#                         ind98.append(i)
#                     if srez[i] < max_pres * 0.98 and len(ind98) == 1:
#                         ind98.append(i)
#                 min_feature = 0
#                 for i in range(ind98[0], int((ind98[0] + ind98[1]) / 2)):
#                     t0 = int(i - quan_pred) + 1
#                     t2 = int(i + quan_pred)
#                     pres1 = pres_data_no_deriv.iloc[t0 + 1:i + 1, :].values
#                     pres2 = pres_data_no_deriv.iloc[i:t2, :].values
#                     x1 = pres1[:, 0]
#                     y1 = pres1[:, 1]
#                     c1, stats1 = Poly.polyfit(x1, y1, 1, full=True)
#                     y_lin1 = Poly.polyval(x1, c1)
#                     r2 = r2_score(y1, y_lin1)
#                     k1 = c1[1]
#                     x2 = pres2[:, 0]
#                     y2 = pres2[:, 1]
#                     c2, stats2 = Poly.polyfit(x2, y2, 1, full=True)
#                     y_lin2 = Poly.polyval(x2, c2)
#                     r2 += r2_score(y2, y_lin2)
#                     k2 = c2[1]
#                     tan = abs((k1 - k2) / (1 + k1 * k2))
#                     atan = round(np.arctan(tan), 5)
#                     if atan > min_feature:
#                         min_feature = atan
#                         finalindexes2[0] = i
#
#                 min_feature = 0
#
#                 for i in range(int((ind98[0] + ind98[1]) / 2), ind98[1]):
#                     t0 = int(i - quan_pred) + 1
#                     t2 = int(i + quan_pred)
#                     pres1 = pres_data_no_deriv.iloc[t0 + 1:i + 1, :].values
#                     pres2 = pres_data_no_deriv.iloc[i:t2, :].values
#                     x1 = pres1[:, 0]
#                     y1 = pres1[:, 1]
#                     c1, stats1 = Poly.polyfit(x1, y1, 1, full=True)
#                     y_lin1 = Poly.polyval(x1, c1)
#                     r2 = r2_score(y1, y_lin1)
#                     k1 = c1[1]
#                     x2 = pres2[:, 0]
#                     y2 = pres2[:, 1]
#                     c2, stats2 = Poly.polyfit(x2, y2, 1, full=True)
#                     y_lin2 = Poly.polyval(x2, c2)
#                     r2 += r2_score(y2, y_lin2)
#                     k2 = c2[1]
#                     tan = abs((k1 - k2) / (1 + k1 * k2))
#                     atan = round(np.arctan(tan), 5)
#                     if atan > min_feature:
#                         min_feature = atan
#                         finalindexes2[1] = i
#
#                 # depths
#                 depth_data_no_deriv = depth_data.iloc[:, [0, 2]]
#                 depth_data_no_deriv.iloc[:, 0] = depth_data_no_deriv.iloc[:, 0].apply(lambda x:
#                                                                                       x * 24 * 60 - depth_data_no_deriv.iloc[
#                                                                                                     :,
#                                                                                                     0].min())
#
#                 srez = depth_data_no_deriv.iloc[:, 1]
#                 max_depth = srez.max()
#                 ind98 = []
#                 for i in range(srez.count()):
#                     if srez[i] > max_depth * 0.98 and len(ind98) == 0:
#                         ind98.append(i)
#                     if srez[i] < max_depth * 0.98 and len(ind98) == 1:
#                         ind98.append(i)
#                 min_feature = 0
#
#                 for i in range(ind98[0], int((ind98[0] + ind98[1]) / 2)):
#                     t0 = int(i - quan_depth) + 1
#                     t2 = int(i + quan_depth)
#                     depth1 = depth_data_no_deriv.iloc[t0 + 1:i + 1, :].values
#                     depth2 = depth_data_no_deriv.iloc[i:t2, :].values
#                     x1 = depth1[:, 0]
#                     y1 = depth1[:, 1]
#                     c1, stats1 = Poly.polyfit(x1, y1, 1, full=True)
#                     y_lin1 = Poly.polyval(x1, c1)
#                     r2 = r2_score(y1, y_lin1)
#                     k1 = c1[1]
#                     x2 = depth2[:, 0]
#                     y2 = depth2[:, 1]
#                     c2, stats2 = Poly.polyfit(x2, y2, 1, full=True)
#                     y_lin2 = Poly.polyval(x2, c2)
#                     r2 += r2_score(y2, y_lin2)
#                     k2 = c2[1]
#                     tan = abs((k1 - k2) / (1 + k1 * k2))
#                     atan = round(np.arctan(tan), 5)
#                     if atan > min_feature:
#                         min_feature = atan
#                         finalindexes2[2] = i
#
#                 min_feature = 0
#
#                 for i in range(int((ind98[0] + ind98[1]) / 2), ind98[1]):
#                     t0 = int(i - quan_depth) + 1
#                     t2 = int(i + quan_depth)
#                     depth1 = depth_data_no_deriv.iloc[t0 + 1:i + 1, :].values
#                     depth2 = depth_data_no_deriv.iloc[i:t2, :].values
#                     x1 = depth1[:, 0]
#                     y1 = depth1[:, 1]
#                     c1, stats1 = Poly.polyfit(x1, y1, 1, full=True)
#                     y_lin1 = Poly.polyval(x1, c1)
#                     r2 = r2_score(y1, y_lin1)
#                     k1 = c1[1]
#                     x2 = depth2[:, 0]
#                     y2 = depth2[:, 1]
#                     c2, stats2 = Poly.polyfit(x2, y2, 1, full=True)
#                     y_lin2 = Poly.polyval(x2, c2)
#                     r2 += r2_score(y2, y_lin2)
#                     k2 = c2[1]
#                     tan = abs((k1 - k2) / (1 + k1 * k2))
#                     atan = round(np.arctan(tan), 5)
#                     if atan > min_feature:
#                         min_feature = atan
#                         finalindexes2[3] = i
#                 finalindexes.append(finalindexes2)
#             else:
#                 sum2 = 0
#                 indexes2 = []
#                 localindexes = []
#                 sums = []
#                 for i in range(0, len(pres_pred) - 1):
#                     if pres_pred[i] == 2 and pres_pred[i + 1] == 2:
#                         sum2 += 1
#                         localindexes.append(i)
#                     elif pres_pred[i] == 2 and pres_pred[i + 1] != 2:
#                         sums.append(sum2)
#                         sum2 = 0
#                         indexes2.append(localindexes)
#                         localindexes = []
#                 seeked_seque = sums.index(max(sums))
#                 pres_1_seeked_index, pres_2_seeked_index = indexes2[seeked_seque][0], indexes2[seeked_seque][-1]
#                 sampleindexes.append(pres_1_seeked_index)
#                 sampleindexes.append(pres_2_seeked_index + 1)
#
#                 # по глубинам
#                 sum2 = 0
#                 indexes2 = []
#                 localindexes = []
#                 sums = []
#                 for i in range(0, len(depth_pred) - 1):
#                     if depth_pred[i] == 2 and depth_pred[i + 1] == 2:
#                         sum2 += 1
#                         localindexes.append(i)
#                     elif depth_pred[i] == 2 and depth_pred[i + 1] != 2:
#                         sums.append(sum2)
#                         sum2 = 0
#                         indexes2.append(localindexes)
#                         localindexes = []
#                 seeked_seque = sums.index(max(sums))
#                 depth_1_seeked_index, depth_2_seeked_index = indexes2[seeked_seque][0], indexes2[seeked_seque][-1]
#                 sampleindexes.append(depth_1_seeked_index)
#                 sampleindexes.append(depth_2_seeked_index + 1)
#                 finalindexes.append(sampleindexes)
#         return finalindexes
#
#     @lru_cache()
#     def bias_and_splitting(self):
#         indexes = self.divide_et_impera()
#         freshdata = []
#         central_dots=[]
#         for i, d in enumerate(self.data):
#             ind = indexes[i]
#             central_dots.append([(ind[0] + ind[1]) / 2, (ind[2] + ind[3]) / 2])
#             delimeter_pres = int((ind[0] + ind[1]) / 2)
#             delimeter_depth = int((ind[2] + ind[3]) / 2)
#             dt1 = d.iloc[ind[0], 0] - d.iloc[ind[2], 3]
#             dt2 = d.iloc[ind[1], 0] - d.iloc[ind[3], 3]
#             datab1 = d.copy()
#             datab2 = d.copy()
#             target_name = d.columns[3]
#             datab1[target_name] = datab1[target_name].apply(lambda x: x + dt1)
#             datab2[target_name] = datab2[target_name].apply(lambda x: x + dt2)
#             db1 = pd.concat([datab1.iloc[:delimeter_pres, 0],
#                              datab1.iloc[:delimeter_pres, 1],
#                              datab1.iloc[:delimeter_pres, 2],
#                              datab1.iloc[:delimeter_depth, 3],
#                              datab1.iloc[:delimeter_depth, 4]], axis=1)
#             pre_db2_1 = pd.concat([datab2.iloc[delimeter_pres:, 0],
#                                    datab2.iloc[delimeter_pres:, 1],
#                                    datab2.iloc[delimeter_pres:, 2]], axis=1)
#             pre_db2_1.reset_index(inplace=True, drop=True)
#             pre_db2_2 = pd.concat([datab2.iloc[delimeter_depth:, 3],
#                                    datab2.iloc[delimeter_depth:, 4]], axis=1)
#             pre_db2_2.reset_index(inplace=True, drop=True)
#             pre_db2_2.dropna(inplace=True, how='all')
#             db2 = pd.concat([pre_db2_1, pre_db2_2], axis=1)
#             double_data = [db1, db2]
#             freshdata.append(double_data)
#         return freshdata, central_dots
#
#     def support_dots(self, incles, data, intervals):
#         depths = []
#         elongations = []
#         pressures = []
#         temperatures = []
#         times = []
#         n = 0
#         for i, d in zip(incles, data):
#             tempd = []
#             tempe = []
#             tempp = []
#             tempt = []
#             tempti = []
#             for polka in d:
#                 temp_depths = []
#                 temp_elongations = []
#                 temp_pressures = []
#                 temp_temperatures = []
#                 temp_times = []
#                 max_depth = int(max(polka.iloc[:, 4]))
#                 min_depth = int(max(min(polka.iloc[:, 4]), 0))
#                 for depth in range(min_depth, max_depth):
#                     if depth % intervals[n] == 0 and (max_depth - depth) > 30:
#                         temp_depths.append(depth)  # глубины
#                         temp_elongations.append(round(Other.searchAndInterpolate(i, depth), 2))  # удлинения
#                         search_array = polka.iloc[:, [4, 3]]  # давления
#                         search_array.dropna(inplace=True, how='all')
#                         temp_time = Other.searchAndInterpolate(search_array, depth)
#                         temp_times.append(temp_time)
#                         search_array = polka.iloc[:, [0, 1]]
#                         search_array.dropna(inplace=True, how='all')
#                         temp_pressures.append(round(Other.searchAndInterpolate(search_array, temp_time), 2))
#                         search_array = polka.iloc[:, [0, 2]]  # температуры
#                         search_array.dropna(inplace=True, how='all')
#                         temp_temperatures.append(round(Other.searchAndInterpolate(search_array, temp_time), 2))
#                 temp_depths.append(max_depth)  # глубины
#                 tempd.append(temp_depths)
#                 temp_elongations.append(round(Other.searchAndInterpolate(i, max_depth), 2))  # удлинения
#                 tempe.append(temp_elongations)
#                 # давления
#                 search_array = polka.iloc[:, [4, 3]]
#                 search_array.dropna(inplace=True, how='all')
#                 temp_time = Other.searchAndInterpolate(search_array, max_depth)
#                 search_array = polka.iloc[:, [0, 1]]
#                 search_array.dropna(inplace=True, how='all')
#                 temp_pressures.append(round(Other.searchAndInterpolate(search_array, temp_time), 2))
#                 tempp.append(temp_pressures)
#                 # температуры
#                 search_array = polka.iloc[:, [0, 2]]
#                 search_array.dropna(inplace=True, how='all')
#                 temp_temperatures.append(round(Other.searchAndInterpolate(search_array, temp_time), 2))
#                 tempt.append(temp_temperatures)
#                 temp_times.append(temp_time)
#                 tempti.append(temp_times)
#             n += 1
#             depths.append(tempd)
#             elongations.append(tempe)
#             pressures.append(tempp)
#             temperatures.append(tempt)
#             times.append(tempti)
#         return depths, elongations, pressures, temperatures, times
#
#     def zips(self):
#         data,_ = self.bias_and_splitting()
#         incles = []
#         for d in self.data:
#             incl = d.iloc[:, [5, 6]]
#             incl.dropna(inplace=True, how='all')
#             incles.append(incl)
#         intervals = [self.delta for d in data]
#         depths, elongations, pressures, temperatures, times = self.support_dots(incles, data, intervals)
#         calctables = []
#         for d, e, p, t in zip(depths, elongations, pressures, temperatures):
#             calctable = []
#             for i in range(0, 2):
#                 kt = len(d[i])
#                 sample = []
#                 for j in range(0, kt):
#                     if j == 0:
#                         ro = None
#                     else:
#                         ro = round((p[i][j] - p[i][j - 1]) / (d[i][j] - e[i][j] - d[i][j - 1] + e[i][j - 1]) * 10, 3)
#                     union = [d[i][j], e[i][j], p[i][j], t[i][j], ro]
#                     sample.append(union)
#                 calctable.append(sample)
#             calctables.append(calctable)
#         for i, t in enumerate(calctables):
#             for nomer_polki, polka in enumerate(t):
#                 a = polka[-1][4]
#                 b = polka[-2][4]
#                 c = polka[-3][4]
#                 mean_ro = (a + b + c) / 3
#                 cond1 = (max(a, mean_ro) - a) / max(a, mean_ro) > 0.15
#                 cond2 = (max(b, mean_ro) - b) / max(b, mean_ro) > 0.15
#                 cond3 = (max(c, mean_ro) - c) / max(c, mean_ro) > 0.15
#                 if cond1 or cond2 or cond3 or len(polka) < 7:
#                     temp_incles = [incles[i], ]
#                     temp_data = [data[i], ]
#                     depths, elongations, pressures, temperatures, retiming = self.support_dots(temp_incles, temp_data,
#                                                                                                intervals=[
#                                                                                                    self.delta / 2, ])
#
#                     for d, e, p, te, tim in zip(depths, elongations, pressures, temperatures, retiming):
#                         kt = len(d[nomer_polki])
#                         sample = []
#                         time_sample = []
#                         for j in range(0, kt):
#                             union = [d[nomer_polki][j], e[nomer_polki][j], p[nomer_polki][j], te[nomer_polki][j]]
#                             sample.append(union)
#                             time_sample.append(tim[nomer_polki][j])
#                         calctables[i][nomer_polki] = sample
#                         times[i][nomer_polki] = time_sample
#                 for p in polka:
#                     if len(p) == 5:
#                         p.pop()
#         print(sample, times)
#         #print(calctables, times)
#         return calctables, times
#
#
def PplFabric(field, wname, data):
    return Ppl(field, wname, data, True)

def calcBorders(typeList, depthShelfs):
    def disintegrateAbortions(typeList):
        newList = list(typeList)
        length = len(newList)
        if length < 5: return
        for j in range(2, length - 3):
            left = [newList[j - 2], newList[j - 1]]
            right = [newList[j + 1], newList[j + 2]]
            if (newList[j] not in left) and (newList[j] not in right):
                if len(set(left).intersection(set(right))) == 0:
                    continue
                counter = Counter(left + right)
                newList[j] = counter.most_common(1)[0][0]

        return newList

    GOB = None
    OWB = None
    GWB = None
    typeList = disintegrateAbortions(typeList)
    uniqueSet = set(typeList)
    sortedListOfPhases = sorted(uniqueSet)
    if len(uniqueSet) == 2:
        index = typeList.index(sortedListOfPhases[-1])
        if sortedListOfPhases == ['Gas', 'Oil']:
            GOB = (depthShelfs[index] + depthShelfs[index - 1]) / 2
        elif sortedListOfPhases == ['Oil', 'Water']:
            OWB = (depthShelfs[index] + depthShelfs[index - 1]) / 2
        elif sortedListOfPhases == ['Gas', 'Water']:
            GWB = (depthShelfs[index] + depthShelfs[index - 1]) / 2
    elif len(uniqueSet) == 3:
        sequence = []
        for i in typeList:
            if i not in sequence:
                sequence.append(i)
        if sequence == ['Gas', 'Oil', 'Water']:
            index1 = typeList.index(sortedListOfPhases[1])
            index2 = typeList.index(sortedListOfPhases[2])
            GOB = (depthShelfs[index1] + depthShelfs[index1 - 1]) / 2
            OWB = (depthShelfs[index2] + depthShelfs[index2 - 1]) / 2
    return GOB, OWB, GWB