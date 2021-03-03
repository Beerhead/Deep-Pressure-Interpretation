import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from numpy.polynomial import polynomial as Poly
from functools import lru_cache

import Other


class Ppl:

    def __init__(self, field, wname, data, new=False):
        self.field = field
        self.well_name = wname
        self.data = data
        self.table_models = None
        self.checks = None
        self.table_ind = None
        self.ro = None
        self.ppl = None
        self.final_data = None
        self.final_fig = None
        self.sup_times = None
        self.data_from_sup_times = None
        self.vdp = None
        self.layer = None
        if not new:
            self.get_well_params()

    def get_well_params(self):
        with Other.sql_query_old('FWDB.db') as query:
            query.exec("SELECT Wells.VDP FROM Wells "
                       "INNER JOIN BF2 "
                       "ON BF2.IDField=Wells.Field "
                       "WHERE Wells.Well=" + self.well_name +
                       " AND BF2.Field=" + '"' + self.field + '"')
            query.first()
            self.vdp = query.value("VDP")
            query.exec("SELECT Layers.Layer FROM Layers "
                       "INNER JOIN Well_Layer ON Layers.IDLayer=Well_Layer.IDL "
                       "INNER JOIN Wells ON Wells.IDWell=Well_Layer.IDW "
                       "INNER JOIN BF2 ON BF2.IDField=Wells.Field "
                       "WHERE Wells.Well=" + self.well_name +
                       " AND BF2.Field=" + '"' + self.field + '"')
            query.first()
            self.layer = query.value('Layer')

        temp_pressure_time_series = self.data.iloc[:, 0]
        temp_pressure_time_series.dropna(inplace=True)
        self.research_date = str(temp_pressure_time_series[len(temp_pressure_time_series) // 2])[:10]
        if self.data.iloc[:, 0].dtype != 'datetime64[ns]':
            self.data.iloc[:, 0] = pd.to_datetime(self.data.iloc[:, 0], dayfirst=True)
        if self.data.iloc[:, 3].dtype != 'datetime64[ns]':
            self.data.iloc[:, 3] = pd.to_datetime(self.data.iloc[:, 3], dayfirst=True)
        self.incl = self.data.iloc[:, [5, 6]]
        self.incl.dropna(inplace=True, how='all')
        self.vdp_elong = Other.search_and_interpolate(self.incl, self.vdp)


class AutoInterpretation:

    def __init__(self, vhoddata, alt=False):
        self.data = vhoddata
        self.alt = alt

    def transform_data(self):
        print("TRANSFORM DATA")
        self.fed_data = []
        for d in self.data:
            kt_pres = d.iloc[:, 0].count()
            kt_dep = d.iloc[:, 4].count()
            templistp = []
            for p in range(1, kt_pres - 1):
                templistp.append((d.iloc[p + 1, 1] - d.iloc[p - 1, 1]) / (d.iloc[p + 1, 0] -
                                                                          d.iloc[p - 1, 0]).total_seconds() / 86400)
            templistp.append(templistp[-1])
            templistp.insert(0, templistp[0])
            templisttime1 = []
            for p in range(0, kt_pres):
                templisttime1.append(d.iloc[p, 0].day +
                                     d.iloc[p, 0].hour / 24 +
                                     d.iloc[p, 0].minute / 1440 +
                                     d.iloc[p, 0].second / 86400)

            templistd = []
            for p in range(1, kt_dep - 1):
                templistd.append((d.iloc[p + 1, 4] - d.iloc[p - 1, 4]) / (d.iloc[p + 1, 3] -
                                                                          d.iloc[p - 1, 3]).total_seconds() / 86400)
            templistd.append(templistd[-1])
            templistd.insert(0, templistd[0])
            templisttime2 = []
            for p in range(0, kt_dep):
                templisttime2.append(d.iloc[p, 3].day +
                                     d.iloc[p, 3].hour / 24 +
                                     d.iloc[p, 3].minute / 1440 +
                                     d.iloc[p, 3].second / 86400)
            p_max = max((p for p in templistp if p < 10000))
            p_min = min((p for p in templistp if p > -10000))
            d_max = max((d for d in templistd if d < 100000))
            d_min = min((d for d in templistd if d > -100000))
            ext1 = pd.Series(templisttime1)
            ext2 = pd.Series(templistp)
            ext3 = d.iloc[:, 1]
            ext4 = pd.Series(templisttime2)
            ext5 = pd.Series(templistd)
            ext6 = d.iloc[:, 4]
            ext2.mask(ext2 > 10000, p_max, inplace=True)
            ext2.mask(ext2 < -10000, p_min, inplace=True)
            ext5.mask(ext5 > 100000, d_max, inplace=True)
            ext5.mask(ext5 < -100000, d_min, inplace=True)
            prescaled_sample = pd.concat([ext1, ext2, ext3, ext4, ext5, ext6], axis=1)
            self.fed_data.append(prescaled_sample)
        return self.fed_data

    def divide_et_impera(self):
        print("DIVIDE ET IMPERA")
        data = self.transform_data()
        modelp = joblib.load('rfc_model_pres.pkl')
        modeld = joblib.load('rfc_model_depths.pkl')
        finalindexes = []
        for sample in data:
            sampleindexes = []
            kt_pres = sample.iloc[:, 0].count()
            kt_dep = sample.iloc[:, 3].count()
            pres_data = sample.iloc[:kt_pres, 0:3]
            depth_data = sample.iloc[:kt_dep, 3:]
            scalerp = MinMaxScaler()
            scalerp.fit(pres_data)
            scalerd = MinMaxScaler()
            scalerd.fit(depth_data)
            pipeline_pres = Pipeline([
                ("scaler", scalerp),
                ("etc", modelp)
            ])
            pipeline_depth = Pipeline([
                ("scaler", scalerd),
                ("etc", modeld)
            ])
            pres_pred = pipeline_pres.predict(pres_data)
            depth_pred = pipeline_depth.predict(depth_data)
            pres_pred = pres_pred.tolist()
            depth_pred = depth_pred.tolist()
            if self.alt:
                quan_pred = min(pres_pred.count(2) / 2, 30)
                quan_depth = min(depth_pred.count(2) / 2, 30)
                quan_pred = max(quan_pred, 5)
                quan_depth = max(quan_depth, 5)
                finalindexes2 = [None, None, None, None]

                # pressures
                pres_data_no_deriv = pres_data.iloc[:, [0, 2]]
                pres_data_no_deriv.iloc[:, 0] = pres_data_no_deriv.iloc[:,
                                                0].apply(lambda x: x * 24 * 60 - pres_data_no_deriv.iloc[:, 0].min())

                srez = pres_data_no_deriv.iloc[:, 1]
                max_pres = srez.max()
                ind98 = []
                for i in range(srez.count()):
                    if srez[i] > max_pres * 0.98 and len(ind98) == 0:
                        ind98.append(i)
                    if srez[i] < max_pres * 0.98 and len(ind98) == 1:
                        ind98.append(i)
                min_feature = 0
                for i in range(ind98[0], int((ind98[0] + ind98[1]) / 2)):
                    t0 = int(i - quan_pred) + 1
                    t2 = int(i + quan_pred)
                    pres1 = pres_data_no_deriv.iloc[t0 + 1:i + 1, :].values
                    pres2 = pres_data_no_deriv.iloc[i:t2, :].values
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
                    atan = round(np.arctan(tan), 5)
                    if atan > min_feature:
                        min_feature = atan
                        finalindexes2[0] = i

                min_feature = 0

                for i in range(int((ind98[0] + ind98[1]) / 2), ind98[1]):
                    t0 = int(i - quan_pred) + 1
                    t2 = int(i + quan_pred)
                    pres1 = pres_data_no_deriv.iloc[t0 + 1:i + 1, :].values
                    pres2 = pres_data_no_deriv.iloc[i:t2, :].values
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
                    atan = round(np.arctan(tan), 5)
                    if atan > min_feature:
                        min_feature = atan
                        finalindexes2[1] = i

                # depths
                depth_data_no_deriv = depth_data.iloc[:, [0, 2]]
                depth_data_no_deriv.iloc[:, 0] = depth_data_no_deriv.iloc[:, 0].apply(lambda x:
                                                                                      x * 24 * 60 - depth_data_no_deriv.iloc[
                                                                                                    :,
                                                                                                    0].min())

                srez = depth_data_no_deriv.iloc[:, 1]
                max_depth = srez.max()
                ind98 = []
                for i in range(srez.count()):
                    if srez[i] > max_depth * 0.98 and len(ind98) == 0:
                        ind98.append(i)
                    if srez[i] < max_depth * 0.98 and len(ind98) == 1:
                        ind98.append(i)
                min_feature = 0

                for i in range(ind98[0], int((ind98[0] + ind98[1]) / 2)):
                    t0 = int(i - quan_depth) + 1
                    t2 = int(i + quan_depth)
                    depth1 = depth_data_no_deriv.iloc[t0 + 1:i + 1, :].values
                    depth2 = depth_data_no_deriv.iloc[i:t2, :].values
                    x1 = depth1[:, 0]
                    y1 = depth1[:, 1]
                    c1, stats1 = Poly.polyfit(x1, y1, 1, full=True)
                    y_lin1 = Poly.polyval(x1, c1)
                    r2 = r2_score(y1, y_lin1)
                    k1 = c1[1]
                    x2 = depth2[:, 0]
                    y2 = depth2[:, 1]
                    c2, stats2 = Poly.polyfit(x2, y2, 1, full=True)
                    y_lin2 = Poly.polyval(x2, c2)
                    r2 += r2_score(y2, y_lin2)
                    k2 = c2[1]
                    tan = abs((k1 - k2) / (1 + k1 * k2))
                    atan = round(np.arctan(tan), 5)
                    if atan > min_feature:
                        min_feature = atan
                        finalindexes2[2] = i

                min_feature = 0

                for i in range(int((ind98[0] + ind98[1]) / 2), ind98[1]):
                    t0 = int(i - quan_depth) + 1
                    t2 = int(i + quan_depth)
                    depth1 = depth_data_no_deriv.iloc[t0 + 1:i + 1, :].values
                    depth2 = depth_data_no_deriv.iloc[i:t2, :].values
                    x1 = depth1[:, 0]
                    y1 = depth1[:, 1]
                    c1, stats1 = Poly.polyfit(x1, y1, 1, full=True)
                    y_lin1 = Poly.polyval(x1, c1)
                    r2 = r2_score(y1, y_lin1)
                    k1 = c1[1]
                    x2 = depth2[:, 0]
                    y2 = depth2[:, 1]
                    c2, stats2 = Poly.polyfit(x2, y2, 1, full=True)
                    y_lin2 = Poly.polyval(x2, c2)
                    r2 += r2_score(y2, y_lin2)
                    k2 = c2[1]
                    tan = abs((k1 - k2) / (1 + k1 * k2))
                    atan = round(np.arctan(tan), 5)
                    if atan > min_feature:
                        min_feature = atan
                        finalindexes2[3] = i
                finalindexes.append(finalindexes2)
            else:
                sum2 = 0
                indexes2 = []
                localindexes = []
                sums = []
                for i in range(0, len(pres_pred) - 1):
                    if pres_pred[i] == 2 and pres_pred[i + 1] == 2:
                        sum2 += 1
                        localindexes.append(i)
                    elif pres_pred[i] == 2 and pres_pred[i + 1] != 2:
                        sums.append(sum2)
                        sum2 = 0
                        indexes2.append(localindexes)
                        localindexes = []
                seeked_seque = sums.index(max(sums))
                pres_1_seeked_index, pres_2_seeked_index = indexes2[seeked_seque][0], indexes2[seeked_seque][-1]
                sampleindexes.append(pres_1_seeked_index)
                sampleindexes.append(pres_2_seeked_index + 1)

                # по глубинам
                sum2 = 0
                indexes2 = []
                localindexes = []
                sums = []
                for i in range(0, len(depth_pred) - 1):
                    if depth_pred[i] == 2 and depth_pred[i + 1] == 2:
                        sum2 += 1
                        localindexes.append(i)
                    elif depth_pred[i] == 2 and depth_pred[i + 1] != 2:
                        sums.append(sum2)
                        sum2 = 0
                        indexes2.append(localindexes)
                        localindexes = []
                seeked_seque = sums.index(max(sums))
                depth_1_seeked_index, depth_2_seeked_index = indexes2[seeked_seque][0], indexes2[seeked_seque][-1]
                sampleindexes.append(depth_1_seeked_index)
                sampleindexes.append(depth_2_seeked_index + 1)
                finalindexes.append(sampleindexes)
        print(finalindexes)
        return finalindexes

    @lru_cache()
    def bias_and_splitting(self):
        print("BIAS AND SPLITTING")
        indexes = self.divide_et_impera()
        freshdata = []
        for i, d in enumerate(self.data):
            ind = indexes[i]
            delimeter_pres = int((ind[0] + ind[1]) / 2)
            delimeter_depth = int((ind[2] + ind[3]) / 2)
            dt1 = d.iloc[ind[0], 0] - d.iloc[ind[2], 3]
            dt2 = d.iloc[ind[1], 0] - d.iloc[ind[3], 3]
            datab1 = d.copy()
            datab2 = d.copy()
            target_name = d.columns[3]
            datab1[target_name] = datab1[target_name].apply(lambda x: x + dt1)
            datab2[target_name] = datab2[target_name].apply(lambda x: x + dt2)
            db1 = pd.concat([datab1.iloc[:delimeter_pres, 0],
                             datab1.iloc[:delimeter_pres, 1],
                             datab1.iloc[:delimeter_pres, 2],
                             datab1.iloc[:delimeter_depth, 3],
                             datab1.iloc[:delimeter_depth, 4]], axis=1)
            pre_db2_1 = pd.concat([datab2.iloc[delimeter_pres:, 0],
                                   datab2.iloc[delimeter_pres:, 1],
                                   datab2.iloc[delimeter_pres:, 2]], axis=1)
            pre_db2_1.reset_index(inplace=True, drop=True)
            pre_db2_2 = pd.concat([datab2.iloc[delimeter_depth:, 3],
                                   datab2.iloc[delimeter_depth:, 4]], axis=1)
            pre_db2_2.reset_index(inplace=True, drop=True)
            pre_db2_2.dropna(inplace=True, how='all')
            db2 = pd.concat([pre_db2_1, pre_db2_2], axis=1)
            double_data = [db1, db2]
            freshdata.append(double_data)
        return freshdata

    def support_dots(self, incles, data, intervals):
        print("SUPPORT DOTS")
        depths = []
        elongations = []
        pressures = []
        temperatures = []
        times = []
        n = 0
        for i, d in zip(incles, data):
            tempd = []
            tempe = []
            tempp = []
            tempt = []
            tempti = []
            for polka in d:
                temp_depths = []
                temp_elongations = []
                temp_pressures = []
                temp_temperatures = []
                temp_times = []
                max_depth = int(max(polka.iloc[:, 4]))
                min_depth = int(max(min(polka.iloc[:, 4]), 0))
                for depth in range(min_depth, max_depth):
                    if depth % intervals[n] == 0 and (max_depth - depth) > 30:
                        temp_depths.append(depth)  # глубины
                        temp_elongations.append(round(Other.search_and_interpolate(i, depth), 2))  # удлинения
                        search_array = polka.iloc[:, [4, 3]]  # давления
                        search_array.dropna(inplace=True, how='all')
                        temp_time = Other.search_and_interpolate(search_array, depth)
                        temp_times.append(temp_time)
                        search_array = polka.iloc[:, [0, 1]]
                        search_array.dropna(inplace=True, how='all')
                        temp_pressures.append(round(Other.search_and_interpolate(search_array, temp_time), 2))
                        search_array = polka.iloc[:, [0, 2]]  # температуры
                        search_array.dropna(inplace=True, how='all')
                        temp_temperatures.append(round(Other.search_and_interpolate(search_array, temp_time), 2))
                temp_depths.append(max_depth)  # глубины
                tempd.append(temp_depths)
                temp_elongations.append(round(Other.search_and_interpolate(i, max_depth), 2))  # удлинения
                tempe.append(temp_elongations)
                # давления
                search_array = polka.iloc[:, [4, 3]]
                search_array.dropna(inplace=True, how='all')
                temp_time = Other.search_and_interpolate(search_array, max_depth)
                search_array = polka.iloc[:, [0, 1]]
                search_array.dropna(inplace=True, how='all')
                temp_pressures.append(round(Other.search_and_interpolate(search_array, temp_time), 2))
                tempp.append(temp_pressures)
                # температуры
                search_array = polka.iloc[:, [0, 2]]
                search_array.dropna(inplace=True, how='all')
                temp_temperatures.append(round(Other.search_and_interpolate(search_array, temp_time), 2))
                tempt.append(temp_temperatures)
                temp_times.append(temp_time)
                tempti.append(temp_times)
            n += 1
            depths.append(tempd)
            elongations.append(tempe)
            pressures.append(tempp)
            temperatures.append(tempt)
            times.append(tempti)
        return depths, elongations, pressures, temperatures, times

    def zips(self):
        print("ZIPS")
        data = self.bias_and_splitting()
        incles = []
        for d in self.data:
            incl = d.iloc[:, [5, 6]]
            incl.dropna(inplace=True, how='all')
            incles.append(incl)
        intervals = [100 for d in data]
        depths, elongations, pressures, temperatures, times = self.support_dots(incles, data, intervals)
        calctables = []
        for d, e, p, t in zip(depths, elongations, pressures, temperatures):
            calctable = []
            for i in range(0, 2):
                kt = len(d[i])
                sample = []
                for j in range(0, kt):
                    if j == 0:
                        ro = None
                    else:
                        ro = round((p[i][j] - p[i][j - 1]) / (d[i][j] - e[i][j] - d[i][j - 1] + e[i][j - 1]) * 10, 3)
                    union = [d[i][j], e[i][j], p[i][j], t[i][j], ro]
                    sample.append(union)
                calctable.append(sample)
            calctables.append(calctable)
        for i, t in enumerate(calctables):
            for nomer_polki, polka in enumerate(t):
                a = polka[-1][4]
                b = polka[-2][4]
                c = polka[-3][4]
                mean_ro = (a + b + c) / 3
                cond1 = (max(a, mean_ro) - a) / max(a, mean_ro) > 0.15
                cond2 = (max(b, mean_ro) - b) / max(b, mean_ro) > 0.15
                cond3 = (max(c, mean_ro) - c) / max(c, mean_ro) > 0.15
                if cond1 or cond2 or cond3 or len(polka) < 7:
                    temp_incles = [incles[i], ]
                    temp_data = [data[i], ]

                    depths, elongations, pressures, temperatures, retiming = self.support_dots(temp_incles, temp_data,
                                                                                               intervals=[50, ])

                    for d, e, p, te, tim in zip(depths, elongations, pressures, temperatures, retiming):
                        kt = len(d[nomer_polki])
                        sample = []
                        time_sample = []
                        for j in range(0, kt):
                            union = [d[nomer_polki][j], e[nomer_polki][j], p[nomer_polki][j], te[nomer_polki][j]]
                            sample.append(union)
                            time_sample.append(tim[nomer_polki][j])
                        calctables[i][nomer_polki] = sample
                        times[i][nomer_polki] = time_sample
                for p in polka:
                    if len(p) == 5:
                        p.pop()
        return calctables, times


def Ppl_fabric(field, wname, data):
        return Ppl(field, wname, data, True)
