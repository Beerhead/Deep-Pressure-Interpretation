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
        self.well_name = wname
        self.data = data
        self.resid = Other.make_id()
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
        self.OTS_Field_ID = None
        self.OTS_Well_ID = None
        self.OTS_Mes_ID = None
        self.OTS_New_Mes_ID = None
        self.TP_id = None
        self.region = None
        self.tzeh_id = None
        self.avg_temp_gradient = None
        self.GOB = None
        self.OWB = None
        self.GWB = None
        self.static_level = None
        self.max_depth = None
        self.depth_pressure_temperature_data=None
        self.divider_points = None
        if not new:
            self.get_well_params_old()

    def make_depth_pressure_temperature_data(self):
        if self.final_data is None: return
        self.final_data.to_clipboard()
        S1 = self.final_data.iloc[:, 4]
        S1.dropna(inplace=True, how='all')
        S2, S3 = pd.Series(), pd.Series()
        DFPres = self.final_data.iloc[:, [0,1]]
        DFTemp = self.final_data.iloc[:, [0,2]]
        for i in range(S1.size):
            S2 = S2.append(pd.Series([Other.search_and_interpolate(DFPres, self.final_data.iloc[i, 3])]), ignore_index=True)
            S3 = S3.append(pd.Series([Other.search_and_interpolate(DFTemp, self.final_data.iloc[i, 3])]), ignore_index=True)
        self.depth_pressure_temperature_data = pd.concat([S1, S2, S3], axis = 1)
        self.depth_pressure_temperature_data.to_clipboard()

    def determine_static_level(self):
        if self.GOB:
            self.static_level = self.precise_calc_static_level(self.GOB)
            self.GOB = self.static_level
            return
        if self.GWB:
            self.static_level = self.precise_calc_static_level(self.GWB)
            self.GWB = self.static_level

    def precise_calc_static_level(self, reference_point):
        low_bound = int(max(1, reference_point-50))
        high_bound = int(min(self.max_depth, reference_point+50))
        tempDF=self.final_data.iloc[:, [4, 3]]
        tempDF1 = tempDF.dropna(how='all')
        tempDF = self.final_data.iloc[:, [0, 1]]
        tempDF2 = tempDF.dropna(how='all')
        prev_pres = 9000
        for i in range(low_bound, high_bound):
            time_depth = Other.search_and_interpolate_old(tempDF1, i)
            new_pres = Other.search_and_interpolate(tempDF2, time_depth)
            #print(i, time_depth, new_pres, prev_pres, (new_pres - prev_pres)*10)
            if (new_pres - prev_pres)*10 > 0.7:
                return i
            else:
                prev_pres = new_pres
        return reference_point



    def calc_avg_temp_gradient(self):
        if self.table_ind is None: return
        model = self.table_models[self.table_ind]
        dT = float(model.item(model.rowCount() - 1, 3).text()) - float(model.item(1, 3).text())
        dH = float(model.item(model.rowCount() - 1, 0).text()) - float(model.item(model.rowCount() - 1, 1).text()) - \
             float(model.item(1, 0).text()) + float(model.item(1, 1).text())
        self.avg_temp_gradient = 100 * dT / dH

    def calc_phase_borders(self):
        if self.table_models is None: return
        model = self.table_models[self.table_ind]
        types_list = [model.item(i, 5).text() for i in range(1, model.rowCount())]
        depth_list = [float(model.item(i, 0).text()) for i in range(1, model.rowCount())]
        self.GOB, self.OWB, self.GWB = calc_borders(types_list, depth_list)



    def get_well_params(self):
        def sql_bed(bed_id):
            with Other.sql_query() as connection:
                cursor = connection.cursor()
                cursor.execute('''SELECT ots_bn.sosbed.bedname from ots_bn.sosbed
                                WHERE ots_bn.sosbed.bedid = :bedid''',
                               bedid=bed_id)
                row = cursor.fetchone()
            return row[0]

        with Other.sql_query() as connection:
            cursor = connection.cursor()
            cursor.execute('''SELECT ots_bn.sosperforation.PRFPERFORATEDZONETOP,
                            ots_bn.sosperforation.PRFBEDID
                            from ots_bn.sosperforation
                            WHERE ots_bn.sosperforation.prfwellid = :wellid''',
                           wellid=self.OTS_Well_ID)
            rows = cursor.fetchall()
            rows.sort()
            self.vdp = rows[0][0]
            layer = [sql_bed(i[1]) for i in rows]
            self.layer = set(layer)
            cursor.execute('''SELECT ots_bn.soswell.WELANGULARITYTESTDEPTH,
                            ots_bn.soswell.WELANGULARITYTESTELONGATION,
                            ots_bn.soswell.welorganizationid,
                            ots_bn.soswell.welfieldid
                            from ots_bn.soswell
                            WHERE ots_bn.soswell.welid = :wellid''',
                           wellid=self.OTS_Well_ID)
            rows = cursor.fetchone()
            self.tzeh_id = rows[2]
            self.OTS_Field_ID = rows[3]
            try:
                bytes_len = int.from_bytes(rows[0].read()[3:7], byteorder=byteorder)
                depth = pd.Series(
                    np.frombuffer(pylzma.decompress(rows[0].read()[7:], maxlength=bytes_len), dtype=np.dtype('f')))
            except:
                depth = pd.Series(np.frombuffer(rows[0].read()[3:], dtype=np.dtype('f')))
            try:
                bytes_len = int.from_bytes(rows[1].read()[3:7], byteorder=byteorder)
                elong = pd.Series(
                    np.frombuffer(pylzma.decompress(rows[1].read()[7:], maxlength=bytes_len), dtype=np.dtype('f')))
            except:
                elong = pd.Series(np.frombuffer(rows[1].read()[3:], dtype=np.dtype('f')))
            self.incl = pd.concat([depth, elong], axis=1)
            self.vdp_elong = round(Other.search_and_interpolate(self.incl, self.vdp), 2)
            temp_pressure_time_series = self.data.iloc[:, 0]
            temp_pressure_time_series.dropna(inplace=True)
            self.research_date = str(temp_pressure_time_series[len(temp_pressure_time_series) // 2])[:10]
            self.first_measure_datetime = temp_pressure_time_series.iloc[0]
            self.last_measure_datetime = temp_pressure_time_series.iloc[-1]

            # cursor.execute('''SELECT ots_bn.sosorganizationrelation.orrparentid
            #                 from ots_bn.sosorganizationrelation
            #                 WHERE ots_bn.sosorganizationrelation.orrchildid = :chid''',
            #                chid=self.tzeh_id)
            # rows = cursor.fetchone()
            # self.region_id = rows[0]
            # cursor.execute('''SELECT ots_bn.sosorganization.orgname
            #                 from ots_bn.sosorganization
            #                 WHERE ots_bn.sosorganization.orgid = :regid''',
            #                regid=self.region_id)
            # rows = cursor.fetchone()
            # self.region = rows[0]
            self.TP_id = dict_suborg[self.tzeh_id]
            self.max_depth = self.data.iloc[:, 4].max()
            print(self.max_depth)

    def get_well_params_old(self):
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

    def __init__(self, vhoddata, incl=None, alt=False, delta=100):
        self.data = vhoddata
        self.alt = alt
        self.delta = delta
        if incl:
            for d, i, index in zip(self.data, incl, range(len(self.data))):
                self.data[index] = pd.concat([d, i], axis=1)

    def transform_data(self):
        self.fed_data = []
        for d in self.data:
            kt_pres = d.iloc[:, 0].count()
            kt_dep = d.iloc[:, 4].count()
            print('KT = ', kt_pres, " KD = ", kt_dep)
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
        return finalindexes

    @lru_cache()
    def bias_and_splitting(self):
        indexes = self.divide_et_impera()
        print(indexes)
        freshdata = []
        central_dots=[]
        for i, d in enumerate(self.data):
            ind = indexes[i]
            central_dots.append([(ind[0] + ind[1]) / 2, (ind[2] + ind[3]) / 2])
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
        return freshdata, central_dots

    def support_dots(self, incles, data, intervals):
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
        data,_ = self.bias_and_splitting()
        incles = []
        for d in self.data:
            incl = d.iloc[:, [5, 6]]
            incl.dropna(inplace=True, how='all')
            incles.append(incl)
        intervals = [self.delta for d in data]
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
                                                                                               intervals=[
                                                                                                   self.delta / 2, ])

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

def calc_borders(type_list, depth_shelfs):
    def disintegrate_abortions(type_list):
        new_list = list(type_list)
        length = len(new_list)
        if length < 5: return
        for j in range(2, length - 3):
            left = [new_list[j - 2], new_list[j - 1]]
            right = [new_list[j + 1], new_list[j + 2]]
            if (new_list[j] not in left) and (new_list[j] not in right):
                if len(set(left).intersection(set(right))) == 0:
                    continue
                counter = Counter(left + right)
                new_list[j] = counter.most_common(1)[0][0]

        return new_list

    GOB = None
    OWB = None
    GWB = None
    type_list = disintegrate_abortions(type_list)
    unique_set = set(type_list)
    sorted_list_of_phases = sorted(unique_set)
    #print(' len   =    ', len(unique_set))
    if len(unique_set) == 2:
        index = type_list.index(sorted_list_of_phases[-1])
        if sorted_list_of_phases == ['Gas', 'Oil']:
            GOB = (depth_shelfs[index] + depth_shelfs[index - 1]) / 2
        elif sorted_list_of_phases == ['Oil', 'Water']:
            OWB = (depth_shelfs[index] + depth_shelfs[index - 1]) / 2
        elif sorted_list_of_phases == ['Gas', 'Water']:
            GWB = (depth_shelfs[index] + depth_shelfs[index - 1]) / 2
    elif len(unique_set) == 3:
        sequence = []
        for i in type_list:
            if i not in sequence:
                sequence.append(i)
        if sequence == ['Gas', 'Oil', 'Water']:
            index1 = type_list.index(sorted_list_of_phases[1])
            index2 = type_list.index(sorted_list_of_phases[2])
            GOB = (depth_shelfs[index1] + depth_shelfs[index1 - 1]) / 2
            OWB = (depth_shelfs[index2] + depth_shelfs[index2 - 1]) / 2
    #print(type_list)
    return GOB, OWB, GWB