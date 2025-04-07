import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmaps
import pandas as pd
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
import xarray as xr

from tqdm.autonotebook import tqdm


# nc
def readNc(input_path):
    cn = xr.open_dataset(input_path)
    return cn


def read_data(path):
    data = np.load(path)
    print(path, 'read data')
    return data


# 合并一年的nc
# 计算annual max temperature
def combine_annual_nc(year):
    nc_list = []
    for month in ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]:
        data = readNc(str(year) + "_" + month + "\data.nc")
        re = np.zeros(shape=(data["t2m"].shape[0], 96, 140)) * np.nan

        re[:, :, :] = data["t2m"][:, :, :]
        day = data["t2m"].shape[0] / 24
        re_reshaped = re.reshape(int(day), 24, 96, 140)
        re_max = np.nanmax(re_reshaped, axis=1)
        nc_list.append(re_max)
        print("month:" + month + " done.")
    integrated_data = np.concatenate(nc_list, axis=0)  # (days,481,701)
    print("year: " + str(year) + " done.")
    return integrated_data


# 计算annual max temperature
def combine_annual_nc_0d05(year):
    nc_list = []
    target_lon = np.arange(70, 140, 0.05)
    target_lat = np.arange(12, 60, 0.05)


    for month in ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]:
        data = readNc(str(year) + "_" + month + "/data.nc")
        data = data.reindex(longitude=target_lon, latitude=target_lat, method='nearest')

        re = np.zeros(shape=(data["t2m"].shape[0], 960, 1400)) * np.nan
        re[:, :, :] = data["t2m"][:, :, :]
        day = data["t2m"].shape[0] / 24
        re_reshaped = re.reshape(int(day), 24, 960, 1400)
        re_max = np.nanmax(re_reshaped, axis=1)
        nc_list.append(re_max)
        print("month:" + month + " done.")
    integrated_data = np.concatenate(nc_list, axis=0)  # (days,481,701)
    print("year: " + str(year) + " done.")
    return integrated_data


def heatwave_severity(data, th75, th25):
    hws = []
    for value in data:
        se = (value - th25) / (th75 - th25)
        hws.append(se)
    mean_hws = np.nansum(hws)
    return mean_hws


# 678月内heatwave
def get_heatwave_event(all_data, th90, th75, th25):
    if all_data.shape[0] == 365:
        data = all_data[151:243, :, :]
    elif all_data.shape[0] == 366:
        data = all_data[152:244, :, :]

    # output
    freq = np.zeros(shape=(960, 1400))
    days = np.zeros(shape=(960, 1400))
    durations = np.zeros(shape=(960, 1400))
    severity = np.zeros(shape=(960, 1400))
    freq_monthly = np.zeros(shape=(3, 960, 1400))


    for i in range(960):
        for j in range(1400):
            event_days = 0
            event_freq = 0
            event_start = None
            event_end = None
            event_inter = 0
            event_durations = []
            event_severity = []
            event_starts = []
            event_ends = []

            for k in range(data.shape[0]):
                if data[k, i, j] > th90[i, j]:  # 热浪日
                    event_days += 1  # 总天数+1
                    event_inter = 0  # 清空间隔
                    if event_start is None:
                        event_start = k  # 如果是第一次，则k为起始日
                    if event_end is not None:
                        event_end = None  # 保障机制
                else:  # 非热浪日
                    event_inter += 1  # 间隔+1
                    if event_inter == 1:
                        event_end = k - 1  # k时第一次出现间隔，则将k-1确定为事件结束
                    elif event_inter >= 2:  # 此时符合事件结束条件
                        # 判断是否符合热浪事件条件
                        if (event_start is not None) and (event_end is not None):
                            if event_end - event_start >= 2:  # 持续时间超过3天
                                event_freq += 1
                                du = event_end - event_start + 1
                                event_durations.append(du)
                                se = heatwave_severity(data[event_start:event_end + 1, i, j], th75[i, j], th25[i, j])
                                event_severity.append(se)
                                event_starts.append(event_start)
                                event_ends.append(event_end)
                                event_start = None
                                event_end = None
                            else:  # 持续时间不超过3天
                                event_start = None
                                event_end = None

                # 最后一日的处理
                if k == data.shape[0] - 1:
                    if event_start is not None and event_end is None:
                        if k - event_start >= 2:  # 符合条件
                            event_freq += 1
                            du = k - event_start + 1
                            event_durations.append(du)
                            se = heatwave_severity(data[event_start:k + 1, i, j], th75[i, j], th25[i, j])
                            event_severity.append(se)
                            event_starts.append(event_start)
                            event_ends.append(k)
                            event_start = None
                            event_end = None
                        else:
                            event_start = None
                            event_end = None
                    elif event_start is not None and event_end is not None:
                        if event_end - event_start >= 2:  # 符合条件
                            event_freq += 1
                            du = event_end - event_start + 1
                            event_durations.append(du)
                            se = heatwave_severity(data[event_start:event_end + 1, i, j], th75[i, j], th25[i, j])
                            event_severity.append(se)
                            event_starts.append(event_start)
                            event_ends.append(event_end)
                            event_start = None
                            event_end = None
                        else:
                            event_start = None
                            event_end = None

            # 赋值
            # freq[i, j] = event_freq
            # days[i, j] = event_days
            # durations[i, j] = np.nanmean(event_durations)
            # severity[i, j] = np.nanmean(event_severity)
            # heatwave freq in each month
            freq_6 = 0
            freq_7 = 0
            freq_8 = 0
            for m in range(len(event_starts)):
                if event_starts[m] < 29 and event_ends[m] <= 29:
                    freq_6 += 1
                elif 60 > event_starts[m] >= 30 and event_ends[m] <= 60:
                    freq_7 += 1
                elif 91 > event_starts[m] >= 61 and event_ends[m] <= 91:
                    freq_8 += 1
                elif event_starts[m] < 29 and 60 >= event_ends[m] > 30:
                    freq_6 += 1
                    freq_7 += 1
                elif 60 > event_starts[m] >= 30 and 91 >= event_ends[m] > 61:
                    freq_7 += 1
                    freq_8 += 1
                elif event_starts[m] < 29 and 91 >= event_ends[m] > 61:
                    freq_6 += 1
                    freq_7 += 1
                    freq_8 += 1

            freq_monthly[0, i, j] = freq_6
            freq_monthly[1, i, j] = freq_7
            freq_monthly[2, i, j] = freq_8





    return freq_monthly


if __name__ == "__main__":
    os.chdir(r"/data1/usersdir/zyb/05_graduate/data/methodUse/heatwave")

    # # =========== heatwave identification =====================
    #
    # # -----------------------------------------------------------
    # # ---------- 每年热浪次数、天数、 duration、severity-----------
    # # -----------------------------------------------------------
    # th90 = read_data("max_temperature_summer_90th.npy")[::-1, :]
    # th75 = read_data("max_temperature_summer_75th.npy")[::-1, :]
    # th25 = read_data("max_temperature_summer_25th.npy")[::-1, :]
    # freq_list = []
    # days_list = []
    # duration_list = []
    # severity_list = []
    #
    # os.chdir(r"/data1/usersdir/zyb/05_graduate/data/methodUse/drought")
    # twsdsi = read_data("spei_1972_2022.npy")
    # monthly_drought = twsdsi.reshape(51, 12, 960, 1400)
    #
    # event_list = []
    #
    # os.chdir(r"/data1/usersdir/zyb/05_graduate/data/methodUse/heatwave")
    # index = 28
    # for year in tqdm(range(2000, 2023)):
    #     event_monthly = np.zeros(shape=(3, 960, 1400))
    #
    #     data = read_data('daily_max_temperature_' + str(year) + '.npy')[:, ::-1, :]
    #     heat_freq_monthly = get_heatwave_event(data, th90, th75, th25)    # (3, 960, 1400)
    #     heat_freq_monthly[heat_freq_monthly > 0] = 1
    #     drought_monthly = monthly_drought[index, 5:8, :, :]          # (3, 960, 1400)
    #     drought_monthly[drought_monthly > -1.5] = 0
    #     drought_monthly[drought_monthly <= -1.5] = 1
    #
    #     index += 1
    #     for i in range(3):
    #         index1 = np.where((heat_freq_monthly[i] == 1) & (drought_monthly[i] == 0))
    #         index2 = np.where((heat_freq_monthly[i] == 0) & (drought_monthly[i] == 1))
    #         index3 = np.where((heat_freq_monthly[i] == 0) & (drought_monthly[i] == 0))
    #         index4 = np.where((heat_freq_monthly[i] == 1) & (drought_monthly[i] == 1))
    #         event_monthly[i][index1] = 1  # heatwave
    #         event_monthly[i][index2] = 2  # drought
    #         event_monthly[i][index3] = 3  # normal
    #         event_monthly[i][index4] = 4  # compound
    #     event_list.append(event_monthly)
    #
    # integrated = np.stack(event_list, axis=0)  # (23, 3,960, 1400)
    # np.save('/data1/usersdir/zyb/05_graduate/data/result/event_monthly_2000_2022.npy', integrated)


    # ============================================================
    # ========== VIs detrend =====================================
    # # === evi ====
    # os.chdir(r"/data1/usersdir/zyb/05_graduate/data/methodUse/monthly-EVI")
    # path_list = os.listdir(r"/data1/usersdir/zyb/05_graduate/data/methodUse/monthly-EVI")
    # path_list.sort()
    # evi_list = []
    # evi_200001 = np.zeros(shape=(960, 1400)) * np.nan
    # evi_list.append(evi_200001)
    #
    # evi_list2 = []
    # index = 0
    # for path in tqdm(path_list):
    #     if index < 138:
    #         evi = read_data(path)
    #         evi = evi[600:1560, 5000:6400]
    #         evi_list.append(evi)
    #     else:
    #         evi = read_data(path)
    #         evi = evi[600:1560, 5000:6400]
    #         evi_list2.append(evi)
    #
    #     index += 1
    #
    # integrated1 = np.stack(evi_list, axis=0)   # (23*12, 960, 1400)
    # integrated2 = np.stack(evi_list2, axis=0)  # (23*12, 960, 1400)
    # new_list = [integrated1, integrated2]
    # integrated = np.concatenate(new_list, axis=0)
    # np.save('/data1/usersdir/zyb/05_graduate/data/methodUse/evi_preprocess.npy', integrated)
    # integrated = read_data("/data1/usersdir/zyb/05_graduate/data/methodUse/evi_preprocess.npy")
    # # mean = np.nanmean(integrated, axis=0)   # (960, 1400)
    # # detrend_evi = integrated - mean
    #
    # # deseasonal
    # deseason = integrated.reshape(23, 12, 960, 1400)  # (23, 12, 960, 1400)
    #
    # deseason = deseason[3:11, :, :, :]   # 03-10(3:11), 04-09(4:10), 04-10(4:11)
    # season_mean = np.nanmean(deseason, axis=0)  # (12,960,1400)
    # deseason = deseason[:] - season_mean
    # # detrend
    # detrend = deseason.reshape(8 * 12, 960, 1400)     # 2000-2022 (23*12)
    #
    # time = np.zeros(shape=(8 * 12, 960, 1400)) * np.nan
    # date = 1
    # for index in range(96):   # 2000-2022 (276)
    #     time[index] = np.ones(shape=(960, 1400)) * date
    #     date += 1
    #
    # slope = (96 * np.nansum(detrend * time, axis=0) - (np.nansum(detrend, axis=0) * np.nansum(time, axis=0))) / (
    #             96 * np.nansum(time * time, axis=0) - (np.nansum(time, axis=0) * np.nansum(time, axis=0)))
    # inter = np.nanmean(detrend, axis=0) - slope * np.nanmean(time, axis=0)
    # print(slope.shape)
    # print(inter.shape)
    #
    # estimate = slope * time + inter
    # detrend = detrend - estimate
    #
    # print(detrend.shape)



    # # === ndvi ====
    # os.chdir(r"/data1/usersdir/zyb/05_graduate/data/methodUse/monthly-NDVI")
    # path_list = os.listdir(r"/data1/usersdir/zyb/05_graduate/data/methodUse/monthly-NDVI")
    # path_list.sort()
    # evi_list = []
    # evi_200001 = np.zeros(shape=(960, 1400)) * np.nan
    # evi_list.append(evi_200001)
    #
    # evi_list2 = []
    # index = 0
    # for path in tqdm(path_list):
    #     if index < 138:
    #         evi = read_data(path)
    #         evi = evi[600:1560, 5000:6400]
    #         evi_list.append(evi)
    #     else:
    #         evi = read_data(path)
    #         evi = evi[600:1560, 5000:6400]
    #         evi_list2.append(evi)
    #
    #     index += 1
    #
    # integrated = np.stack(evi_list, axis=0)  # (23*12, 960, 1400)
    # integrated2 = np.stack(evi_list2, axis=0)  # (23*12, 960, 1400)
    # integrated = [integrated, integrated2]
    # integrated = np.concatenate(integrated, axis=0)
    # np.save('/data1/usersdir/zyb/05_graduate/data/methodUse/ndvi_preprocess.npy', integrated)
    # integrated = read_data("/data1/usersdir/zyb/05_graduate/data/methodUse/ndvi_preprocess.npy")
    #
    #
    # # deseasonal
    # deseason = integrated.reshape(23, 12, 960, 1400)  # (23, 12, 960, 1400)
    # season_mean = np.nanmean(deseason, axis=0)  # (12,960,1400)
    # deseason = deseason[:] - season_mean
    # # detrend
    # detrend = deseason.reshape(23 * 12, 960, 1400)
    #
    # time = np.zeros(shape=(23 * 12, 960, 1400)) * np.nan
    # date = 1
    # for index in range(276):
    #     time[index] = np.ones(shape=(960, 1400)) * date
    #     date += 1
    #
    # slope = (276 * np.nansum(detrend * time, axis=0) - (np.nansum(detrend, axis=0) * np.nansum(time, axis=0))) / (
    #         276 * np.nansum(time * time, axis=0) - (np.nansum(time, axis=0) * np.nansum(time, axis=0)))
    # inter = np.nanmean(detrend, axis=0) - slope * np.nanmean(time, axis=0)
    # print(slope.shape)
    # print(inter.shape)
    #
    # estimate = slope * time + inter
    # detrend = detrend - estimate
    #
    # print(detrend.shape)

    
    # # === sif ====
    # os.chdir(r"/data1/usersdir/zyb/05_graduate/data/methodUse/monthly-CSIF/china_region")
    # path_list = os.listdir(r"/data1/usersdir/zyb/05_graduate/data/methodUse/monthly-CSIF/china_region")
    # path_list.sort()
    # sif_list = []
    # for path in tqdm(path_list):
    #     data = read_data(path)
    #     # sif = data[:, 600:1560, 5000:6400]
    #     sif_list.append(data)
    # integrated = np.concatenate(sif_list, axis=0)  # (23*12, 960, 1400)
    # # deseasonal
    # deseason = integrated.reshape(23, 12, 960, 1400)  # (23, 12, 960, 1400)
    # deseason = deseason[4:10, :, :, :]  # 03-10(3:11), 04-09(4:10), 04-10(4:11)
    # season_mean = np.nanmean(deseason, axis=0)    # (12,960,1400)
    # deseason = deseason[:] - season_mean
    # # detrend
    # detrend = deseason.reshape(6*12, 960, 1400)  # 2000-2022 (23*12)
    #
    # time = np.zeros(shape=(6*12, 960, 1400)) * np.nan
    # date = 1
    # for index in range(72):
    #     time[index] = np.ones(shape=(960, 1400)) * date
    #     date += 1
    #
    # slope = (72 * np.nansum(detrend * time, axis=0) - (np.nansum(detrend,axis=0) * np.nansum(time,axis=0))) / (72 * np.nansum(time * time, axis=0) - (np.nansum(time,axis=0) * np.nansum(time,axis=0)))
    # inter = np.nanmean(detrend, axis=0) - slope * np.nanmean(time, axis=0)
    # print(slope.shape)
    # print(inter.shape)
    #
    # estimate = slope * time + inter
    # detrend = detrend - estimate
    #
    #
    # print(detrend.shape)

    #
    # np.save('/data1/usersdir/zyb/05_graduate/data/result/sif_monthly_2004_2009_deseason_detrend.npy', detrend)
    # np.save('/data1/usersdir/zyb/05_graduate/data/result/evi_monthly_2003_2010_deseason_detrend.npy', detrend)
    # np.save('/data1/usersdir/zyb/05_graduate/data/result/ndvi_monthly_2000_2022_deseason_detrend.npy', detrend)



    #'''
    # # ===================== extreme impacts ==================================
    os.chdir(r"/data1/usersdir/zyb/05_graduate/data/result")

    event = read_data("event_monthly_2000_2022.npy")
    # ndvi = read_data("ndvi_monthly_2000_2022_deseason_detrend.npy")
    # evi = read_data("evi_monthly_2003_2010_deseason_detrend.npy")
    sif = read_data("sif_monthly_2004_2009_deseason_detrend.npy")

    # ndvi = ndvi.reshape(23, 12, 960, 1400)
    # evi = evi.reshape(8, 12, 960, 1400)
    sif = sif.reshape(6, 12, 960, 1400)

    event_reshape = event.reshape(69, 960, 1400)
    event_reshape = event_reshape[12:30, :, :]     # 03-10(9:33), 04-09(12:30), 04-10(12:33)

    # ndvi_reshape = ndvi[:, 5:8, :, :].reshape(69, 960, 1400)
    # evi_reshape = evi[:, 5:8, :, :].reshape(24, 960, 1400)
    sif_reshape = sif[:, 5:8, :, :].reshape(18, 960, 1400)


    print(event_reshape.shape)
    # print(ndvi_reshape.shape)
    # print(evi_reshape.shape)
    print(sif_reshape.shape)

    # 2000-2022 (69, 960, 1400)
    normal_value = np.zeros(shape=(18, 960, 1400)) * np.nan
    heatwave_value = np.zeros(shape=(18, 960, 1400)) * np.nan
    drought_value = np.zeros(shape=(18, 960, 1400)) * np.nan
    compound_value = np.zeros(shape=(18, 960, 1400)) * np.nan

    # normal_value = np.zeros(shape=(69, 3, 960, 1400)) * np.nan
    # heatwave_value = np.zeros(shape=(69, 3, 960, 1400)) * np.nan
    # drought_value = np.zeros(shape=(69, 3, 960, 1400)) * np.nan
    # compound_value = np.zeros(shape=(69, 3, 960, 1400)) * np.nan

    print(event_reshape.shape[0])
    for i in tqdm(range(event_reshape.shape[0])):
        # 1   heatwave
        # 2   drought
        # 3   normal
        # 4   compound
        event = event_reshape[i]

        # # ndvi
        # heatwave_value[i, :, :] = ndvi_reshape[i, :, :]
        # heatwave_value[i][event != 1] = np.nan
        # drought_value[i, :, :] = ndvi_reshape[i, :, :]
        # drought_value[i][event != 2] = np.nan
        # normal_value[i, :, :] = ndvi_reshape[i, :, :]
        # normal_value[i][event != 3] = np.nan
        # compound_value[i, :, :] = ndvi_reshape[i, :, :]
        # compound_value[i][event != 4] = np.nan
        # # evi
        # heatwave_value[i, :, :] = evi_reshape[i, :, :]
        # heatwave_value[i][event != 1] = np.nan
        # drought_value[i, :, :] = evi_reshape[i, :, :]
        # drought_value[i][event != 2] = np.nan
        # normal_value[i, :, :] = evi_reshape[i, :, :]
        # normal_value[i][event != 3] = np.nan
        # compound_value[i, :, :] = evi_reshape[i, :, :]
        # compound_value[i][event != 4] = np.nan
        # sif
        heatwave_value[i, :, :] = sif_reshape[i, :, :]
        heatwave_value[i][event != 1] = np.nan
        drought_value[i, :, :] = sif_reshape[i, :, :]
        drought_value[i][event != 2] = np.nan
        normal_value[i, :, :] = sif_reshape[i, :, :]
        normal_value[i][event != 3] = np.nan
        compound_value[i, :, :] = sif_reshape[i, :, :]
        compound_value[i][event != 4] = np.nan

    print(normal_value.shape)
    print(heatwave_value.shape)
    print(drought_value.shape)
    print(compound_value.shape)

    np.save('/data1/usersdir/zyb/05_graduate/data/result/heatwave_impacts_monthly_2004_2009_sif.npy', heatwave_value)
    np.save('/data1/usersdir/zyb/05_graduate/data/result/drought_impacts_monthly_2004_2009_sif.npy', drought_value)
    np.save('/data1/usersdir/zyb/05_graduate/data/result/normal_impacts_monthly_2004_2009_sif.npy', normal_value)
    np.save('/data1/usersdir/zyb/05_graduate/data/result/compound_impacts_monthly_2004_2009_sif.npy', compound_value)

    #'''


    # python /data1/usersdir/zyb/05_graduate/05_vegetation.py