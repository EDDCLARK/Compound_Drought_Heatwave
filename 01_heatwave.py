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
        data = all_data[59:151, :, :]
    elif all_data.shape[0] == 366:
        data = all_data[60:152, :, :]

    # output
    freq = np.zeros(shape=(960, 1400))
    days = np.zeros(shape=(960, 1400))
    durations = np.zeros(shape=(960, 1400))
    severity = np.zeros(shape=(960, 1400))

    for i in range(960):
        for j in range(1400):
            event_days = 0
            event_freq = 0
            event_start = None
            event_end = None
            event_inter = 0
            event_durations = []
            event_severity = []

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
                            event_start = None
                            event_end = None
                        else:
                            event_start = None
                            event_end = None

            # 赋值
            freq[i, j] = event_freq
            days[i, j] = event_days
            durations[i, j] = np.nanmean(event_durations)
            severity[i, j] = np.nanmean(event_severity)
    return freq, days, durations, severity


if __name__ == "__main__":
    os.chdir(r"/data1/usersdir/zyb/05_graduate/data/methodUse/heatwave")

    # =========== heatwave identification =====================

    # -----------------------------------------------------------
    # ----------  计算daily max temperature ------------
    # -----------------------------------------------------------
    # for year in tqdm(range(1973, 2023)):
    #     data = combine_annual_nc_0d05(year)
    #     print(data.shape)
    #     np.save(r'daily_max_temperature_' + str(year) + '.npy', data)
    #     print(str(year) + " output done.")

    # -----------------------------------------------------------
    # ---------- 计算逐像元的每年 夏季（678月） 每日Tmax的 75和25分位数 90分位数 ---------------
    # -----------------------------------------------------------
    # npy_list = []
    # for year in tqdm(range(2012, 2023)):
    #     data = read_data('daily_max_temperature_' + str(year) + '.npy')
    #     if data.shape[0] == 365:
    #         summer_data = data[151:243, :, :]
    #     elif data.shape[0] == 366:
    #         summer_data = data[152:244, :, :]
    #     print(summer_data.shape)
    #     print(str(year) + " summer done.")
    #     npy_list.append(summer_data)
    #     print(len(npy_list))
    # integrated_data = np.concatenate(npy_list, axis=0)  # (days,481,701)
    # np.save('intedata_4.npy', integrated_data)

    # data = read_data('intedata_3.npy')
    # data2 = read_data('intedata_4.npy')
    # integrated_data = np.concatenate([data, data2], axis=0)
    # np.save('intedata_5.npy', integrated_data)
    # print(integrated_data.shape)

    # integrated_data = read_data('intedata_5.npy')
    # value_90th = np.nanpercentile(integrated_data, 90, axis=0)
    # value_25th = np.nanpercentile(integrated_data, 25, axis=0)
    # value_75th = np.nanpercentile(integrated_data, 75, axis=0)
    # print(value_90th.shape)
    # print(value_75th.shape)
    # print(value_25th.shape)
    # np.save('max_temperature_summer_90th.npy', value_90th)
    # np.save('max_temperature_summer_75th.npy', value_75th)
    # np.save('max_temperature_summer_25th.npy', value_25th)

    # tile combine to avoid killing processes
    # th90_1 = read_data('max_temperature_summer_90th_1.npy')
    # th90_2 = read_data('max_temperature_summer_90th_2.npy')
    # th75_1 = read_data('max_temperature_summer_75th_1.npy')
    # th75_2 = read_data('max_temperature_summer_75th_2.npy')
    # th25_1 = read_data('max_temperature_summer_25th_1.npy')
    # th25_2 = read_data('max_temperature_summer_25th_2.npy')

    # new_th90 = np.zeros(shape=(960, 1400)) * np.nan
    # new_th75 = np.zeros(shape=(960, 1400)) * np.nan
    # new_th25 = np.zeros(shape=(960, 1400)) * np.nan
    #
    # new_th90[:, :700] = th90_1[:, :]
    # new_th90[:, 700:] = th90_2[:, :]
    # new_th75[:, :700] = th75_1[:, :]
    # new_th75[:, 700:] = th75_2[:, :]
    # new_th25[:, :700] = th25_1[:, :]
    # new_th25[:, 700:] = th25_2[:, :]

    # -----------------------------------------------------------
    # ---------- 每年热浪次数、天数、 duration、severity-----------
    # -----------------------------------------------------------
    th90 = read_data("max_temperature_summer_90th.npy")[::-1, :]
    th75 = read_data("max_temperature_summer_75th.npy")[::-1, :]
    th25 = read_data("max_temperature_summer_25th.npy")[::-1, :]
    freq_list = []
    days_list = []
    duration_list = []
    severity_list = []
    for year in tqdm(range(1972, 2023)):
        data = read_data('daily_max_temperature_' + str(year) + '.npy')[:, ::-1, :]
        freq, days, durations, severity = get_heatwave_event(data, th90, th75, th25)
        freq_list.append(freq)
        days_list.append(days)
        duration_list.append(durations)
        severity_list.append(severity)
    integrated_freq = np.stack(freq_list, axis=0)  # (41,481,701)
    integrated_days = np.stack(days_list, axis=0)  # (41,481,701)
    integrated_durations = np.stack(duration_list, axis=0)  # (41,481,701)
    integrated_severity = np.stack(severity_list, axis=0)  # (41,481,701)

    season = 'haru'

    np.save('/data1/usersdir/zyb/heatwave/heatwave_freq_1972_2022' + season + '.npy', integrated_freq)
    np.save('/data1/usersdir/zyb/heatwave/heatwave_days_1972_2022' + season + '.npy', integrated_days)
    np.save('/data1/usersdir/zyb/heatwave/heatwave_mean_duration_1972_2022' + season + '.npy', integrated_durations)
    np.save('/data1/usersdir/zyb/heatwave/heatwave_mean_severity_1972_2022' + season + '.npy', integrated_severity)



    # python /data1/usersdir/zyb/05_graduate/01_heatwave.py