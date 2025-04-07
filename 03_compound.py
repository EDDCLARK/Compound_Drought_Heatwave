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





def heatwave_severity(data, th75, th25):
    hws = []
    for value in data:
        se = (value - th25) / (th75 - th25)
        hws.append(se)
    sum_hws = np.nansum(hws)
    return sum_hws


def compound_severity(data, th75, th25, twsdsi, event_start, year_days):
    hws = []
    day = event_start

    if year_days == 365:
        for value in data:
            if day < 30:
                se = ((value - th25) / (th75 - th25)) * (-1 * twsdsi[5])
            if day >= 30 and day < 61:
                se = ((value - th25) / (th75 - th25)) * (-1 * twsdsi[6])
            if day >= 61 and day < 92:
                se = ((value - th25) / (th75 - th25)) * (-1 * twsdsi[7])

            day += 1

            hws.append(se)
    elif year_days == 366:
        for value in data:
            if day < 30:
                se = ((value - th25) / (th75 - th25)) * (-1 * twsdsi[5])
            if day >= 30 and day < 61:
                se = ((value - th25) / (th75 - th25)) * (-1 * twsdsi[6])
            if day >= 61 and day < 92:
                se = ((value - th25) / (th75 - th25)) * (-1 * twsdsi[7])

            day += 1
            hws.append(se)

    sum_hws = np.nansum(hws)
    return sum_hws



# 6annual  678月 compound heatwave drought
# 678月内heatwave
def get_compound_event(all_data, th90, th75, th25, drought):

    if all_data.shape[0] == 365:
        data = all_data[151:243, :, :]
        for i in range(92):
            if i < 30:
                data[i, :, :][drought[5] > -1.5] = 0
            elif i >= 30 and i < 61:
                data[i, :, :][drought[6] > -1.5] = 0
            elif i >= 61 and i < 92:
                data[i, :, :][drought[7] > -1.5] = 0



    elif all_data.shape[0] == 366:
        data = all_data[152:244, :, :]
        for i in range(92):
            if i < 30:
                data[i, :, :][drought[5] > -1.5] = 0
            elif i >= 30 and i < 61:
                data[i, :, :][drought[6] > -1.5] = 0
            elif i >= 61 and i < 92:
                data[i, :, :][drought[7] > -1.5] = 0

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
                                se = compound_severity(data[event_start:event_end + 1, i, j], th75[i, j], th25[i, j], drought[:,i,j], event_start, all_data.shape[0])
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
                            se = compound_severity(data[event_start:k + 1, i, j], th75[i, j], th25[i, j], drought[:,i,j], event_start, all_data.shape[0])
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
                            se = compound_severity(data[event_start:event_end + 1, i, j], th75[i, j], th25[i, j], drought[:,i,j], event_start, all_data.shape[0])
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
    os.chdir(r"/data1/usersdir/zyb/05_graduate/data/methodUse/")

    twsdsi = read_data(r"drought/spei_1972_2022.npy")
    print(twsdsi.shape)
    monthly_twsdsi = twsdsi.reshape(51, 12, 960, 1400)

    th90 = read_data("heatwave/max_temperature_summer_90th.npy")[::-1, :]
    th75 = read_data("heatwave/max_temperature_summer_75th.npy")[::-1, :]
    th25 = read_data("heatwave/max_temperature_summer_25th.npy")[::-1, :]
    freq_list = []
    days_list = []
    duration_list = []
    severity_list = []
    index = 0
    for year in tqdm(range(1972, 2023)):
        data = read_data('heatwave/daily_max_temperature_' + str(year) + '.npy')[:, ::-1, :]
        freq, days, durations, severity = get_compound_event(data, th90, th75, th25, monthly_twsdsi[index])
        freq_list.append(freq)
        days_list.append(days)
        duration_list.append(durations)
        severity_list.append(severity)
        index += 1
    integrated_freq = np.stack(freq_list, axis=0)  # (41,481,701)
    integrated_days = np.stack(days_list, axis=0)  # (41,481,701)
    integrated_durations = np.stack(duration_list, axis=0)  # (41,481,701)
    integrated_severity = np.stack(severity_list, axis=0)  # (41,481,701)

    np.save('/data1/usersdir/zyb/05_graduate/data/result/compound_freq_1972_2022.npy', integrated_freq)
    np.save('/data1/usersdir/zyb/05_graduate/data/result/compound_days_1972_2022.npy', integrated_days)
    np.save('/data1/usersdir/zyb/05_graduate/data/result/compound_mean_duration_1972_2022.npy', integrated_durations)
    np.save('/data1/usersdir/zyb/05_graduate/data/result/compound_mean_severity_1972_2022.npy', integrated_severity)

    # python /data1/usersdir/zyb/05_graduate/03_compound.py