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


# 678月drought事件
def get_drought_events(data):
    monthly_data = data.reshape(51,12,960,1400)

    # output
    freq = np.zeros(shape=(51,960,1400))
    days = np.zeros(shape=(51,960,1400))
    durations = np.zeros(shape=(51,960,1400))
    severity = np.zeros(shape=(51,960,1400))

    for year in tqdm(range(51)):
        da = monthly_data[year]

        for i in range(960):
            for j in range(1400):
                event_days = 0
                event_freq = 0
                event_start = None
                event_end = None
                event_inter = 0
                event_durations = []
                event_severity = []

                for month in range(12):
                    if da[month, i, j] <= -1.5:  # 干旱月
                        event_days += 1  # 总天数+1
                        event_inter = 0  # 清空间隔
                        if event_start is None:
                            event_start = month  # 如果是第一次，则month为起始月
                        if event_end is not None:
                            event_end = None  # 保障机制
                    else:  # 非干旱日
                        event_inter += 1  # 间隔+1
                        if event_inter == 1:
                            event_end = month - 1  # k时第一次出现间隔，则将k-1确定为事件结束
                        elif event_inter >= 2:  # 此时符合事件结束条件
                            if (event_start is not None) and (event_end is not None):
                                event_freq += 1
                                du = event_end - event_start + 1
                                event_durations.append(du)
                                se = np.nansum(da[event_start:event_end + 1, i, j]) * -1
                                event_severity.append(se)
                                event_start = None
                                event_end = None

                    # 最后一月的处理
                    if month == 11:
                        if event_start is not None and event_end is None:
                            event_freq += 1
                            du = month - event_start + 1
                            event_durations.append(du)
                            se = np.nansum(da[event_start:month + 1, i, j]) * -1
                            event_severity.append(se)
                            event_start = None
                            event_end = None

                        elif event_start is not None and event_end is not None:
                            event_freq += 1
                            du = event_end - event_start + 1
                            event_durations.append(du)
                            se = np.nansum(da[event_start:event_end + 1, i, j]) * -1
                            event_severity.append(se)
                            event_start = None
                            event_end = None

                freq[year,i,j] = event_freq
                days[year,i, j] = event_days
                durations[year,i, j] = np.nanmean(event_durations)
                severity[year,i, j] = np.nanmean(event_severity)
    return freq, days, durations, severity










if __name__ == "__main__":
    os.chdir(r"/data1/usersdir/zyb/05_graduate/data/methodUse/drought")
    # data = readNc(r"spei/spei03.nc")
    #
    # target_lon = np.arange(70, 140, 0.05)
    # target_lat = np.arange(12, 60, 0.05)
    #
    # data = data.sel(time=slice('1972-01-01', '2022-12-31')).sel(lon=slice(70, 140)).sel(lat=slice(12, 60)).reindex(lon=target_lon, lat=target_lat, method='nearest')
    # np.save(r'spei_1972_2022.npy', data["spei"][:, ::-1, :])
    # print(data["spei"].shape)

    # -----------------------------------------------------------
    # ---------- 每年干旱 次数、天数、 duration、severity-----------
    # -----------------------------------------------------------
    twsdsi = read_data("spei_1972_2022.npy")
    freq, days, durations, severity = get_drought_events(twsdsi)
    np.save('/data1/usersdir/zyb/05_graduate/data/result/drought_freq_1972_2022.npy', freq)
    np.save('/data1/usersdir/zyb/05_graduate/data/result/drought_days_1972_2022.npy', days)
    np.save('/data1/usersdir/zyb/05_graduate/data/result/drought_mean_duration_1972_2022.npy', durations)
    np.save('/data1/usersdir/zyb/05_graduate/data/result/drought_mean_severity_1972_2022.npy', severity)

    # python /data1/usersdir/zyb/05_graduate/02_drought.py

