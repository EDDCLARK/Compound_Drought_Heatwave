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




if __name__ == "__main__":

    # # =============  combine 16 day data ================================
    # for year in tqdm(range(2000, 2023)):
    #     os.chdir("/data2/CSIF_2000_2022/" + str(year))
    #     path_list = os.listdir("/data2/CSIF_2000_2022/" + str(year))
    #     path_list.sort()
    #
    #     data = xr.open_mfdataset(path_list, concat_dim="time", combine='nested')
    #     reshaped = data['clear_daily_SIF'].values.reshape(23, 4, 3600, 7200)
    #     result = np.nanmean(reshaped, axis=1)
    #     print(result.shape)
    #     os.chdir("/data1/usersdir/zyb/05_graduate/data/methodUse/16-day-CSIF")
    #     np.save("global_csif_16day_" + str(year), result[:, ::-1, :])

    # =============  combine monthly data ================================
    for year in tqdm(range(2000, 2023)):
        os.chdir("/data2/CSIF_2000_2022/" + str(year))
        path_list = os.listdir("/data2/CSIF_2000_2022/" + str(year))
        path_list.sort()

        data = xr.open_mfdataset(path_list, concat_dim="time", combine='nested')
        reshaped = data['clear_daily_SIF'].values #.reshape(23, 4, 3600, 7200)

        result = np.zeros(shape=(12, 3600, 7200))

        result[0][:, :] = np.nanmean(reshaped[0:8, :, :], axis=0)[:,:]
        result[1][:, :] = np.nanmean(reshaped[8:15, :, :], axis=0)[:, :]
        result[2][:, :] = np.nanmean(reshaped[15:23, :, :], axis=0)[:, :]
        result[3][:, :] = np.nanmean(reshaped[23:30, :, :], axis=0)[:, :]
        result[4][:, :] = np.nanmean(reshaped[30:38, :, :], axis=0)[:, :]
        result[5][:, :] = np.nanmean(reshaped[38:46, :, :], axis=0)[:, :]
        result[6][:, :] = np.nanmean(reshaped[46:53, :, :], axis=0)[:, :]
        result[7][:, :] = np.nanmean(reshaped[53:61, :, :], axis=0)[:, :]
        result[8][:, :] = np.nanmean(reshaped[61:69, :, :], axis=0)[:, :]
        result[9][:, :] = np.nanmean(reshaped[69:76, :, :], axis=0)[:, :]
        result[10][:, :] = np.nanmean(reshaped[76:84, :, :], axis=0)[:, :]
        result[11][:, :] = np.nanmean(reshaped[84:, :, :], axis=0)[:, :]
        print(result.shape)



        os.chdir("/data1/usersdir/zyb/05_graduate/data/methodUse/monthly-CSIF")
        np.save("global_csif_16day_" + str(year), result[:, ::-1, :])



    # python /data1/usersdir/zyb/05_graduate/04_csif.py