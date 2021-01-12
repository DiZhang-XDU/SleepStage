import argparse
import glob
import math
import ntpath
import os
import shutil
import urllib
# import urllib2

from datetime import datetime

import numpy as np
import pandas as pd

from mne import Epochs, pick_types, find_events
from mne.io import concatenate_raws, read_raw_edf

import dhedfreader

stage_dict = {
    "W": 0,
    "N1": 1,
    "N2": 2,
    "N3": 3,
    "REM": 4,
}
EPOCH_SEC_SIZE = 30

def Resample(input_signal,src_fs,tar_fs):
    '''
    :param input_signal:输入信号
    :param src_fs:输入信号采样率
    :param tar_fs:输出信号采样率
    :return:输出信号
    '''
    dtype = input_signal.dtype
    audio_len = len(input_signal)
    # audio_time_max = 1.0*(audio_len-1) / src_fs
    audio_time_max = 1.0*audio_len / src_fs
    src_time = 1.0 * np.linspace(0,audio_len,audio_len) / src_fs
    tar_time = 1.0 * np.linspace(0,np.int(audio_time_max*tar_fs),np.int(audio_time_max*tar_fs)) / tar_fs
    output_signal = np.interp(tar_time,src_time,input_signal).astype(dtype)

    return output_signal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="D:/SLEEP/shhs/polysomnography/edfs/shhs1/",
                        help="File path to the CSV or NPY file that contains walking data.")
    parser.add_argument("--anno_dir", type=str, default="D:/SLEEP/shhs/polysomnography/annotations-events-profusion/shhs1/",
                        help="x")
    parser.add_argument("--output_dir", type=str, default="../prepared_data/SHHS-NPZ/",
                        help="Directory where to save outputs.")
    args = parser.parse_args()

    # Output dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir)


    # Read raw and annotation EDF files
    psg_fnames = glob.glob(os.path.join(args.data_dir, "*.edf"))
    ann_fnames = glob.glob(os.path.join(args.anno_dir, "*.xml"))
    psg_fnames.sort()
    ann_fnames.sort()
    psg_fnames = np.asarray(psg_fnames)
    ann_fnames = np.asarray(ann_fnames)


    f_all = open(os.path.join(args.output_dir, 'all.txt'),'a')
    for i in len(psg_fnames): 

        folder_npz = os.path.join(args.output_dir, '{:04d}'.format(i))
        if os.path.exists(folder_npz) is False:
            os.mkdir(folder_npz)

        raw = read_raw_edf(psg_fnames[i], preload=True, stim_channel=None)
        
        # get raw Sample Freq and Channel Name
        sampling_rate = raw.info['sfreq']
        resample = False if sampling_rate == 125 else True
        print('【sampling rate】:',sampling_rate)

        # # for MASS
        # c3 = [c for c in raw.ch_names if 'EEG C3' in c][0]
        # c4 = [c for c in raw.ch_names if 'EEG C4' in c][0]
        # eogl = [c for c in raw.ch_names if 'EOG Left' in c][0]
        # eogr = [c for c in raw.ch_names if 'EOG Right' in c][0]
        # emg = [c for c in raw.ch_names if 'EMG Chin' in c][0]
        # channel_name = (c3, c4, eogl, eogr, emg)

        names_SHHS_C3 = ['EEG(sec)', 'EEG2', 'EEG 2', 'EEG(SEC)', 'EEG sec']
        c3 = [n for n in names_SHHS_C3 if n in raw.ch_names][0]
        c4 = 'EEG'
        eogl, eogr = 'EOG(L)','EOG(R)'
        emg = 'EMG'
        channel_name = (c3, c4, eogl, eogr, emg)

        raw_ch_tmp = []
        for cn in channel_name:
            raw_ch_df = raw.to_data_frame(scaling_time=100.0)[cn]
            raw_ch_df = raw_ch_df.to_frame()
            raw_ch_df.set_index(np.arange(len(raw_ch_df)))
            if resample:
                raw_ch_tmp.append(Resample(raw_ch_df.values[:].flatten(), sampling_rate, 125).reshape([-1,1]))
            else:
                raw_ch_tmp.append(raw_ch_df.values[:])
        raw_ch = np.concatenate(raw_ch_tmp, axis = 1) 

        # Get anno
        from anno_tool import xml
        ann = xml(ann_fnames[i])
        assert len(raw_ch) % (EPOCH_SEC_SIZE * 125) == 0

        n_epochs = len(raw_ch) // (EPOCH_SEC_SIZE * 125)
        raw_ch_slim = raw_ch[:int(n_epochs * EPOCH_SEC_SIZE * 125)]

        x = np.asarray(np.split(raw_ch_slim, n_epochs)).astype(np.float32)
        y = np.asarray(ann).astype(np.int32)

        assert len(x) == len(ann)

        y_onehot = np.zeros((y.shape[0], 5))
        for j in range(y.shape[0]):
            y_onehot[j][y[j]] = 1.

        print("Data shape: {}, {}".format(x.shape, y.shape))

        # Save
        for j in range(y.shape[0]):
            path_npz = os.path.join(folder_npz, '{:04d}.npz'.format(j))
            save_dict = {
                "X":x[j],
                "y":y[j],
                "y_onehot":y_onehot[j]
            }
            np.savez(path_npz, **save_dict)
            f_all.write(path_npz[1:] + ',' + str(y[j]) + '\n')
        with open(os.path.join(args.output_dir, 'refer.txt'), 'a') as f:
                f.write('{:04d},{}\n'.format(i, psg_fnames[i]))
        print ("\n================i={:3d} done==================\n".format(i))
    f_all.close()

if __name__ == "__main__":
    main()
