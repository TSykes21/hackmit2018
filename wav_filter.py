import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
import numpy as np
from numpy.fft import fft, fftfreq
import os
import pandas as pd

rootdir = './data'





for subdir, dirs, files in os.walk(rootdir):
    count = 0
    for file in files:
        if '.wav' not in file: continue
        rate, data = wav.read(os.path.join(subdir, file))
        try:
            if data.shape[1] > 1:
                data = data.T[0]
        except IndexError:
            pass

        i = 0
        while data[i] == 0:
            i += 1
        data = data[i:i+rate]


        if data.shape[0] < rate:
            zeroes = np.zeros(rate - data.shape[0])
            data = np.concatenate((data,zeroes), axis=0)

        data_len = data.shape[0]

        reduced = []
        ct = 0
        len = 0
        for pt in data:
            if ct % 75 == 0:
                reduced.append(pt)
                len += 1
            ct += 1

        data = np.asarray(reduced)
        data_len = len

        # plot initial sound wave:
        # x_axis = np.arange(len)
        # plt.plot(x_axis, data)
        # plt.show()

        # create spectrograph
        spectrograph = np.zeros((20,14))
        time_frame = int(data_len/20)
        column = 0
        for i in range(time_frame, data_len + 1, time_frame):
            freqs = fftfreq(time_frame) # cycles/second, if data_len is in seconds

            fft_out = fft(data[i - time_frame:i])
            true_fft = 2.0 * np.abs(fft_out)/data_len

            mask = freqs > 0
            freqs = freqs[mask]
            true_fft = true_fft[mask]

            # fill row of graph
            for j in range(0, true_fft.size):
                spectrograph[int(i/time_frame)-1,j] = true_fft[j]
                column += 1
            column = 0
        #print(spectrograph)
        #plt.imshow(spectrograph, cmap=plt.cm.gray)
        #plt.show()



        #plt.plot(freqs, true_fft)
        #plt.show()










