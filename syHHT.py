#!/usr/bin/env python
# coding: utf-8

"""
Hilbert Huang Transform

class HHT

HHT = syHHT.HHT(sample_freq, interval):
HHT.sig2imfs(sig, max_imf = None)
HHT.plot_imfs(sig, imfs)
HHT.hht2image(self, inst_freq, inst_amp, freqsol = 50, timesol = 50)
HHT.sig2hht(self, sig, max_imf = None, freqsol = 50, timesol = 50, mode = '')
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import math
from PyEMD import EEMD
# from PyEMD import CEEMDAN
eemd = EEMD()

class HHT:
    def __init__(self, sample_freq, interval):
        self.sf = sample_freq
        self.interval = interval
        self.time = np.linspace(0, interval/sample_freq, interval)
        self.t0 = self.time[0] # 0
        self.t1 = self.time[-1] # interval/sample_freq 초
        self.td = self.t1 - self.t0 
        
    def sig2imfs(self, sig, max_imf = None):
        return eemd.eemd(sig, max_imf = max_imf)

    def plot_imfs(self, sig, imfs):
        plt.figure(figsize=(16, 2))
        plt.title("Signal"); plt.plot(sig, 'k'); plt.xticks([])
        plt.figure(figsize=(16, 8))
        plt.suptitle("IMFs", y = 0.92)
        for i in range(len(imfs)):
            if i == len(imfs) - 1:
                plt.subplot(len(imfs), 1, i+1); plt.plot(imfs[i], 'r')
            else:
                plt.subplot(len(imfs), 1, i+1); plt.plot(imfs[i], 'b'); plt.xticks([])

    def PAhilbert(self, imf):
        """
        Phase, Amplitude of Hilbert
        """
        H = hilbert(imf)
        amp = np.abs(H)
        phase = np.arctan2(H.imag, H.real)
        phase = np.unwrap(phase)
    #     phase = np.unwrap(np.angle(H))
        return phase, amp

    def FAhilbert(self, imfs, sample_freq = 12000):
        """
        Frequency, Amplitude of Hilbert
        """
        n_imfs = imfs.shape[0]
        freq = []
        amp = []
        for i in range(n_imfs - 1):
            imf = imfs[i, :]
            inst_phase, inst_amp = self.PAhilbert(imf)
            inst_freq = (np.diff(inst_phase) / (2*math.pi)*self.sf) 
    #         inst_freq = (2*math.pi) / np.diff(inst_phase)
            inst_freq = np.insert(inst_freq, len(inst_freq), inst_freq[-1])
            inst_amp = np.insert(inst_amp, len(inst_amp), inst_amp[-1])

            freq.append(inst_freq)
            amp.append(inst_amp)

        return np.asarray(freq).T, np.asarray(amp).T

    def hht2image(self, inst_freq, inst_amp, freqsol = 50, timesol = 50, mode = ''):

        inst_freq = np.abs(inst_freq)
        freqsol=freqsol
        if mode == 'exponential':
            bins = np.linspace(4, 13, freqsol)
            p = np.digitize(inst_freq, 2**bins)
        elif mode == 'linear':
            bins = np.linspace(0, 6000, freqsol)
            p = np.digitize(inst_freq, bins)

        timesol = timesol
        t = np.ceil((timesol-1) * (self.time-self.t0) / self.td)
        t = t.astype(int)

        hilbert_spectrum = np.zeros([timesol, freqsol]) 
        for i in range(len(self.time)): # 12000 time step
            for j in range(inst_freq.shape[-1]): # N
                if p[i, j] >= 0 and p[i, j] < freqsol:
                    hilbert_spectrum[t[i], p[i,j]] += inst_amp[i, j]
        hilbert_spectrum = abs(hilbert_spectrum)
        hilbert_spectrum_T = hilbert_spectrum.T

        image = np.flipud(hilbert_spectrum_T)
        return image

    def sig2hht(self, sig, max_imf = None, freqsol = 50, timesol = 50, mode = '', info = False):
        
        imfs = self.sig2imfs(sig, max_imf = max_imf)
        
        inst_freq, inst_amp = self.FAhilbert(imfs)

        if mode == 'channel_wise':
            hhts = []
            for f, a in zip(inst_freq.T, inst_amp.T):
                hhts.append(self.hht2image(f.reshape(-1, 1), a.reshape(-1, 1), freqsol = freqsol, timesol = timesol, mode = 'exponential'))
            hhts = np.asarray(hhts)
        else:
            hhts = self.hht2image(inst_freq, inst_amp, freqsol = freqsol, timesol = timesol, mode = 'linear')
        
        if info != False:
            print("Instantaneous frequency / shape: {}, max: {}, min: {}".format(inst_freq.shape, np.max(inst_freq), np.min(abs(inst_freq))))
            print("Instantaneous amplitude / shape: {}, max: {}, min: {}".format(inst_amp.shape, np.max(inst_amp), np.min(inst_amp)))
            print("Signal / shape:", sig.shape)
            print("IMFs / shape:", imfs.shape)
            print("Hilbert huang transform / shape: {}".format(hhts.shape))

            """
            Signal / shape: (3000,)
            IMFs / shape: (10, 3000)
            Instantaneous frequency / shape: (3000, 9), max: 5997.782445849687, min: 0.0015049612887193785
            Instantaneous amplitude / shape: (3001, 9), max: 0.19043989229003674, min: 3.333147052300031e-05
            Hilbert huang transform / shape: (9, 50, 50)
            """
        return hhts
