# -*- coding: utf-8 -*-
__author__ = 'lufo'

import record
import time
import math
import mfcc
import numpy
import scipy.io.wavfile as wav


def get_mfcc_feat(filename, winstep=0.01, nfilt=40, numcep=13, preemph=0.95, appendEnergy=True):
    """
    Compute MFCC features from an audio signal.

    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param numcep: the number of cepstrum to return, default 13
    :param nfilt: the number of filters in the filterbank, default 26.
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.95.
    :param appendEnergy: if this is true, the zeroth cepstral coefficient is replaced with the log of the total frame energy.
    :returns mfcc_feat: A numpy array of size (NUMFRAMES by numcep) containing features. Each row holds 1 feature vector.
    :returns fbank_feat: A numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector.
    """
    (rate, signal) = wav.read(filename)
    # print signal
    # print type(signal[0])
    if rate == 44100:
        winlen = 0.02
        nfft = 1024
    elif rate == 16000:
        winlen = 0.025
        nfft = 512
    mfcc_feat = mfcc.mfcc(signal=signal, samplerate=rate, winlen=winlen, winstep=winstep, nfft=nfft,
                          nfilt=40, numcep=13, preemph=preemph, appendEnergy=appendEnergy)
    numpy.savetxt("result.txt", mfcc_feat, delimiter=",")
    fbank_feat = mfcc.logfbank(signal=signal, samplerate=rate, winlen=winlen, winstep=winstep, nfft=nfft,
                               nfilt=40, preemph=preemph)
    numpy.savetxt("result2.txt", fbank_feat, delimiter=",")
    return mfcc_feat, fbank_feat


def endpointed_record():
    """
    record a voice and save it in nonblocking.wav,start to record when you input a number and stop when you stop saying
    """
    rec = record.Recorder(channels=1, rate=44100, endpointed=True)
    with rec.open('nonblocking.wav', 'wb') as recfile2:
        input('please input a number to start to record')
        recfile2.start_recording()
        time.sleep(100.0)
        recfile2.stop_recording()


def main():
    # endpointed_record()
    mfcc_feat, fbank_feat = get_mfcc_feat(filename="recorded.wav", winstep=0.01,
                                          nfilt=40, numcep=13, preemph=0.95, appendEnergy=False)


if __name__ == '__main__':
    main()