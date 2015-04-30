# -*- coding: utf-8 -*-
__author__ = 'lufo'

import numpy
import pyaudio
import wave
import struct
import SR


class Recorder(object):
    """
    recorder class for recording audio to a WAV file.
    Records in mono by default.
    """

    def __init__(self, channels=1, rate=44100, frames_per_buffer=1024, endpointed=False, using_kmeans=0,
                 read_feature_from_file=0, DTW_obj=[]):
        self.channels = channels
        self.rate = rate
        self.frames_per_buffer = frames_per_buffer
        self.endpointed = endpointed
        self.using_kmeans = using_kmeans
        self.read_feature_from_file = read_feature_from_file
        self.DTW_obj = DTW_obj

    def open(self, filename, mode='wb', time_synchronous=0):
        return RecordingFile(filename, mode, self.channels, self.rate,
                             self.frames_per_buffer, self.endpointed, time_synchronous, self.using_kmeans,
                             self.read_feature_from_file, DTW_obj=self.DTW_obj)


class RecordingFile(object):
    """
    Recording when there is sounds if time_synchronous = 0,else recording all the time
    """

    def __init__(self, filename, mode, channels,
                 rate, frames_per_buffer, endpointed, time_synchronous, using_kmeans, read_feature_from_file=0,
                 DTW_obj=[]):
        self.filename = filename
        self.mode = mode
        self.channels = channels
        self.rate = rate
        self.frames_per_buffer = frames_per_buffer
        self._pa = pyaudio.PyAudio()
        self.wavefile = self._prepare_file(self.filename, self.mode)
        self._stream = None
        self.endpointed = endpointed
        self.forget_factor = 1.0
        self.level = 0.0
        self.threshold = 0.0
        self.background = 0.0
        self.adjustment = 0.05
        self.is_speech = False
        self.past_is_speech = False
        self.max_energy = 0.0
        self.index = 0.0  # 滤掉前五帧
        self.begin_index = 0
        self.end_index = 0
        self.time_synchronous = time_synchronous
        self.using_kmeans = using_kmeans
        self.DTW_obj = DTW_obj
        self.need_train = 1
        self.read_feature_from_file = read_feature_from_file

    def __enter__(self):
        return self

    def __exit__(self, exception, value, traceback):
        self.close()

    def record(self, duration):
        # Use a stream with no callback function in blocking mode
        self._stream = self._pa.open(format=pyaudio.paInt16,
                                     channels=self.channels,
                                     rate=self.rate,
                                     input=True,
                                     frames_per_buffer=self.frames_per_buffer)
        for _ in range(int(self.rate / self.frames_per_buffer * duration)):
            audio = self._stream.read(self.frames_per_buffer)
            self.wavefile.writeframes(audio)
        return None

    def start_recording(self):
        # Use a stream with a callback in non-blocking mode
        self._stream = self._pa.open(format=pyaudio.paInt16,
                                     channels=self.channels,
                                     rate=self.rate,
                                     input=True,
                                     frames_per_buffer=self.frames_per_buffer,
                                     stream_callback=self.callback())
        self._stream.start_stream()
        return self

    def stop_recording(self):
        self._stream.stop_stream()
        return self

    def callback(self):
        def callback(in_data, frame_count, time_info, status):
            state = pyaudio.paContinue
            if self.endpointed:
                current = []
                for i in range(0, frame_count):
                    current.append(struct.unpack('h', in_data[2 * i:2 * i + 2])[0])
                    # print current
                self.endpointing(current)
                if self.is_speech or self.time_synchronous:
                    self.wavefile.writeframes(in_data)
            return in_data, state

        return callback


    def close(self):
        self._stream.close()
        self._pa.terminate()
        self.wavefile.close()

    def _prepare_file(self, filename, mode='wb'):
        wavefile = wave.open(filename, mode)
        wavefile.setnchannels(self.channels)
        wavefile.setsampwidth(self._pa.get_sample_size(pyaudio.paInt16))
        wavefile.setframerate(self.rate)
        return wavefile

    def endpointing(self, current):
        energy = 0.0
        # print current
        # print len(current)
        self.is_speech = False
        for i in range(len(current)):
            energy += pow(current[i], 2)
        energy = 10 * numpy.log(energy)
        # print self.threshold, self.level, self.background
        # print energy
        if self.index == 0:
            self.level = energy
        # threshold equals to max energy of first ten frames divide 4
        if self.index < 10 and self.index >= 0:
            if self.max_energy < energy:
                self.max_energy = energy
            self.background += energy
            if self.index == 9:
                self.background /= 10
                self.threshold = self.max_energy / 4
        if self.index >= 10:
            if energy < self.background:
                self.background = energy
            else:
                self.background += (energy - self.background) * self.adjustment
            self.level = (self.level * self.forget_factor + energy) / (self.forget_factor + 1)
            if self.level < self.background:
                self.level = self.background
            if self.level - self.background > self.threshold:
                self.is_speech = True
            if self.is_speech != self.past_is_speech:
                if self.is_speech:
                    self.begin_index = self.index
                    # print "speech begin at %d\n" % self.begin_index
                    # print 'energy = %d\n' % energy
                else:
                    self.end_index = self.index
                    if self.time_synchronous and self.end_index - self.begin_index > 8:
                        # if self.need_train:
                        #    self.need_train = 0
                        #    self.DTW_obj = SR.training_model(5, self.using_kmeans)[0]
                        SR.test(self.time_synchronous, 1, 5, self.using_kmeans, self.begin_index,
                                self.end_index, self.DTW_obj, self.read_feature_from_file)
                        # print "speech end at %d\n" % self.end_index
                        # print 'energy = %d\n' % energy
            self.past_is_speech = self.is_speech
        self.index += 1