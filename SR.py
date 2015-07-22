# -*- coding: utf-8 -*-
__author__ = 'lufo'

import record
import time
import math
import random
import mfcc
import numpy
import levenshtein_distance
import spell_checker
import DTW
import re
import kmeans
from scipy import io
import profile
import segment
import generate_lexical_tree
import scipy.io.wavfile as wav
import continuous_speech_recognition_using_FSM
import trained_feature_list

CONTINUOUS_SPEECH_LIST = ['0123456789', '9876543210', '1234567890', '0987654321', '1357902468', '8642097531']
TEST_RECORD1 = ['2345678', '2345', '1234', '2689981', '4568652', '8625362', '6986', '2654', '2567843', '4567',
                '9876543', '2348', '8654321', '4562', '1265', '4512121', '8912456', '2451', '6189', '8642135',
                '1987', '2012456', '0124', '1098', '8901542']
TEST_RECORD2 = ['911385', '826414052002', '8212176342', '7343332190377', '2212', '123456', '6890372344',
                '72184347924', '55555', '37274921']


def get_mfcc_feat(filename, winstep=0.01, nfilt=40, numcep=13, preemph=0.95, appendEnergy=True, begin_index=-1,
                  end_index=-1):
    """
    Compute MFCC features from an audio signal.

    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param numcep: the number of cepstrum to return, default 13
    :param nfilt: the number of filters in the filterbank, default 26.
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.95.
    :param appendEnergy: if this is true, the zeroth cepstral coefficient is replaced with the log of the total frame energy.
    :param begin_index: the frame when record begin
    :param end_index: the frame when record end
    :return mfcc_feat: A numpy array of size (NUMFRAMES by numcep) containing features. Each row holds 1 feature vector.
    :return fbank_feat: A numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector.
    :return normalization_feature: the 39-dimensions feature after finite difference and normalization
    """
    (rate, signal) = wav.read(filename)
    if begin_index != -1:
        signal = signal[begin_index * 44100 * 0.02:end_index * 44100 * 0.02]
    # print signal
    # print type(signal)
    if rate == 44100:
        winlen = 0.02
        nfft = 1024
    elif rate == 16000:
        winlen = 0.025
        nfft = 512
    mfcc_feat = mfcc.mfcc(signal=signal, samplerate=rate, winlen=winlen, winstep=winstep, nfft=nfft,
                          nfilt=nfilt, numcep=numcep, preemph=preemph, appendEnergy=appendEnergy)
    numpy.savetxt("result.txt", mfcc_feat, delimiter=",")
    fbank_feat = mfcc.logfbank(signal=signal, samplerate=rate, winlen=winlen, winstep=winstep, nfft=nfft,
                               nfilt=nfilt, preemph=preemph)
    # numpy.savetxt("result2.txt", fbank_feat, delimiter=",")
    normalization_feature = mfcc.normalization(mfcc_feat)
    return mfcc_feat, fbank_feat, normalization_feature


def get_mfcc_feat_from_file(filename):
    normalization_feature = []
    file = open(filename)
    for frame in file:
        normalization_feature.append(numpy.array(map(float, frame.split())))
    file.close()
    return numpy.array(normalization_feature)


def endpointed_record(filename='recording.wav', time_synchronous=0, using_kmeans=0, read_feature_from_file=0,
                      DTW_obj=[], record_continuous=0):
    """
    record a voice and save it in recording.scipy.io.wavfile,start to record when you input a number and stop when you stop saying
    :param filename: 录音保存到的文件名
    :param time_synchronous: 0: just record
                             1: record and do DTW
    :param using_kmeans: 1 if using k-means to templates to generate some states
    :param DTW_obj:a DTW object,the trained model
    :param record_continuous:1,从开始到结束所有的音都录进去,0,不录静音部分
    """
    rec = record.Recorder(channels=1, rate=44100, endpointed=True, frames_per_buffer=int(44100 * 0.02),
                          using_kmeans=using_kmeans, read_feature_from_file=read_feature_from_file, DTW_obj=DTW_obj,
                          record_continuous=record_continuous)
    with rec.open(filename, 'wb', time_synchronous=time_synchronous) as recfile2:
        input('please input a number to start to record\n')
        recfile2.start_recording()
        input()
        recfile2.stop_recording()


def record_isolated_speech(**kwargs):
    """
    录制单词
    :param kwargs:
    :return:
    """
    for i in xrange(8, 11):
        if i < 10:
            print 'please speak ' + str(i) + ' 10 times'
        else:
            print 'record o 10 times'
        for j in xrange(10):
            endpointed_record(
                filename='./isolated_record_normal/' + (str(i) if i < 10 else 'o') + '_' + str(j) + '.wav',
                record_continuous=1)


def record_continuous_speech(**kwargs):
    """
    为project6录制连续语音
    :param kwargs:
    :return:
    """
    for continuous_speech in CONTINUOUS_SPEECH_LIST:
        print 'please speak ' + continuous_speech + ' 5 times'
        for i in xrange(5):
            endpointed_record(filename='./continuous_speech/' + continuous_speech + '_' + str(i) + '.wav',
                              record_continuous=1)


def record_for_project5(**kwargs):
    """
    为project5录制测试用例
    :return:
    """
    '''
    for test in TEST_RECORD1[]:
        print 'please speak ' + test
        endpointed_record(filename='./record_for_project5_continuous/test1/' + test + '.wav', record_continuous=1)
    '''
    for test in TEST_RECORD2[-3:]:
        print 'please speak ' + test
        endpointed_record(filename='./record_for_project5_continuous/test2/' + test + '.wav', record_continuous=1)


def test_for_project5(**kwargs):
    error = 0
    template, transform_list, begin_states, word_in_template, number_of_frames_in_each_state, covariance_matrix_in_each_state, mean_in_each_state = continuous_speech_recognition_using_FSM.get_transform_relationship_FSM_table(
        '5_1_FSM.txt')
    for test in TEST_RECORD1:
        print 'record is:' + test
        input_feature = get_mfcc_feat('./record_for_project5/test1/' + test + '.wav')[2]
        recognized = continuous_speech_recognition_using_FSM.continuous_speech_recognition(input_feature, template,
                                                                                           transform_list,
                                                                                           begin_states,
                                                                                           word_in_template,
                                                                                           number_of_frames_in_each_state,
                                                                                           covariance_matrix_in_each_state,
                                                                                           mean_in_each_state,
                                                                                           insertion_penalty=200 if len(
                                                                                               test) <= 4 else 0,
                                                                                           using_continuous_feature=0)
        levenshtein_distance_obj = levenshtein_distance.LevenshteinDistance([test], string_or_list=0)
        error += levenshtein_distance_obj.levenshtein_distance(recognized, strategy=0, string_or_list=0)[0]
        print 'recognized is ' + recognized
        print 'error is', error
    error = 0
    template, transform_list, begin_states, word_in_template, number_of_frames_in_each_state, covariance_matrix_in_each_state, mean_in_each_state = continuous_speech_recognition_using_FSM.get_transform_relationship_FSM_table(
        '5_3_FSM.txt')
    for test in TEST_RECORD2:
        print 'record is:' + test
        input_feature = get_mfcc_feat('./record_for_project5/test2/' + test + '.wav')[2]
        recognized = continuous_speech_recognition_using_FSM.continuous_speech_recognition(input_feature, template,
                                                                                           transform_list,
                                                                                           begin_states,
                                                                                           word_in_template,
                                                                                           number_of_frames_in_each_state,
                                                                                           covariance_matrix_in_each_state,
                                                                                           mean_in_each_state,
                                                                                           insertion_penalty=300,
                                                                                           using_continuous_feature=0)
        levenshtein_distance_obj = levenshtein_distance.LevenshteinDistance([test], string_or_list=0)
        error += levenshtein_distance_obj.levenshtein_distance(recognized, strategy=0, string_or_list=0)[0]
        print 'recognized is ' + recognized
        print 'error is', error


def compare_two_articles(filename1, filename2):
    """
    compute levenshtein distance between two articles
    :param filename1: article1's filename
    :param filename2: article2's filename
    :return: levenshtein distance between two articles
    """
    templates = [re.split('\W+', spell_checker.read_file(filename1).strip())]
    input_string = re.split('\W+', spell_checker.read_file(filename2).strip())
    print templates
    print input_string
    levenshtein_distance_obj = levenshtein_distance.LevenshteinDistance(templates=templates, string_or_list=1)
    print levenshtein_distance_obj.levenshtein_distance(input_string=input_string, strategy=1, string_or_list=1)


def get_templates_and_inputs(digit, number_of_templates, read_feature_from_file=0):
    """
    split 10 records of a digit to templates and inputs random
    :param digit: the digit you want to split,0-9
    :param number_of_templates: templates' numbers
    :param read_feature_from_file:1 then read mfcc feature from file,0 then compute mfcc feature from records
    :return: a template list and a input list
    """
    templates_index_list = random.sample(xrange(0, 10), number_of_templates)
    inputs_list = []
    templates_list = []
    for i in xrange(0, 10):
        if i in templates_index_list:
            if read_feature_from_file:
                templates_list.append(
                    map(list, get_mfcc_feat_from_file('./feature/' + str(digit) + '_' + str(i) + '.txt')))
            else:
                templates_list.append( \
                    map(list, get_mfcc_feat(filename="./records/" + str(digit) + '_' + str(i) + '.wav', winstep=0.01, \
                                            nfilt=40, numcep=13, preemph=0.95, appendEnergy=True)[2]))

        else:
            if read_feature_from_file:
                inputs_list.append(
                    map(list, get_mfcc_feat_from_file('./feature/' + str(digit) + '_' + str(i) + '.txt')))
            else:
                inputs_list.append( \
                    map(list, get_mfcc_feat(filename="./records/" + str(digit) + '_' + str(i) + '.wav', winstep=0.01, \
                                            nfilt=40, numcep=13, preemph=0.95, appendEnergy=True)[2]))
    return templates_list, inputs_list


def get_isolated_templates(i, number_of_records=10):
    """
    获取第i个isolated语音的mfcc
    :param i: 第i个,0-9表示数字0-9,10表示silence
    :param number_of_records: 选取的录音数
    :return:
    """
    mfcc_list = []
    for j in xrange(number_of_records):
        filename = './records/' + str(i) + '_' + str(j) + '.wav'
        print filename
        mfcc_list.append( \
            map(list, get_mfcc_feat(filename=filename, winstep=0.01, \
                                    nfilt=40, numcep=13, preemph=0.95, appendEnergy=True)[2]))
    return mfcc_list


def get_continuous_templates(i, number_of_records=5):
    """
    获取第i个连续语音的mfcc,使用number_of_records个录音做训练
    :return: templates_list,每个元素是第i个连续语音的mfcc
    """
    mfcc_list = []
    for j in xrange(number_of_records):
        filename = './continuous_speech/' + CONTINUOUS_SPEECH_LIST[i] + '_' + str(j) + '.wav'
        print filename
        mfcc_list.append( \
            map(list, get_mfcc_feat(filename=filename, winstep=0.01, \
                                    nfilt=40, numcep=13, preemph=0.95, appendEnergy=True)[2]))
    return mfcc_list


def get_digit_template_from_continuous_speech():
    """
    从连续语音中训练出每种连续录音的特征并写入文件
    :return:
    """
    trained_feature_list = []  # 第i个元素为第i个连续录音的训练后的特征,每个特征有60states,每5个states代表一个单词,每个states有39个特征
    for i in xrange(6):
        mfcc_list = get_continuous_templates(i)
        # trained_feature_list.append(kmeans.k_means(mfcc_list,60))
        trained_feature_list.append(
            kmeans.train_continuous_model(mfcc_list, CONTINUOUS_SPEECH_LIST[i])[0])  # 10个数字,2个sil
        print len(trained_feature_list[0]), len(trained_feature_list[0][0])
    with open('trained_feature_list.py', 'w') as fw:
        fw.write('trained_feature_list=' + str(trained_feature_list))


def get_digit_feature_from_continuous_speech(states_for_each_word=5):
    """
    从连续录音的特征中分出每个单词的特征,设每个特征为feature,feature[0]有5个states,feature[0][0]表示第1个state,每个state有39维的特征
    :return:
    """
    # covariance_matrix, mean, number_of_frames_in_each_state_for_each_template
    digit_feature = [[[0 for k in xrange(39)] for j in xrange(states_for_each_word)] for i in
                     xrange(11)]  # 0-9个元素表示0-9这10个数字,第11个元素表示silence
    added_times = [0 for i in xrange(11)]  # 第i个元素代表第i个word的feature是由几个加起来的
    my_trained_feature_list = trained_feature_list.trained_feature_list  # 有6个元素,每个代表一种连续语音的特征,每个特征有60个states
    for i, trained_feature in enumerate(my_trained_feature_list):
        trained_feature = trained_feature[0]
        j = 0
        while j < len(trained_feature):
            if j == 0 or j == 55:  # silence
                index = 10
            else:
                index = int(CONTINUOUS_SPEECH_LIST[i][j / 5 - 1])
            added_times[index] += 1
            for k in xrange(5):
                digit_feature[index][k] = list(
                    map(lambda x: x[0] + x[1], zip(digit_feature[index][k], trained_feature[j + k])))
            j += 5
    for i in xrange(11):
        for j in xrange(states_for_each_word):
            digit_feature[i][j] = map(lambda x: x / float(added_times[i]), digit_feature[i][j])
    return digit_feature


def training_model(number_of_templates, using_kmeans, read_feature_from_file=0):
    """
    training the model using DTW to recognize the records
    :param number_of_templates: int, using number_of_templates to train the model
    :param using_kmeans: 1 if using k-means to templates to generate some states
    :param read_feature_from_file:1 then read mfcc feature from file,0 then compute mfcc feature from records
    :return: DTW_obj:a DTW object,the trained model
    :return: inputs:list,every element of it is a template using for test
    """
    templates = []
    inputs = []
    for digit in xrange(0, 10):
        temp = get_templates_and_inputs(digit, number_of_templates, read_feature_from_file)
        # print 'temp',temp[0]
        # print 'k_means(temp[0], number_of_states=5)',kmeans.k_means(temp[0], number_of_states=5)
        templates.extend(
            kmeans.k_means(temp[0], number_of_states=5)[0] if using_kmeans and number_of_templates == 5 else temp[0])
        inputs.extend(temp[1])
    DTW_obj = DTW.DTW(templates)
    return DTW_obj, inputs


def test(time_synchronous, pruning, number_of_templates, using_kmeans, begin_index, end_index, DTW_obj, inputs=[]):
    """
    :param time_synchronous: 0: using recorded data as input
                             1: using time synchronous record as input
    :param begin_index: the frame when record begin
    :param end_index: the frame when record end
    :param pruning: pruning if equals to 1
    :param using_kmeans: 1 if using k-means to templates to generate some states
    :param: DTW_obj:a DTW object,the trained model
    :param: inputs:list,every element of it is a template using for test
    """
    if not time_synchronous:
        # when pruning equals to 0,use numbers of templates 1-5 to compare once,otherwise use 5 templates to compare until accuracy is big enough
        if pruning == 0 or number_of_templates == 5:
            accuracy = 0.0
            total_accuracy = 1.0  # product of every loop's accuracy
            while accuracy < 0.7:
                wrong = 0
                for i in xrange(0, len(inputs)):
                    right_digit = i / (10 - number_of_templates)
                    distance, predict_digit, path = DTW_obj.DTW(inputs[i], strategy=pruning,
                                                                accuracy=total_accuracy,
                                                                number_of_templates=1 if kmeans else number_of_templates)
                    if right_digit != predict_digit:
                        wrong += 1
                accuracy = 1 - float(wrong) / len(inputs)
                total_accuracy *= accuracy
                print 'when using ' + str(number_of_templates) + ' templates the accuracy is', accuracy
                if pruning == 0:
                    break
    else:
        if number_of_templates == 5:
            input = map(list, get_mfcc_feat(filename='recording.wav', winstep=0.01, \
                                            nfilt=40, numcep=13, preemph=0.95, appendEnergy=False, \
                                            begin_index=begin_index, end_index=end_index)[2])
            distance, template_index, path = DTW_obj.DTW(input, 0)
            predict_digit = template_index if using_kmeans else  template_index / number_of_templates
            print 'when using ' + str(number_of_templates) + ' predict you spoke', predict_digit


def test_DTW(time_synchronous=0, begin_index=0, end_index=0, pruning=0, using_kmeans=0, read_feature_from_file=0,
             DTW_obj=[], inputs=[]):
    """
    test all records in inputs list using different number of templates(1-5)
    :param time_synchronous: 0: using recorded data as input
                             1: using time synchronous record as input
    :param begin_index: the frame when record begin
    :param end_index: the frame when record end
    :param pruning: pruning if equals to 1
    :param using_kmeans: 1 if using k-means to templates to generate some states
    :param read_feature_from_file:1 then read mfcc feature from file,0 then compute mfcc feature from records
    """
    for number_of_templates in xrange(1, 6):
        if number_of_templates == 5 or (not using_kmeans):
            if DTW_obj == [] or inputs == []:
                DTW_obj, inputs = training_model(number_of_templates, using_kmeans, read_feature_from_file)
            test(time_synchronous, pruning, number_of_templates, using_kmeans, begin_index, end_index, DTW_obj, inputs)


def test_continuous_speech_recognition_using_FSM(using_continuous_feature=0):
    input_feature = get_mfcc_feat('recording.wav')[2]
    template, transform_list, begin_states, word_in_template = continuous_speech_recognition_using_FSM.get_transform_relationship_FSM_table(
        '5_1_FSM.txt')
    print continuous_speech_recognition_using_FSM.continuous_speech_recognition(input_feature, template, transform_list,
                                                                                begin_states, word_in_template,
                                                                                insertion_penalty=0,
                                                                                using_continuous_feature=using_continuous_feature)


def main():
    # record_isolated_speech()
    # get_digit_template_from_continuous_speech()
    record_for_project5()
    # test_for_project5()
    # endpointed_record(record_continuous=0)
    # record_continuous_speech()
    # print type(mfcc_feat)
    # print  levenshtein_distance.levenshtein_distance('elephan', ['lephant'], 1)
    # templates = ['abc']
    # levenshtein_distance_obj = levenshtein_distance.LevenshteinDistance(templates, string_or_list=0)
    # print levenshtein_distance_obj.levenshtein_distance('bc', strategy=1, string_or_list=0)
    # spell_checker.spell_checker_io_using_file('story.txt', 'storychecked.txt')
    # compare_two_articles('storychecked.txt', 'storycorrect.txt')
    # normalization_feature_0 = map(list, normalization_feature_0)
    # normalization_feature_1 = map(list, normalization_feature_1)
    # DTW_obj = DTW.DTW([normalization_feature_0])
    # distance, template_index, path = DTW_obj.DTW(normalization_feature_1, 0)
    # print '\nthis is the result using DTW without pruning\n'
    # test_DTW(pruning=0, using_kmeans=0, read_feature_from_file=0,DTW_obj=DTW_obj)
    # print '\nthis is the result using DTW with pruning\n'
    # test_DTW(pruning=1, using_kmeans=0, read_feature_from_file=0,DTW_obj=DTW_obj)
    # DTW_obj, inputs = training_model(number_of_templates=5, using_kmeans=1, read_feature_from_file=0)
    # print '\nthis is the result using HMM model with pruning\n'
    # test_DTW(pruning=1, using_kmeans=1, read_feature_from_file=0, DTW_obj=DTW_obj, inputs=inputs)
    # input('input a number to begin\n')
    # endpointed_record(time_synchronous=1, using_kmeans=1, read_feature_from_file=0, DTW_obj=DTW_obj)
    # templates = [[[1], [2], [100], [100], [5], [6], [7], [8], [9], [10]], \
    # [[100], [100], [55], [88], [12], [99], [2], [3], [33], [10]]]
    # kmeans.k_means(templates)
    # lexical_tree = generate_lexical_tree.generate_lexical_tree(
    # generate_lexical_tree.get_word_list_from_file('dict_1.txt'))
    # lexical_tree = generate_lexical_tree.generate_lexical_tree(['booking', 'booming'])
    # transform_list_out = generate_lexical_tree.generate_transform_list_out(lexical_tree, ' ')
    # transform_list_in = generate_lexical_tree.generate_transform_list_in(transform_list_out)
    # templates = [''.join([i.keys()[0] for i in transform_list_out])[1:]]
    # levenshtein_distance_obj = levenshtein_distance.LevenshteinDistance(templates, string_or_list=0)
    # spell_checker.spell_checker_io_using_file('typos.txt', 'checked.txt', lexical_tree=1,
    # transform_list_in=transform_list_in,
    #                                          transform_list_out=transform_list_out, templates=templates)
    # print levenshtein_distance_obj.levenshtein_distance('aaaziefoose', strategy=1, string_or_list=0, lexical_tree=1,
    #                                                    transform_list_in=transform_list_in,
    #                                                    transform_list_out=transform_list_out)
    # compare_two_articles('segmented.txt', 'segmented1.txt')
    # segment.segment('unsegmented0.txt', 'dict_1.txt')


if __name__ == '__main__':
    # profile.run("main()")
    main()
