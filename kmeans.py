# -*- coding: utf-8 -*-
__author__ = 'lufo'

import numpy
import math
import DTW


def get_covariance(data_array):
    """
    get covariance matrix of input data,we only need diagonal data so we assume that all off-diagonal terms in the matrix are 0
    :param data_array:array,every element of it is a data which has some features(in this project the number of features is 39)
    :return:array,the covariance matrix and the mean of data array
    """
    number_of_data = len(data_array)
    number_of_features = len(data_array[0]) if number_of_data != 0 else 0
    covariance_matrix = numpy.zeros([number_of_features, number_of_features])
    mean = numpy.zeros(number_of_features)
    for data in data_array:
        numpy.add(mean, data, mean)
    numpy.divide(mean, number_of_data, mean)
    for i in xrange(number_of_features):
        temp = numpy.zeros(number_of_data)
        for j in xrange(number_of_data):
            temp[j] = data_array[j][i] - mean[i]
        covariance_matrix[i][i] = max(numpy.dot(temp, temp.transpose()), 0.00001) / number_of_data
    return covariance_matrix, mean


def get_mahalanobis_distance(covariance_matrix, mean, segment):
    """
    get mahalanobis distance between a state and a segment
    :param covariance_matrix: array,the covariance matrix of a state
    :param mean: the mean of a state
    :param segment: array,a segment that has some features(in this project the number of features is 39)
    :return: mahalanobis distance,node cost for DTW
    """
    if not len(covariance_matrix):  # this state has no frame
        return [float('inf'), float('inf')]
    inv_covariance_matrix = numpy.linalg.inv(covariance_matrix)
    # print 'covariance_matrix', covariance_matrix
    # print 'inv_covariance_matrix', inv_covariance_matrix
    difference_between_segment_and_mean = numpy.subtract(segment, mean)
    mahalanobis_distance = numpy.dot(numpy.dot(difference_between_segment_and_mean.transpose(), inv_covariance_matrix), \
        difference_between_segment_and_mean)
    # print 'mahalanobis_distance', mahalanobis_distance
    node_cost = 0
    for i in xrange(len(covariance_matrix)):
        node_cost += math.log(2 * math.pi * covariance_matrix[i][i])
    # print 'first part', node_cost
    node_cost += mahalanobis_distance
    node_cost /= 2
    return mahalanobis_distance, node_cost


def initialize_states(templates, number_of_templates, number_of_states=5):
    """
    initialize states for each template,if this template has 12 frames,then the number of frames in each state is 2,2,2,3,3
    :param templates: list,each element of it is a template,each element of a template is a frame,which has 39 features
    :param number_of_states: number of states in each template
    :return: number_of_frames_in_state_for_each_template[i][j] represent for the number of frames in ith template jth state
    """
    number_of_frames_in_each_state_for_each_template = []
    for i in xrange(number_of_templates):
        # get number_of_frames_in_each_state_for_each_template
        length = len(templates[i])
        small_number_of_elements_in_current_state = length / number_of_states  # if length is 12,
        # then there are 3 states have 2 frames and 2 states have 3 frames,we call 2 small number and 3 big number
        number_of_big_number = length % number_of_states
        number_of_frames_in_each_state = [small_number_of_elements_in_current_state for j in \
                                          xrange(number_of_states - number_of_big_number)]
        number_of_frames_in_each_state.extend \
            ([small_number_of_elements_in_current_state + 1 for j in xrange(number_of_big_number)])
        number_of_frames_in_each_state_for_each_template.append(number_of_frames_in_each_state)
    # print number_of_frames_in_each_state_for_each_template
    return number_of_frames_in_each_state_for_each_template


def get_edge_cost(number_of_frames_in_each_state_for_each_template, number_of_templates, number_of_states=5):
    """
    get edge cost from one state to other state
    :param number_of_frames_in_each_state_for_each_template: number_of_frames_in_state_for_each_template[i][j] represent for the number of frames in ith template jth state
    :param number_of_states: number of states in each template
    :return: a edge cost list and state for each template,
            edge_cost[i][j] represent for the cost transform from state i to j,begin from 1,0 represents for dummy state
            state_for_each_template[i][j] represent for the state of ith template's jth frame,begin from 0,no dummy state
    """
    number_of_frames_before_this_state_for_each_template = []
    # state_for_each_template[i][j] represent for the state of ith template's jth frame,begin from 0
    state_for_each_template = []
    for number_of_frames_in_each_state in number_of_frames_in_each_state_for_each_template:
        # get number_of_frames_before_this_state_for_each_template
        number_of_frames_before_this_state = [0 for j in xrange(number_of_states)]
        for j in xrange(1, number_of_states):
            number_of_frames_before_this_state[j] = number_of_frames_before_this_state[j - 1] \
                                                    + number_of_frames_in_each_state[j - 1]
        number_of_frames_before_this_state_for_each_template.append(number_of_frames_before_this_state)
        # get state_for_each_template
        state = []
        for j in xrange(number_of_states):
            state.extend([j for k in xrange(number_of_frames_in_each_state[j])])
        state_for_each_template.append(state)
    edge_cost = []  # edge_cost[i][j] represent for the cost transform from i to j
    for i in xrange(-1, number_of_states):
        edge_cost_i = [0 for j in xrange(number_of_states + 1)]
        number_of_frames_in_ith_state = 0
        number_of_frames_that_next_frame_in_state = [0 for j in xrange(number_of_states)]
        if i is -1:  # begin state
            number_of_frames_in_ith_state = number_of_templates
            for j in xrange(number_of_templates):
                number_of_frames_that_next_frame_in_state[state_for_each_template[j][0]] += 1
        else:
            for j in xrange(number_of_templates):
                number_of_frames_in_ith_state += number_of_frames_in_each_state_for_each_template[j][i]
                if i is not number_of_states - 1:  # the last state can't translate to other state
                    number_of_frames_that_next_frame_in_state[ \
                        state_for_each_template[j][number_of_frames_before_this_state_for_each_template[j][i + 1]]] += 1
        number_of_frames_that_next_frame_in_state[i] = number_of_frames_in_ith_state - sum(
            number_of_frames_that_next_frame_in_state)
        for j in xrange(number_of_states):
            # if ith state has no frame,then we think it's impossible for any frame transform to ith state
            if number_of_frames_in_ith_state == 0 or number_of_frames_that_next_frame_in_state[j] == 0:
                edge_cost_i[j + 1] = float('inf')
            else:
                edge_cost_i[j + 1] = -math.log(
                    number_of_frames_that_next_frame_in_state[j] / float(number_of_frames_in_ith_state))
        edge_cost_i[0] = float('inf')
        edge_cost.append(edge_cost_i)
    return edge_cost, state_for_each_template


def get_covariance_and_mean_for_each_state(templates, state_for_each_template, number_of_states=5):
    """
    get covariance and mean for each state
    :param templates: list,each element of it is a template,each element of a template is a frame,which has 39 features
    :param state_for_each_template[i][j] represent for the state of ith template's jth frame,begin from 0
    :param number_of_states: number of states in each template
    :return: covariance_matrix, a list,covariance_matrix[i] represent for ith state's covariance matrix
    :return: mean,a list,mean[i] represent for ith state's mean
    """
    # total_cost[i][j][k] represent for the ith template's jth frame's total cost from its origin state to state k
    # frames_in_each_state[i] represent for ith state's frames,each of them has 39 features
    frames_in_each_state = [[] for i in xrange(number_of_states)]
    covariance_matrix = []  # covariance_matrix[i] represent for ith state's covariance matrix
    mean = []  # mean[i] represent for ith state's mean
    for i in xrange(len(state_for_each_template)):
        for j in xrange(len(state_for_each_template[i])):
            frames_in_each_state[state_for_each_template[i][j]].append(templates[i][j])
    for frames_in_one_state in frames_in_each_state:
        temp_covariance_matrix, temp_mean = get_covariance(numpy.array(frames_in_one_state))
        covariance_matrix.append(temp_covariance_matrix)
        mean.append(temp_mean)
    return covariance_matrix, mean


def get_number_of_frames_in_each_state_for_each_template_by_state_for_each_template(state_for_each_template,
                                                                                    number_of_states):
    """
    get_number_of_frames_in_each_state_for_each_template_by_state_for_each_template
    :param: state_for_each_template[i][j] represent for the state of ith template's jth frame,begin from 0,no dummy state
    :param number_of_states: number of states in each template
    :return: number_of_frames_in_state_for_each_template[i][j] represent for the number of frames in ith template jth state
    """
    number_of_frames_in_each_state_for_each_template = []
    for i in xrange(len(state_for_each_template)):
        number_of_frames_in_each_state = [0 for j in xrange(number_of_states)]
        for j in xrange(len(state_for_each_template[i])):
            number_of_frames_in_each_state[state_for_each_template[i][j]] += 1
        number_of_frames_in_each_state_for_each_template.append(number_of_frames_in_each_state)
    return number_of_frames_in_each_state_for_each_template


def k_means(templates, number_of_states=5):
    """
    using k-means to templates to get a template has 5 states
    :param templates: list,each element of it is a template,each element of a template is a frame,which has 39 features
    :param number_of_states: number of states in each template
    :return: list,the template after doing k-means
    """
    # we assume that frames in each state is continuous
    # print 'kmeans'
    number_of_templates = len(templates)
    # print 'number_of_templates', number_of_templates
    # number_of_frames_in_state_for_each_template[i][j] represent for the number of frames in ith template jth state
    number_of_frames_in_each_state_for_each_template = initialize_states \
        (templates, number_of_templates, number_of_states)
    edge_cost, state_for_each_template = get_edge_cost(number_of_frames_in_each_state_for_each_template,
                                                       number_of_templates, number_of_states)
    covariance_matrix, mean = get_covariance_and_mean_for_each_state(templates, state_for_each_template,
                                                                     number_of_states)
    cluster_changed = True
    iteration_times = 0
    while cluster_changed:
        # print 'covariance_matrix', covariance_matrix
        # print '\nedge cost', edge_cost
        # print 'mean', mean
        # print 'state_for_each_template', state_for_each_template
        # print 'number_of_frames_in_each_state_for_each_template', number_of_frames_in_each_state_for_each_template
        iteration_times += 1
        #print 'iteration_times', iteration_times
        cluster_changed = False
        viterbi_search_object = DTW.DTW([mean])
        for i in xrange(len(templates)):
            # print 'templates', templates
            number_of_frames = len(templates[i])
            cost, template, path = viterbi_search_object.DTW(templates[i][:], strategy=0, cost_function=1,
                                                             covariance_matrix=covariance_matrix,
                                                             edge_cost=edge_cost)
            #print 'path', path
            #print 'cost', cost
            temp_state_for_one_template = [0 for j in xrange(number_of_frames)]
            for j in xrange(number_of_frames - 1):
                temp_state_for_one_template[j] = path[number_of_frames - 2 - j][1] - 1
            temp_state_for_one_template[number_of_frames - 1] = number_of_states - 1
            if temp_state_for_one_template != state_for_each_template[i]:
                cluster_changed = True
                state_for_each_template[i] = temp_state_for_one_template[:]
        number_of_frames_in_each_state_for_each_template = get_number_of_frames_in_each_state_for_each_template_by_state_for_each_template(
            state_for_each_template, number_of_states)
        #print 'state_for_each_template', state_for_each_template
        #print 'number_of_frames_in_each_state_for_each_template',number_of_frames_in_each_state_for_each_template
        edge_cost = get_edge_cost(number_of_frames_in_each_state_for_each_template,
                                  number_of_templates, number_of_states)[0]
        #print 'edge cost', edge_cost
        covariance_matrix, mean = get_covariance_and_mean_for_each_state(templates, state_for_each_template,
                                                                         number_of_states)
        #print 'covariance_matrix', covariance_matrix
        #print 'mean', mean
    return [map(list, mean)]