# -*- coding: utf-8 -*-
__author__ = 'lufo'

import SR
import kmeans

# 尼玛英文注释写不下去了
def get_info_FSM_table(filename, **kwargs):
    """
    从保存有限状态机转移方式的文件中读取有限状态机的信息
    :param filename: 文件名
    :param kwargs:
    :return: start_states,开始状态列表
             terminal_states,结束状态列表
             nonemitting_transform_list,列表,表示non-emitting state间的转移关系,每个元素为一个字典,如果第i个字典是{1:[1,2]},表示从状态i可以通过1,2两条边转移到状态1,如果为{1:[]},表示无条件转移到状态一
    """
    start_states = []
    terminal_states = []
    nonemitting_transform_list = []
    with open(filename, 'r') as fr:
        for line in fr:
            info = line.split()
            if info[0] == 'N_States:':
                num_states = int(info[1])
                nonemitting_transform_list = [{} for i in xrange(num_states)]
            # in FSM table Start_State and Terminal_States can only list as number split by space
            elif info[0] == 'Start_State:':
                for temp in info[1:]:
                    start_states.append(int(temp))
            elif info[0] == 'Terminal_States:':
                for temp in info[1:]:
                    terminal_states.append(int(temp))
            elif info[0] == 'Edge':
                edges = []
                for temp in info[3:]:
                    if '-' in temp:
                        begin, end = map(int, temp.split('-'))
                        for i in xrange(begin, end + 1):
                            edges.append(i)
                    elif temp.isdigit():
                        edges.append(int(temp))
                    else:
                        pass  # N means there is always a path
                nonemitting_transform_list[int(info[1])][int(info[2])] = edges
            else:
                print 'invalid table'
    return start_states, terminal_states, nonemitting_transform_list


def get_transform_relationship_FSM_table(filename, states_in_each_word=5, using_continuous_feature=0, **kwargs):
    """
    从有限状态机中获取模板,状态转移列表,起始状态,每个nonemitting state所对应的emitting state 在模板中连续
    :param filename: 有限状态机文件名
    :param states_in_each_word: 每个单词有多少states组成
    :param kwargs:
    :return: template,states的模板
             transform_list,第i个元素为列表,代表第i个state可以从哪些states进入
             begin_list,t=0时可进入的states
             word_in_template,第i个表示template中第i个元素代表的word
    """
    start_states, terminal_states, nonemitting_transform_list = get_info_FSM_table(filename)
    number_of_nonemitting_states = len(nonemitting_transform_list)
    template = []
    digit_template = []  # 第i个值表示数字i的template
    number_of_frames_in_each_state = []  # 第i个元素代表state i的frame数
    number_of_frames_in_each_state_for_digit = []  # 第i个元素代表数字i中每个state拥有的frame数量
    covariance_matrix_in_each_state = []
    mean_in_each_state = []
    covariance_matrix_for_each_digit = []
    mean_for_each_digit = []
    word_in_template = []  # 第i个表示template中第i个元素代表的word
    if using_continuous_feature:
        digit_template = SR.get_digit_feature_from_continuous_speech()
    for digit in xrange(0, 10):
        temp = SR.get_templates_and_inputs(digit, number_of_templates=10)
        if not using_continuous_feature:
            result = kmeans.k_means(temp[0], number_of_states=5)
            digit_template.extend(result[0])
            covariance_matrix_for_each_digit.append(result[2])
            mean_for_each_digit.append(result[3])
            temp_number_of_frames_in_each_state = [0 for i in xrange(states_in_each_word)]
            for number_of_frames_in_each_state_in_each_template in result[5]:
                for i in xrange(states_in_each_word):
                    temp_number_of_frames_in_each_state[i] += number_of_frames_in_each_state_in_each_template[i]
            number_of_frames_in_each_state_for_digit.append(temp_number_of_frames_in_each_state)
    # print number_of_frames_in_each_state
    # 获取number_of_emitting_states_begin_from_nonemitting_states
    number_of_emitting_states_begin_from_nonemitting_states = []  # 第i个元素表示源于第i个nonemitting_state的edge包含的states个数
    for nonemitting_state in nonemitting_transform_list:
        # nonemitting_state形如{1: [2, 3, 4, 5, 6, 7, 8, 9], 3: []}
        temp = []
        for edge_list in nonemitting_state.values():
            temp.extend(edge_list)
        for i in temp:
            # 获取template
            covariance_matrix_in_each_state.extend(covariance_matrix_for_each_digit[i])
            mean_in_each_state.extend(mean_for_each_digit[i])
            number_of_frames_in_each_state.extend(number_of_frames_in_each_state_for_digit[i])
            template.extend(digit_template[i])
            word_in_template.append(i)
        number_of_emitting_states = len(set(temp))
        number_of_emitting_states_begin_from_nonemitting_states.append(number_of_emitting_states)
    # 获取begin_states_index_for_each_nonemitting_state
    begin_states_index_for_each_nonemitting_state = []  # 第i个元素表示源于第i个nonemitting_state的edge包含的states之前有多少states
    for i in xrange(number_of_nonemitting_states):
        begin_states_index_for_each_nonemitting_state.append(
            sum(number_of_emitting_states_begin_from_nonemitting_states[:i]))
    # 获取emitting_out_list,第i个元素为一个列表,代表第i个nonemitting state可以进入的states
    emitting_out_list = [[] for i in xrange(number_of_nonemitting_states)]
    for i, nonemitting_state in enumerate(nonemitting_transform_list):
        # nonemitting_state形如{1: [2, 3, 4, 5, 6, 7, 8, 9], 3: []}
        begin_index = begin_states_index_for_each_nonemitting_state[i]
        number_of_edges = sum(map(len, nonemitting_state.values()))
        for j in xrange(number_of_edges):
            emitting_out_list[i].append(states_in_each_word * (begin_index + j))
    changed = True
    # 考虑无条件跳转的情况
    while changed:
        new_emitting_out_list = emitting_out_list[:]
        for i, nonemitting_state in enumerate(nonemitting_transform_list):
            for key in nonemitting_state.keys():
                if nonemitting_state[key] == []:
                    new_emitting_out_list[i].extend(new_emitting_out_list[key])
        changed = (new_emitting_out_list != emitting_out_list)
    emitting_out_list = new_emitting_out_list
    # 获取emitting_in_list,第i个元素为一个列表,代表第i个nonemitting state可以由哪些states进入
    emitting_in_list = [[] for i in xrange(number_of_nonemitting_states)]
    for i, nonemitting_state in enumerate(nonemitting_transform_list):
        # nonemitting_state形如{1: [2, 3, 4, 5, 6, 7, 8, 9], 3: []}
        begin_index = begin_states_index_for_each_nonemitting_state[i]
        j = 0
        for key in nonemitting_state.keys():
            for value in nonemitting_state[key]:
                emitting_in_list[key].append(states_in_each_word * (begin_index + j))
                j += 1
    changed = True
    # 考虑无条件跳转的情况
    while changed:
        new_emitting_in_list = emitting_in_list[:]
        for i, nonemitting_state in enumerate(nonemitting_transform_list):
            for key in nonemitting_state.keys():
                if nonemitting_state[key] == []:
                    new_emitting_in_list[key].extend(new_emitting_in_list[i])
        changed = (new_emitting_in_list != emitting_in_list)
    emitting_in_list = new_emitting_in_list
    # 计算transform_list,每个元素为一个列表,表示可以进入这个state的states
    transform_list = [[] for i in
                      xrange(sum(number_of_emitting_states_begin_from_nonemitting_states) * states_in_each_word)]
    for i in xrange(number_of_nonemitting_states):
        for next_state in emitting_out_list[i]:
            for cur_state in emitting_in_list[i]:
                transform_list[next_state].append(cur_state + 4)
    for i, element in enumerate(transform_list):
        element.append(i)
        if i % states_in_each_word > 0:
            element.append(i - 1)
        if i % states_in_each_word > 1:
            element.append(i - 2)
    begin_states = emitting_out_list[0]
    return template, transform_list, begin_states, word_in_template, number_of_frames_in_each_state, covariance_matrix_in_each_state, mean_in_each_state


def continuous_speech_recognition(input_feature, template, transform_list, begin_states, word_in_template,
                                  number_of_frames_in_each_state, covariance_matrix_in_each_state, mean_in_each_state,
                                  states_in_each_word=5, insertion_penalty=0, **kwargs):
    """
    进行连续语音识别
    :param input_feature: 输入的mfcc特征
    :param template: 模板的mfcc
    :param transform_list: 第i个元素为列表,代表第i个state可以从哪些states进入
    :param begin_states: t=0时可进入的states
    :param word_in_template: 第i个表示template中第i个元素代表的word
    :param states_in_each_word: 模板中每个word有多少states
    :param insertion_penalty: 开始新的word的惩罚,防止出现过多word
    :param kwargs:
    :return:
    """
    length_of_input = len(input_feature)
    length_of_template = len(template)
    cost_matrix = [[float('inf') for i in xrange(length_of_template)] for j in xrange(length_of_input)]
    last_state_index = [[[0, 0] for i in xrange(length_of_template)] for j in xrange(length_of_input)]
    # print begin_states
    # print transform_list[190]
    # 更新输入第一个frame的cost
    for state in begin_states:
        # cost_matrix[0][state] = numpy.linalg.norm(numpy.subtract(template[state], input_feature[0]))  # 欧氏距离
        cost_matrix[0][state] = kmeans.get_mahalanobis_distance(covariance_matrix_in_each_state[state],
                                                                mean_in_each_state[state], input_feature[0])[1]
    # 更新整个矩阵
    for i in xrange(1, length_of_input):
        for j in xrange(length_of_template):
            '''
            for last_state in transform_list[j]:
                new_cost = cost_matrix[i - 1][last_state]
                if j % 5 == 0:
                    new_cost += insertion_penalty
                if cost_matrix[i][j] > new_cost:
                    min_index = last_state
                    cost_matrix[i][j] = new_cost
            cost_matrix[i][j] += numpy.linalg.norm(numpy.subtract(template[j], input_feature[i]))
            last_state_index[i][j] = [i - 1, min_index]
            '''
            for last_state in transform_list[j]:
                edge_cost = (number_of_frames_in_each_state[j] - 10.0) / number_of_frames_in_each_state[
                    j] if j == last_state else 10.0 / number_of_frames_in_each_state[last_state]
                new_cost = cost_matrix[i - 1][last_state] + edge_cost
                if j % 5 == 0:
                    new_cost += insertion_penalty
                if cost_matrix[i][j] > new_cost:
                    min_index = last_state
                    cost_matrix[i][j] = new_cost
            cost_matrix[i][j] += kmeans.get_mahalanobis_distance(covariance_matrix_in_each_state[j],
                                                                 mean_in_each_state[j], input_feature[i])[1]
            last_state_index[i][j] = [i - 1, min_index]
    # 加上[length_of_template - 50:length_of_template]表示只从terminal state取,最后长度一定是4或7
    min_cost = min(cost_matrix[length_of_input - 1][(length_of_template - 50):length_of_template])
    cur_index = cost_matrix[length_of_input - 1].index(min_cost)
    path = []
    i = length_of_input - 1
    while i >= 0:
        if cur_index % 5 == 0:
            path.append(cur_index / states_in_each_word)
        cur_index = last_state_index[i][cur_index][1]
        i -= 1
    # print path
    # word_list去重+翻转
    word_list = []
    last_index = -1
    top_index_in_word_list = -1
    for index in path:
        if last_index != index:
            last_index = index
            times = 1
        else:
            times += 1
        if times > 0 and index != top_index_in_word_list:
            word_list.append(word_in_template[index])
            top_index_in_word_list = index
    word_list.reverse()
    # print min_cost, word_list
    return ''.join(map(str, word_list))
