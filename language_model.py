# -*- coding: utf-8 -*-
__author__ = 'lufo'

import cPickle
import math

# 训练语言模型，将每个字母当作一个单词，共有26个小写字母+空格共27个token

COUNT = {}  # dict，key为单词序列，value为出现次数
P = {}  # dict，key为单词序列，value为概率，p['abc']表示已知bc出现，下一个是a的概率
TOTAL_TOKENS = 0  # TOTAL_TOKENS，所有token出现总次数
alpha = {}  # lambda是保留字


def dict_add_one(dict, key, **kwargs):
    """
    给词典dict[key]加1
    """
    dict[key] = 1 + dict[key] if dict.has_key(key) else 1


def get_n_gram_COUNT(training_filename, **kwargs):
    """
    获取训练数据中长度为1，2，3的每个单词序列出现的次数和所有token出现总次数
    :param training_filename: 训练数据文件名
    :param kwargs:
    """
    global COUNT, TOTAL_TOKENS
    with open(training_filename) as train_data:
        for sentence in train_data:
            sentence = sentence.strip() + '%'  # '%'代表句子结束(</s>)
            TOTAL_TOKENS += (len(sentence))
            for i in xrange(len(sentence)):
                if i == 0:
                    dict_add_one(COUNT, sentence[i])
                    dict_add_one(COUNT, '$' + sentence[i])  # '$'代表句子开头有(<s>)
                elif i == 1:
                    dict_add_one(COUNT, sentence[i])
                    dict_add_one(COUNT, sentence[i - 1:i + 1])
                    dict_add_one(COUNT, '$' + sentence[i - 1:i + 1])
                else:
                    dict_add_one(COUNT, sentence[i])
                    dict_add_one(COUNT, sentence[i - 1:i + 1])
                    dict_add_one(COUNT, sentence[i - 2:i + 1])


def get_p(word_sequence, **kwargs):
    """
    计算语言模型单词序列出现的概率
    :param word_sequence: abc表示计算已知ab出现，下一个是c的概率
    :param kwargs:
    """
    global P, COUNT, TOTAL_TOKENS, alpha
    if not alpha.has_key(word_sequence):
        alpha[word_sequence] = 0.5
    if P.has_key(word_sequence):
        return P[word_sequence]
    elif len(word_sequence) == 1:
        P[word_sequence] = COUNT[word_sequence] / float(TOTAL_TOKENS)
    else:
        if not COUNT.has_key(word_sequence[1:]):
            P[word_sequence] = get_p(word_sequence[1:])
        elif not COUNT.has_key(word_sequence):
            # P[word_sequence] = (1 - alpha[word_sequence]) * get_p(word_sequence[1:])
            P[word_sequence] = get_p(word_sequence[1:])
        else:
            P[word_sequence] = alpha[word_sequence] * COUNT[word_sequence] / COUNT[word_sequence[1:]] + (1 - alpha[
                word_sequence]) * get_p(word_sequence[1:])
    return P[word_sequence]


def test(test_filename, n, **kwargs):
    """
    获得测试集每个句子的概率和所有句子长度之和
    :param test_filename: 测试文件名
    :param n: n_gram，n<=3，n=1表示unigram，n=2表示bigram，n=3表示trigram
    :param kwargs:
    :return: p,list 每个元素为一个句子的概率，M 所有句子长度之和
    """
    if n > 3:
        print 'n must smaller than 4'
        return -1
    p = []
    M = 0
    with open(test_filename) as test_data:
        for sentence in test_data:
            sentence = sentence.strip() + '%'
            sentence_length = len(sentence)
            M += sentence_length
            p_temp = 1
            for i in xrange(len(sentence)):
                if n == 1:
                    word_sequence = sentence[i]
                elif n == 2:
                    word_sequence = '$' + sentence[i] if i == 0 else sentence[i - 1:i + 1]
                else:
                    if i == 0:
                        word_sequence = '$' + sentence[i]
                    elif i == 1:
                        word_sequence = '$' + sentence[i - 1:i + 1]
                    else:
                        word_sequence = '$' + sentence[i - 2:i + 1]
                # p_temp *= pow(get_p(word_sequence), -1.0 / sentence_length)
                p_temp *= get_p(word_sequence)
            # print p_temp
            p.append(p_temp)
    return p, M


def get_perplexity(p, M, **kwargs):
    """
    获取perplexity
    :param p: list 每个元素为一个句子的概率
    :param M: 所有句子长度之和
    :param kwargs:
    :return: perplexity
    """
    return pow(2, sum(map(lambda x: math.log(x, 2), p)) / -M)


def get_alpha(n, **kwargs):
    """
    训练得到alpha
    :param kwargs:
    :return:
    """
    global alpha, P
    train_filename = 'dev.txt'
    test(train_filename, n)  # 初始化alpha
    next_alpha = alpha
    for j in xrange(5):
        for key in alpha.keys():
            min = float('inf')
            temp = alpha[key]
            for i in [0, 0.99]:
                alpha[key] = i
                p = sum(test(train_filename, n))
                alpha[key] = temp
                if p < min:
                    print key, p, i
                    min = p
                    next_alpha[key] = i
                P = {}
        alpha = next_alpha


def main():
    global alpha
    get_n_gram_COUNT('train.txt')
    '''
    get_alpha(2)
    print 'get tri'
    get_alpha(3)
    '''
    # 载入训练得到的参数
    with open('alpha.pkl', 'rb') as fr:
        alpha = cPickle.load(fr)
    p, M = test('test.txt', 1)
    print 'unigram perplexity =', get_perplexity(p, M)
    p, M = test('test.txt', 2)
    print 'bigram perplexity =', get_perplexity(p, M)
    p, M = test('test.txt', 3)
    print 'trigram perplexity =', get_perplexity(p, M)


if __name__ == '__main__':
    main()
