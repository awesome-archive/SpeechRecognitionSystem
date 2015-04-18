# -*- coding: utf-8 -*-
__author__ = 'lufo'

import re
import levenshtein_distance


def read_file(filename):
    """
    load .txt file
    :param filename: just filename
    :return: string
    """
    return open(filename).read()


def spell_checker(input_string):
    """
    a simple spell checker
    :param input_string: the string you want to be checked
    :return: the checked string
    """
    dict = read_file('dict.txt').split()
    word_list = list(set(re.split('\W', input_string)))
    word_list = [i for i in word_list if len(i) > 0]
    levenshtein_distance_obj = levenshtein_distance.LevenshteinDistance(templates=dict, string_or_list=0)
    for word in word_list:
        input_string = re.sub('(\W)' + word + '(\W)', '\g<1>' + levenshtein_distance_obj.levenshtein_distance \
                                                                    (input_string=word, strategy=0, string_or_list=0,
                                                                     case_sensitive=0)[1][1:] + '\g<2>', input_string)
        print word
    return input_string


def spell_checker_io_using_file(input_filename, output_filename):
    """
    a simple spell checker that input and output using files
    :param input_filename: input filename
    :param output_filename: output filename
    """
    output_file = open(output_filename, 'w')
    output_file.write(spell_checker(read_file(input_filename)))