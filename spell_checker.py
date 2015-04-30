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


def spell_checker(input_string, lexical_tree=0, transform_list_in=[], transform_list_out=[], templates=[]):
    """
    a simple spell checker
    :param input_string: the string you want to be checked
    :return: the checked string
    """
    dict = read_file('dict.txt').split()
    word_list = list(set(re.split('\W', input_string)))
    word_list = [i for i in word_list if len(i) > 0]
    print templates
    levenshtein_distance_obj = levenshtein_distance.LevenshteinDistance(
        templates=dict if templates == [] else templates, string_or_list=0)
    print levenshtein_distance_obj.templates
    for word in word_list:
        right_word = levenshtein_distance_obj.levenshtein_distance(input_string=word, strategy=0, string_or_list=0,
                                                                   case_sensitive=0, lexical_tree=lexical_tree,
                                                                   transform_list_in=transform_list_in,
                                                                   transform_list_out=transform_list_out)[1].lower()
        input_string = re.sub('(\W)' + word + '(\W)', '\g<1>' + right_word[1:] + '\g<2>', input_string)
        print right_word
    return input_string


def spell_checker_io_using_file(input_filename, output_filename, lexical_tree=0, transform_list_in=[],
                                transform_list_out=[], templates=[]):
    """
    a simple spell checker that input and output using files
    :param input_filename: input filename
    :param output_filename: output filename
    """
    output_file = open(output_filename, 'w')
    output_file.write(spell_checker(read_file(input_filename), lexical_tree=lexical_tree,
                                    transform_list_in=transform_list_in,
                                    transform_list_out=transform_list_out, templates=templates))