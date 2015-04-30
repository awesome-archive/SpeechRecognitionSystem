# -*- coding: utf-8 -*-
__author__ = 'lufo'

import generate_lexical_tree
import levenshtein_distance
import re


def segment(filename, dictname):
    """

    :param filename:
    :param dictname:
    :return:
    """
    unsegment_text = re.sub(re.compile('\s+'), '', open(filename).read())
    #lexical_tree = generate_lexical_tree.generate_lexical_tree(['sea', 'tree'])
    lexical_tree = generate_lexical_tree.generate_lexical_tree(generate_lexical_tree.get_word_list_from_file(dictname))
    transform_list_out = generate_lexical_tree.generate_transform_list_out(lexical_tree, ' ')
    templates = [''.join([i.keys()[0] for i in transform_list_out])[1:]]
    levenshtein_distance_obj = levenshtein_distance.LevenshteinDistance(templates, string_or_list=0)
    begin_distance = generate_lexical_tree.generate_begin_distance(levenshtein_distance_obj, transform_list_out)
    # for list in transform_list_out:
    #    if not list.values()[0]:
    #        list.values()[0].append(0)
    transform_list_in = generate_lexical_tree.generate_transform_list_in(transform_list_out)
    min_distance, best_template, path = levenshtein_distance_obj.levenshtein_distance \
        (unsegment_text, strategy=1, string_or_list=0, lexical_tree=1, \
         transform_list_in=transform_list_in, transform_list_out=transform_list_out, begin_distance=begin_distance,
         segment=1)
    print min_distance
    print best_template
    print path