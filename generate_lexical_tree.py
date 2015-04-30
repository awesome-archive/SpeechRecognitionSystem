# -*- coding: utf-8 -*-
__author__ = 'lufo'

import Queue

index = 1


def get_word_list_from_file(filename):
    """
    get_word_list_from_file
    :param filename: string,name of dict file,in this file each line has a word
    :return: list,each element is a word in dict
    """
    dict = open(filename)
    word_list = []
    for word in dict:
        word_list.append(word.lower().strip())  # delete '\n' in every word's end
    return word_list


def generate_lexical_tree(word_list):
    """
    generate lexical tree from a dict
    :param word_list: list,each element is a word in dict
    :return: a dict represent a lexical tree
    """
    lexical_tree = {}
    generate_sub_lexical_tree(lexical_tree, word_list, 0)
    return lexical_tree


def generate_sub_lexical_tree(sub_lexical_tree, sub_word_list, i):
    """
    generate a sub lexical tree
    :param sub_lexical_tree: a dict represent a lexical tree
    :param sub_word_list: a list that include words in this sub tree
    :param i: int,this sub tree's root is ith letter
    """
    for word in sub_word_list:
        if len(word) > i:
            if not sub_lexical_tree.has_key(word[i]):
                sub_lexical_tree[word[i]] = {}
                if len(word) == i + 1:
                    sub_lexical_tree[word[i].upper()] = {}
    for item in sub_lexical_tree.items():
        new_word_list = [word for word in sub_word_list if len(word) > i and word[i] == item[0]]
        generate_sub_lexical_tree(item[1], new_word_list, i + 1)


def generate_transform_list_out(lexical_tree, root_node):
    """
    generate a transform list from a lexical tree
    :param lexical_tree: a dict represent a lexical tree
    :param root_node: a char represent this tree's root node
    :return: a list,each element is a dict represent a node in this lexical tree,
    this dict's key is the root node of the tree,and it's value is a list include position of this tree's subnodes
    """
    global index
    transform_list_out = [{root_node: []}]
    for item in lexical_tree.items():
        transform_list_out[0][root_node].append(index)
        index += 1
        transform_list_out.extend(generate_transform_list_out(item[1], item[0]))
    return transform_list_out


def generate_transform_list_in(transform_list_out):
    """
    using the transform list that save each node's subnodes to generate the transform list that save ach node's parent nodes
    :param transform_list_out: a list,each element is a dict represent a node in this lexical tree,
    this dict's key is the root node of the tree,and it's value is a list include position of this tree's subnodes
    :return: a list,each element is a dict represent a node in this lexical tree,
    this dict's key is the root node of the tree,and it's value is a list include position of this tree's parent nodes
    """
    transform_list_in = [{i.keys()[0]: []} for i in transform_list_out]
    for i, sub_transform_list in enumerate(transform_list_out):
        for sub_node_index in sub_transform_list.values()[0]:
            transform_list_in[sub_node_index].values()[0].append(i)
    return transform_list_in


def generate_begin_distance(levenshtein_distance_obj,transform_list_out):
    """
    generate begin distance matrix to compute levenshtein distance using lexical tree
    :param: levenshtein_distance_obj,a levenshtein_distance_obj to get the length of distance matrix
    :param: temp_transform_list_out,a list,each element is a dict represent a node in this lexical tree,
    this dict's key is the root node of the tree,and it's value is a list include position of this tree's parent nodes
    :return: a list, ith element is the ith nodes begin cost in lexical tree
    """
    distance = levenshtein_distance_obj.begin_distance[:]
    q = Queue.Queue()
    q.put([transform_list_out[0], 1])
    while not q.empty():
        subnodes = q.get()
        for node_index in subnodes[0].values()[0]:
            distance[node_index] = subnodes[1]
            q.put([transform_list_out[node_index], subnodes[1] + 1])
    return distance