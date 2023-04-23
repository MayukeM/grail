import os
import pdb
import numpy as np
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt


def plot_rel_dist(adj_list, filename):  # 这是一个画图的函数，画出每个关系的数量
    rel_count = []
    for adj in adj_list:
        rel_count.append(adj.count_nonzero())

    fig = plt.figure(figsize=(12, 8))
    plt.plot(rel_count)
    fig.savefig(filename, dpi=fig.dpi)


def process_files(files, saved_relation2id=None):
    '''
    files: Dictionary map of file paths to read the triplets from.  文件路径字典
    saved_relation2id: Saved relation2id (mostly passed from a trained model) which can be used to map relations to pre-defined indices and filter out the unknown ones.
    '''
    entity2id = {}  # 实体到id的映射
    relation2id = {} if saved_relation2id is None else saved_relation2id  # 关系到id的映射,如果没有传入关系到id的映射，就新建一个空字典

    triplets = {}  # 三元组字典，key是文件类型，value是三元组的列表

    ent = 0  # 实体的id
    rel = 0  # 关系的id

    for file_type, file_path in files.items():  # 遍历文件路径字典  file_type是文件类型，file_path是文件路径
        # file_path:'utils/../data/WN18RR_v1/train.txt'
        data = []  # 三元组列表
        with open(file_path) as f:
            file_data = [line.split() for line in f.read().split('\n')[:-1]]
            # ['06083243', '_hypernym', '06037666']
        for triplet in file_data:
            if triplet[0] not in entity2id:
                entity2id[triplet[0]] = ent
                ent += 1
            if triplet[2] not in entity2id:
                entity2id[triplet[2]] = ent
                ent += 1
            if not saved_relation2id and triplet[1] not in relation2id:  # 如果没有传入关系到id的映射，就新建一个空字典
                relation2id[triplet[1]] = rel  # 关系到id的映射
                rel += 1

            # Save the triplets corresponding to only the known relations  只保存已知关系对应的三元组
            if triplet[1] in relation2id:  # 如果关系在关系到id的映射中
                data.append([entity2id[triplet[0]], entity2id[triplet[2]], relation2id[triplet[1]]])

        triplets[file_type] = np.array(data)  # 三元组字典，key是文件类型，value是三元组的列表

    id2entity = {v: k for k, v in entity2id.items()}  # id到实体的映射
    id2relation = {v: k for k, v in relation2id.items()}
    # 构造邻接矩阵，每个关系对应一个邻接矩阵
    # Construct the list of adjacency matrix each corresponding to eeach relation. Note that this is constructed only from the train data.
    adj_list = []
    for i in range(len(relation2id)):  # 遍历关系到id的映射
        idx = np.argwhere(triplets['train'][:, 2] == i)
        adj_list.append(csc_matrix((np.ones(len(idx), dtype=np.uint8), (triplets['train'][:, 0][idx].squeeze(1), triplets['train'][:, 1][idx].squeeze(1))), shape=(len(entity2id), len(entity2id))))
    # 返回的第一个参数是一个列表，列表中的每个元素是一个关系对应的邻接矩阵,用ssp_g
    return adj_list, triplets, entity2id, relation2id, id2entity, id2relation


def save_to_file(directory, file_name, triplets, id2entity, id2relation):
    file_path = os.path.join(directory, file_name)
    with open(file_path, "w") as f:
        for s, o, r in triplets:
            f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o]]) + '\n')
