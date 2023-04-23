from torch.utils.data import Dataset
import timeit
import os
import logging
import lmdb
import numpy as np
import json
import pickle
import dgl
from utils.graph_utils import ssp_multigraph_to_dgl, incidence_matrix
from utils.data_utils import process_files, save_to_file, plot_rel_dist
from .graph_sampler import *
import pdb


def generate_subgraph_datasets(params, splits=['train', 'valid'], saved_relation2id=None, max_label_value=None):

    testing = 'test' in splits  # 如果test在splits中，那么testing为True，作用是判断是否是测试集
    adj_list, triplets, entity2id, relation2id, id2entity, id2relation = process_files(params.file_paths, saved_relation2id)

    # plot_rel_dist(adj_list, os.path.join(params.main_dir, f'data/{params.dataset}/rel_dist.png'))

    data_path = os.path.join(params.main_dir, f'data/{params.dataset}/relation2id.json')  # data_path = 'utils/../data/WN18RR_v1/relation2id.json'
    if not os.path.isdir(data_path) and not testing:  # 如果不存在这个路径，就创建这个路径
        with open(data_path, 'w') as f:
            json.dump(relation2id, f)

    graphs = {}

    for split_name in splits:  # splits = ['train', 'valid']
        graphs[split_name] = {'triplets': triplets[split_name], 'max_size': params.max_links}  # 这一步是为了后面的sample_neg函数

    # Sample train and valid/test links，采样训练集和验证集/测试集的正样本
    for split_name, split in graphs.items():
        logging.info(f"Sampling negative links for {split_name}")  # 打印：为训练集采样负样本
        split['pos'], split['neg'] = sample_neg(adj_list, split['triplets'], params.num_neg_samples_per_link, max_size=split['max_size'], constrained_neg_prob=params.constrained_neg_prob)

    if testing:
        directory = os.path.join(params.main_dir, 'data/{}/'.format(params.dataset))
        save_to_file(directory, f'neg_{params.test_file}_{params.constrained_neg_prob}.txt', graphs['test']['neg'], id2entity, id2relation)

    links2subgraphs(adj_list, graphs, params, max_label_value)  # 生成子图


def get_kge_embeddings(dataset, kge_model):

    path = './experiments/kge_baselines/{}_{}'.format(kge_model, dataset)
    node_features = np.load(os.path.join(path, 'entity_embedding.npy'))
    with open(os.path.join(path, 'id2entity.json')) as json_file:
        kge_id2entity = json.load(json_file)
        kge_entity2id = {v: int(k) for k, v in kge_id2entity.items()}

    return node_features, kge_entity2id


class SubgraphDataset(Dataset):  # 子图数据集
    """Extracted, labeled, subgraph dataset -- DGL Only"""  # 从原始数据集中提取出来的子图数据集

    def __init__(self, db_path, db_name_pos, db_name_neg, raw_data_paths, included_relations=None, add_traspose_rels=False, num_neg_samples_per_link=1, use_kge_embeddings=False, dataset='', kge_model='', file_name=''):

        self.main_env = lmdb.open(db_path, readonly=True, max_dbs=3, lock=False)  # 打开lmdb数据库,lmdb是一个轻量级的数据库，用于存储键值对，类似于字典
        self.db_pos = self.main_env.open_db(db_name_pos.encode())  # 打开数据库，db_name_pos = 'pos'，意思是打开正样本数据库
        self.db_neg = self.main_env.open_db(db_name_neg.encode())  # 打开负样本数据库
        self.node_features, self.kge_entity2id = get_kge_embeddings(dataset, kge_model) if use_kge_embeddings else (None, None)  # 如果使用kge的嵌入，就获取kge的嵌入
        self.num_neg_samples_per_link = num_neg_samples_per_link  # 负样本采样数  # num_neg_samples_per_link = 1
        self.file_name = file_name  # file_name = 'train'
        # `ssp_graph`：一个字典，键为关系名称，值为DGL图对象。每个DGL图对象表示一个封闭子图。
        ssp_graph, __, __, __, id2entity, id2relation = process_files(raw_data_paths, included_relations)  # 从原始数据集中提取出子图
        self.num_rels = len(ssp_graph)  # 子图的关系数量, 这里是9

        # Add transpose matrices to handle both directions of relations.添加转置矩阵以处理关系的两个方向
        if add_traspose_rels:
            ssp_graph_t = [adj.T for adj in ssp_graph]
            ssp_graph += ssp_graph_t
        # 计算了添加反向关系和自连接后的有效关系数量
        # the effective number of relations after adding symmetric adjacency matrices and/or self connections
        self.aug_num_rels = len(ssp_graph)  # aug是augment的缩写，增强的意思
        self.graph = ssp_multigraph_to_dgl(ssp_graph)  # 将子图转换为DGL图
        self.ssp_graph = ssp_graph  # 子图
        self.id2entity = id2entity  # 实体id到实体名称的映射
        self.id2relation = id2relation

        self.max_n_label = np.array([0, 0])  # 存储最大的实体标签和关系标签
        with self.main_env.begin() as txn:  # 使用`main_env`打开一个lmdb数据库，并使用`txn`事务对象读取以下键值对：
            self.max_n_label[0] = int.from_bytes(txn.get('max_n_label_sub'.encode()), byteorder='little')  # 主体实体的最大标签
            self.max_n_label[1] = int.from_bytes(txn.get('max_n_label_obj'.encode()), byteorder='little')  # 宾语实体的最大标签

            self.avg_subgraph_size = struct.unpack('f', txn.get('avg_subgraph_size'.encode()))  # 子图平均大小
            self.min_subgraph_size = struct.unpack('f', txn.get('min_subgraph_size'.encode()))  # 子图最小大小
            self.max_subgraph_size = struct.unpack('f', txn.get('max_subgraph_size'.encode()))  # 子图最大大小
            self.std_subgraph_size = struct.unpack('f', txn.get('std_subgraph_size'.encode()))  # 子图大小的标准差

            self.avg_enc_ratio = struct.unpack('f', txn.get('avg_enc_ratio'.encode()))  # 子图平均编码率
            self.min_enc_ratio = struct.unpack('f', txn.get('min_enc_ratio'.encode()))  # 子图最小编码率
            self.max_enc_ratio = struct.unpack('f', txn.get('max_enc_ratio'.encode()))  # 子图最大编码率
            self.std_enc_ratio = struct.unpack('f', txn.get('std_enc_ratio'.encode()))  # 子图编码率的标准差

            self.avg_num_pruned_nodes = struct.unpack('f', txn.get('avg_num_pruned_nodes'.encode()))  # 子图平均修剪节点数
            self.min_num_pruned_nodes = struct.unpack('f', txn.get('min_num_pruned_nodes'.encode()))  # 子图最小修剪节点数
            self.max_num_pruned_nodes = struct.unpack('f', txn.get('max_num_pruned_nodes'.encode()))  # 子图最大修剪节点数
            self.std_num_pruned_nodes = struct.unpack('f', txn.get('std_num_pruned_nodes'.encode()))  # 子图修剪节点数的标准差

        logging.info(f"Max distance from sub : {self.max_n_label[0]}, Max distance from obj : {self.max_n_label[1]}")  # 打印最大的实体标签和关系标签

        # logging.info('=====================')
        # logging.info(f"Subgraph size stats: \n Avg size {self.avg_subgraph_size}, \n Min size {self.min_subgraph_size}, \n Max size {self.max_subgraph_size}, \n Std {self.std_subgraph_size}")

        # logging.info('=====================')
        # logging.info(f"Enclosed nodes ratio stats: \n Avg size {self.avg_enc_ratio}, \n Min size {self.min_enc_ratio}, \n Max size {self.max_enc_ratio}, \n Std {self.std_enc_ratio}")

        # logging.info('=====================')
        # logging.info(f"# of pruned nodes stats: \n Avg size {self.avg_num_pruned_nodes}, \n Min size {self.min_num_pruned_nodes}, \n Max size {self.max_num_pruned_nodes}, \n Std {self.std_num_pruned_nodes}")

        with self.main_env.begin(db=self.db_pos) as txn:  # 使用`main_env`打开一个lmdb数据库，并使用`txn`事务对象读取以下键值对：
            self.num_graphs_pos = int.from_bytes(txn.get('num_graphs'.encode()), byteorder='little')  # 正样本的数量
        with self.main_env.begin(db=self.db_neg) as txn:
            self.num_graphs_neg = int.from_bytes(txn.get('num_graphs'.encode()), byteorder='little')  # 负样本的数量

        self.__getitem__(0)  # 读取第一个样本

    def __getitem__(self, index):
        with self.main_env.begin(db=self.db_pos) as txn:  # 读取正样本
            str_id = '{:08}'.format(index).encode('ascii')  # 将索引转换为8位的字符串
            nodes_pos, r_label_pos, g_label_pos, n_labels_pos = deserialize(txn.get(str_id)).values()  # 读取正样本
            subgraph_pos = self._prepare_subgraphs(nodes_pos, r_label_pos, n_labels_pos)  # 准备正样本，返回一个字典
        subgraphs_neg = []  # 负样本
        r_labels_neg = []  # 负样本的关系标签
        g_labels_neg = []  # 负样本的图标签
        with self.main_env.begin(db=self.db_neg) as txn:  # 读取负样本
            for i in range(self.num_neg_samples_per_link):  # 读取每个正样本对应的负样本
                str_id = '{:08}'.format(index + i * (self.num_graphs_pos)).encode('ascii')  # 将索引转换为8位的字符串
                nodes_neg, r_label_neg, g_label_neg, n_labels_neg = deserialize(txn.get(str_id)).values()  # 读取负样本
                subgraphs_neg.append(self._prepare_subgraphs(nodes_neg, r_label_neg, n_labels_neg))  # 准备负样本
                r_labels_neg.append(r_label_neg)  # 负样本的关系标签
                g_labels_neg.append(g_label_neg)  # 负样本的图标签

        return subgraph_pos, g_label_pos, r_label_pos, subgraphs_neg, g_labels_neg, r_labels_neg

    def __len__(self):
        return self.num_graphs_pos

    def _prepare_subgraphs(self, nodes, r_label, n_labels):
        subgraph = dgl.DGLGraph(self.graph.subgraph(nodes))  # 从原图中提取子图，返回一个DGLGraph对象
        subgraph.edata['type'] = self.graph.edata['type'][self.graph.subgraph(nodes).parent_eid]
        subgraph.edata['label'] = torch.tensor(r_label * np.ones(subgraph.edata['type'].shape), dtype=torch.long)

        edges_btw_roots = subgraph.edge_id(0, 1)
        rel_link = np.nonzero(subgraph.edata['type'][edges_btw_roots] == r_label)
        if rel_link.squeeze().nelement() == 0:
            subgraph.add_edge(0, 1)
            subgraph.edata['type'][-1] = torch.tensor(r_label).type(torch.LongTensor)
            subgraph.edata['label'][-1] = torch.tensor(r_label).type(torch.LongTensor)

        # map the id read by GraIL to the entity IDs as registered by the KGE embeddings
        kge_nodes = [self.kge_entity2id[self.id2entity[n]] for n in nodes] if self.kge_entity2id else None
        n_feats = self.node_features[kge_nodes] if self.node_features is not None else None
        subgraph = self._prepare_features_new(subgraph, n_labels, n_feats)

        return subgraph

    def _prepare_features(self, subgraph, n_labels, n_feats=None):
        # One hot encode the node label feature and concat to n_featsure
        n_nodes = subgraph.number_of_nodes()
        label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1))
        label_feats[np.arange(n_nodes), n_labels] = 1
        label_feats[np.arange(n_nodes), self.max_n_label[0] + 1 + n_labels[:, 1]] = 1
        n_feats = np.concatenate((label_feats, n_feats), axis=1) if n_feats else label_feats
        subgraph.ndata['feat'] = torch.FloatTensor(n_feats)
        self.n_feat_dim = n_feats.shape[1]  # Find cleaner way to do this -- i.e. set the n_feat_dim
        return subgraph

    def _prepare_features_new(self, subgraph, n_labels, n_feats=None):
        # One hot encode the node label feature and concat to n_featsure
        n_nodes = subgraph.number_of_nodes()
        label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1 + self.max_n_label[1] + 1))
        label_feats[np.arange(n_nodes), n_labels[:, 0]] = 1
        label_feats[np.arange(n_nodes), self.max_n_label[0] + 1 + n_labels[:, 1]] = 1
        # label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1 + self.max_n_label[1] + 1))
        # label_feats[np.arange(n_nodes), 0] = 1
        # label_feats[np.arange(n_nodes), self.max_n_label[0] + 1] = 1
        n_feats = np.concatenate((label_feats, n_feats), axis=1) if n_feats is not None else label_feats
        subgraph.ndata['feat'] = torch.FloatTensor(n_feats)

        head_id = np.argwhere([label[0] == 0 and label[1] == 1 for label in n_labels])
        tail_id = np.argwhere([label[0] == 1 and label[1] == 0 for label in n_labels])
        n_ids = np.zeros(n_nodes)
        n_ids[head_id] = 1  # head
        n_ids[tail_id] = 2  # tail
        subgraph.ndata['id'] = torch.FloatTensor(n_ids)

        self.n_feat_dim = n_feats.shape[1]  # Find cleaner way to do this -- i.e. set the n_feat_dim
        return subgraph
