import os
import argparse
import logging
import torch
from scipy.sparse import SparseEfficiencyWarning

from subgraph_extraction.datasets import SubgraphDataset, generate_subgraph_datasets
from utils.initialization_utils import initialize_experiment, initialize_model
from utils.graph_utils import collate_dgl, move_batch_to_device_dgl

from model.dgl.graph_classifier import GraphClassifier as dgl_model

from managers.evaluator import Evaluator
from managers.trainer import Trainer

from warnings import simplefilter


def main(params):
    simplefilter(action='ignore', category=UserWarning)  # 忽略警告
    simplefilter(action='ignore', category=SparseEfficiencyWarning)  # 忽略警告，这个警告是因为稀疏矩阵的问题

    params.db_path = os.path.join(params.main_dir, f'data/{params.dataset}/subgraphs_en_{params.enclosing_sub_graph}_neg_{params.num_neg_samples_per_link}_hop_{params.hop}')
    # db_path = 'utils/../data/WN18RR_v1/subgraphs_en_True_neg_1_hop_3'
    if not os.path.isdir(params.db_path):  # 如果不存在这个路径，就创建这个路径
        generate_subgraph_datasets(params)  # 生成子图数据集
    # train = SubgraphDataset和valid = SubgraphDataset是用来生成训练集和验证集的
    train = SubgraphDataset(params.db_path, 'train_pos', 'train_neg', params.file_paths,
                            add_traspose_rels=params.add_traspose_rels,  # 是否添加反向关系
                            num_neg_samples_per_link=params.num_neg_samples_per_link,  # 每个关系的负样本数
                            use_kge_embeddings=params.use_kge_embeddings, dataset=params.dataset,  # 是否使用kge嵌入
                            kge_model=params.kge_model, file_name=params.train_file)  # kge模型，训练文件
    valid = SubgraphDataset(params.db_path, 'valid_pos', 'valid_neg', params.file_paths,
                            add_traspose_rels=params.add_traspose_rels,
                            num_neg_samples_per_link=params.num_neg_samples_per_link,
                            use_kge_embeddings=params.use_kge_embeddings, dataset=params.dataset,
                            kge_model=params.kge_model, file_name=params.valid_file)

    params.num_rels = train.num_rels  # 关系数 9
    params.aug_num_rels = train.aug_num_rels  # 9
    params.inp_dim = train.n_feat_dim  # 8
    # 记录最大的标签值，用于在测试集上生成标签
    # Log the max label value to save it in the model. This will be used to cap the labels generated on test set.
    params.max_label_value = train.max_n_label  # 训练集中最大的标签值

    graph_classifier = initialize_model(params, dgl_model, params.load_model)  # 初始化模型，这里是dgl模型

    logging.info(f"Device: {params.device}")
    logging.info(f"Input dim : {params.inp_dim}, # Relations : {params.num_rels}, # Augmented relations : {params.aug_num_rels}")

    valid_evaluator = Evaluator(params, graph_classifier, valid)  # 初始化验证集评估器

    trainer = Trainer(params, graph_classifier, train, valid_evaluator)  # 初始化训练器，这里是dgl模型

    logging.info('Starting training with full batch...')  # 开始训练

    trainer.train()


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)  # 设置日志级别，这里设置为INFO级别，即只有INFO级别的日志才会被输出

    parser = argparse.ArgumentParser(description='TransE model')  # 创建一个ArgumentParser对象，用来解析命令行参数
    # parser意思是解析器，用来解析命令行参数的，ArgumentParser是解析器的一个类，用来解析命令行参数的
    # Experiment setup params，实验设置参数
    parser.add_argument("--experiment_name", "-e", type=str, default="default",  # 保存模型的文件夹
                        help="A folder with this name would be created to dump saved models and log files")
    parser.add_argument("--dataset", "-d", type=str,  # 数据集
                        help="Dataset string")
    parser.add_argument("--gpu", type=int, default=0,  # gpu
                        help="Which GPU to use?")
    parser.add_argument('--disable_cuda', action='store_true',  # 是否使用gpu
                        help='Disable CUDA')
    parser.add_argument('--load_model', action='store_true',  # 是否加载模型
                        help='Load existing model?')
    parser.add_argument("--train_file", "-tf", type=str, default="train",
                        help="Name of file containing training triplets")  # 训练集文件名
    parser.add_argument("--valid_file", "-vf", type=str, default="valid",
                        help="Name of file containing validation triplets")  # 验证集文件名

    # Training regime params，训练参数
    parser.add_argument("--num_epochs", "-ne", type=int, default=100,
                        help="Learning rate of the optimizer")  # 训练轮数
    parser.add_argument("--eval_every", type=int, default=3,
                        help="Interval of epochs to evaluate the model?")  # 每隔多少轮评估一次模型
    parser.add_argument("--eval_every_iter", type=int, default=455,
                        help="Interval of iterations to evaluate the model?")  # 每隔多少次迭代评估一次模型
    parser.add_argument("--save_every", type=int, default=10,
                        help="Interval of epochs to save a checkpoint of the model?")  # 每隔多少轮保存一次模型
    parser.add_argument("--early_stop", type=int, default=100,
                        help="Early stopping patience")  # 提前停止的轮数，如果超过这个轮数，模型的效果没有提升，就停止训练
    parser.add_argument("--optimizer", type=str, default="Adam",
                        help="Which optimizer to use?")  # 优化器
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate of the optimizer")  # 学习率
    parser.add_argument("--clip", type=int, default=1000,
                        help="Maximum gradient norm allowed")  # 梯度裁剪，防止梯度爆炸
    parser.add_argument("--l2", type=float, default=5e-4,
                        help="Regularization constant for GNN weights")  # GNN权重的正则化系数，防止过拟合
    parser.add_argument("--margin", type=float, default=10,  # 位于正负样本之间的距离，用于计算损失
                        help="The margin between positive and negative samples in the max-margin loss")

    # Data processing pipeline params，数据处理参数
    parser.add_argument("--max_links", type=int, default=1000000,
                        help="Set maximum number of train links (to fit into memory)")  # 最大训练样本数
    parser.add_argument("--hop", type=int, default=3,
                        help="Enclosing subgraph hop number")  # 子图的hop数
    parser.add_argument("--max_nodes_per_hop", "-max_h", type=int, default=None,
                        help="if > 0, upper bound the # nodes per hop by subsampling")  # 每个hop的最大节点数
    parser.add_argument("--use_kge_embeddings", "-kge", type=bool, default=False,
                        help='whether to use pretrained KGE embeddings')  # 是否使用预训练的KGE嵌入
    parser.add_argument("--kge_model", type=str, default="TransE",  # 默认使用TransE模型
                        help="Which KGE model to load entity embeddings from")  # KGE模型
    parser.add_argument('--model_type', '-m', type=str, choices=['ssp', 'dgl'], default='dgl',
                        help='what format to store subgraphs in for model')  # 子图的存储格式
    parser.add_argument('--constrained_neg_prob', '-cn', type=float, default=0.0,  # 采样负样本时，是否采样约束的头尾节点
                        help='with what probability to sample constrained heads/tails while neg sampling')
    parser.add_argument("--batch_size", type=int, default=16,  # batch大小，每次训练的样本数
                        help="Batch size")
    parser.add_argument("--num_neg_samples_per_link", '-neg', type=int, default=1,  # 每个正样本采样多少个负样本
                        help="Number of negative examples to sample per positive link")
    parser.add_argument("--num_workers", type=int, default=8,  # 多进程加载数据
                        help="Number of dataloading processes")
    parser.add_argument('--add_traspose_rels', '-tr', type=bool, default=False,  # 是否添加反向关系
                        help='whether to append adj matrix list with symmetric relations')
    parser.add_argument('--enclosing_sub_graph', '-en', type=bool, default=True,
                        help='whether to only consider enclosing subgraph')  # 是否只考虑封闭子图

    # Model params，模型参数
    parser.add_argument("--rel_emb_dim", "-r_dim", type=int, default=32,
                        help="Relation embedding size")  # 关系嵌入的维度
    parser.add_argument("--attn_rel_emb_dim", "-ar_dim", type=int, default=32,
                        help="Relation embedding size for attention")  # 关系嵌入的维度，用于注意力
    parser.add_argument("--emb_dim", "-dim", type=int, default=32,
                        help="Entity embedding size")
    parser.add_argument("--num_gcn_layers", "-l", type=int, default=3,
                        help="Number of GCN layers")  # GCN层数
    parser.add_argument("--num_bases", "-b", type=int, default=4,
                        help="Number of basis functions to use for GCN weights")  # GCN权重的基函数数，用于加速计算
    parser.add_argument("--dropout", type=float, default=0,
                        help="Dropout rate in GNN layers")  # GNN层的dropout，防止过拟合
    parser.add_argument("--edge_dropout", type=float, default=0.5,
                        help="Dropout rate in edges of the subgraphs")  # 子图边的dropout，防止过拟合
    parser.add_argument('--gnn_agg_type', '-a', type=str, choices=['sum', 'mlp', 'gru'], default='sum',
                        help='what type of aggregation to do in gnn msg passing')  # GNN消息传递的聚合方式
    parser.add_argument('--add_ht_emb', '-ht', type=bool, default=True,  # 是否将头尾节点的嵌入与子图的嵌入拼接
                        help='whether to concatenate head/tail embedding with pooled graph representation')
    parser.add_argument('--has_attn', '-attn', type=bool, default=True,
                        help='whether to have attn in model or not')  # 是否使用注意力机制

    params = parser.parse_args()  # 解析参数
    initialize_experiment(params, __file__)  # 初始化实验

    params.file_paths = {  # 数据集路径，包括训练集和验证集
        'train': os.path.join(params.main_dir, 'data/{}/{}.txt'.format(params.dataset, params.train_file)),
        'valid': os.path.join(params.main_dir, 'data/{}/{}.txt'.format(params.dataset, params.valid_file))
    }

    if not params.disable_cuda and torch.cuda.is_available():  # 如果有GPU，使用GPU
        params.device = torch.device('cuda:%d' % params.gpu)  # 指定GPU，如果有多个GPU，可以指定使用哪个GPU
    else:
        params.device = torch.device('cpu')  # 没有GPU，使用CPU

    params.collate_fn = collate_dgl  # 数据加载函数，用于将多个样本组合成一个batch
    params.move_batch_to_device = move_batch_to_device_dgl  # 将batch数据移动到GPU上

    main(params)  # 运行主函数
