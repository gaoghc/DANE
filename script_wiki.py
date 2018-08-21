import numpy as np
import linecache
from Dataset.dataset import Dataset
from Model.model import Model
from Trainer.trainer import Trainer
from Trainer.pretrainer import PreTrainer
from Utils import gpu_info
import os
import random


if __name__=='__main__':

    gpus_to_use, free_memory = gpu_info.get_free_gpu()
    print gpus_to_use, free_memory
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus_to_use

    random.seed(9001)

    dataset_config = {'feature_file': './Database/wiki/features.txt',
                      'graph_file': './Database/wiki/edges.txt',
                      'walks_file': './Database/wiki/walks.txt',
                      'label_file': './Database/wiki/group.txt'}
    graph = Dataset(dataset_config)

    pretrain_config = {
        'net_shape': [200, 100],
        'att_shape': [500, 100],
        'net_input_dim': graph.num_nodes,
        'att_input_dim': graph.num_feas,
        'pretrain_params_path': './Log/wiki/pretrain_params.pkl'}

    model_config = {
        'net_shape': [200, 100],
        'att_shape': [500, 100],
        'net_input_dim': graph.num_nodes,
        'att_input_dim': graph.num_feas,
        'is_init': True,
        'pretrain_params_path': './Log/wiki/pretrain_params.pkl'
    }

    trainer_config = {
        'net_shape': [200, 100],
        'att_shape': [500, 100],
        'net_input_dim': graph.num_nodes,
        'att_input_dim': graph.num_feas,
        'drop_prob': 0.2,
        'learning_rate': 1e-5,
        'batch_size': 100,
        'num_epochs': 500,
        'beta': 1,
        'alpha': 0.05,
        'gamma': 0.05,
        'model_path': './Log/wiki/wiki_model.pkl',
    }

    pretrainer = PreTrainer(pretrain_config)
    pretrainer.pretrain(graph.X, 'net')
    pretrainer.pretrain(graph.Z, 'att')

    model = Model(model_config)
    trainer = Trainer(model, trainer_config)
    trainer.train(graph)
    trainer.infer(graph)

