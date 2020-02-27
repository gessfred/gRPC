from pymongo import MongoClient
from subprocess import Popen, PIPE, check_output
import uuid
import torch
import torch.distributed as dist
import os

"""
This is a internal tool not meant to be used in production :)
"""
class Timeline():
    def __init__(self, model, dataset, description, args, use_cuda):
        print('username:{} password{}'.format(os.environ['mdb-usr'], os.environ['mdb-pwd']))
        client = MongoClient('mongodb://10.98.200.71:27017', username=os.environ['mdb-usr'], password=os.environ['mdb-pwd'])
        self.db = client['admin']['benchmarks']
        path = '/jet/.git'
        self.data = {
            '_id': uuid.uuid4().hex,
            'branch': check_output(['git', '--git-dir', path, 'branch']).decode('utf-8').split(' ')[1].split('\n')[0],
            'commit': check_output(['git', '--git-dir', path, 'show', '--summary']).decode("utf-8").split(' ')[1].split('\n')[0],
            'model': model,
            'dataset': dataset,
            'description': description,
            'backend': dist.get_backend(),
            'worldSize': dist.get_world_size(),
            'rank': dist.get_rank(),
            'dtype': args.dtype,
            'cuda': use_cuda,
            'learningRate': args.lr,
            'gamma': args.gamma
        }
        # Create uuid

        # Create start timestamp
        # Get branch, commit from git
        #self.branch = check_output(['git', '--git-dir', '/jet/.git', ])
        # Get passed description, dataset, model 
        # Get number of nodes, rank from dist
        # Get dtype, use_cuda, learning_rate, gamma from args
    def append(self, data):
        self.data = {**self.data, **data}
    
    def collect(self):
        self.db.insert_one(self.data)