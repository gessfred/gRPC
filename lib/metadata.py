from pymongo import MongoClient
from subprocess import Popen, PIPE, check_output
import uuid
import torch
import torch.distributed as dist
import os

"""
This is a internal tool not meant to be used in production :)
"""
class Metadata():
    def __init__(self, model, dataset, description, args, use_cuda):
        with open(os.environ['MONGO_USR']) as usr:
            with open(os.environ['MONGO_PWD']) as pwd:
                client = MongoClient('mongodb://iccluster095.iccluster.epfl.ch:32396', username=usr.read(), password=pwd.read())
                self.db = client['admin']['benchmarks']
        path = '/pyparsa/.git'
        """

            'branch': check_output(['git', '--git-dir', path, 'branch']).decode('utf-8').split(' ')[1].split('\n')[0],
            'commit': check_output(['git', '--git-dir', path, 'show', '--summary']).decode("utf-8").split(' ')[1].split('\n')[0],
        """
        self.data = {
            '_id': uuid.uuid4().hex,
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