from pymongo import MongoClient
from subprocess import Popen, PIPE, check_output

class Reporter():
    def __init__(self):
        client = MongoClient('mongodb://mongodb-standalone-0.database:27017')
        self.db = client.experiments
        # Create uuid

        # Create start timestamp
        # Get branch, commit from git
        #self.branch = check_output(['git', '--git-dir', '/jet/.git', ])
        # Get passed description, dataset, model 
        # Get number of nodes, rank from dist
        # Get dtype, use_cuda, learning_rate, gamma from args