from pymongo import MongoClient
from subprocess import Popen, PIPE

class Reporter(object):
    def __init__():
        client = MongoClient('mongodb://mongodb-standalone-0.database:27017')
        self.db = client.experiments
        