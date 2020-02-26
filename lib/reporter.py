from pymongo import MongoClient

class Reporter(object):
    def __init__():
        client = MongoClient('mongodb://mongodb-standalone-0.database:27017')
        self.db = client.experiments
    