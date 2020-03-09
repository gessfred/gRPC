import torch
from contextlib import contextmanager
import torch.distributed as dist
import datetime
import time
from pymongo import MongoClient
import uuid
import os

class Timer(object):
    def __init__(self, name):
        super().__init__()
        self.clock = time.perf_counter()
        self.name = name
        self.timestamps = {} # for timeline synchronisation
        self.events = {}
        self.elapsed_time = 0
        self.closed = False
        self.elapsed_times = []
        self.events_durations = {}
        self.start = time.time()

    @contextmanager
    def __call__(self, label):
        start = self.record(label+'_start')
        yield
        end = self.record(label+'_end')
        self.events[label] = [start, end]

    def record(self, label):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self.timestamps[label] = time.perf_counter()
        return event

    def dump(self):
        if not self.closed:
            self.close()
        print('--------------------timeline--------------------')
        print('events: {}'.format(self.events))
        print('timeline: {}'.format(self.timestamps))
        print('elapsed_time: {}'.format(self.elapsed_time))
        print('------------------------------------------------')

    def wait(self, event, handle):
        pass
        
    def track(self, handle):
        pass

    def close(self):
        torch.cuda.synchronize()
        self.closed = True
        self.elapsed_time = time.time() - self.start
        self.events_durations = {k: v[0].elapsed_time(v[1]) for k, v in self.events.items()}

    def upload(self):
        path = '/pyparsa/.git'
        self.close()
        with open(os.environ['MONGO_USR']) as usr:
            with open(os.environ['MONGO_PWD']) as pwd:
                client = MongoClient('mongodb://iccluster095.iccluster.epfl.ch:32396', username=usr.read(), password=pwd.read())
                data = {
                    '_id': uuid.uuid4().hex,
                    'elapsed_time': self.elapsed_time, 
                    'clock': self.clock,
                    'time_stamps': self.timestamps,
                    'events': self.events_durations,
                    'name': self.name,
                    'world_size': dist.get_world_size(),
                    'rank': dist.get_rank(),
                    'backend': dist.get_backend(),
                    'branch': check_output(['git', '--git-dir', path, 'branch']).decode('utf-8').split(' ')[1].split('\n')[0],
                    'commit': check_output(['git', '--git-dir', path, 'show', '--summary']).decode("utf-8").split(' ')[1].split('\n')[0],
                }
                client['admin']['microbenchmarks'].insert_one(data)
