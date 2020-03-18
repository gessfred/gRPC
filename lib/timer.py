import torch
from contextlib import contextmanager
import torch.distributed as dist
import datetime
import time
from pymongo import MongoClient
import uuid
import os
from subprocess import Popen, PIPE, check_output

class TimerBase(object):
    def __init__(self, name):
        super().__init__()
        self.clock = time.perf_counter()
        self.name = name
        self.timestamps = [] # for timeline synchronisation
        self.elapsed_time = 0
        self.closed = False
        self.elapsed_times = []
        self.events_durations = {}
        self.start = time.time()
        self.tracking = []
        self.events = []
        self.ready_events = []

    @contextmanager
    def __call__(self, label, epoch=0):
        pass

    def record(self, label):
        pass

    def dump(self):
        if not self.closed:
            self.close()
        print('--------------------timeline--------------------')
        print('events: {}'.format(self.events))
        print('timeline: {}'.format(self.timestamps))
        print('elapsed_time: {}'.format(self.elapsed_time))
        print('------------------------------------------------')

    def summary(arg):
        return 'no summary'

    def wait(self, event, handle):
        pass
        
    def track(self, handle):
        pass

    def close(self):
        self.epoch()
        self.closed = True
        self.elapsed_time = time.time() - self.start

    def epoch(self):
        torch.cuda.synchronize()
        for rec in self.events:
            label = rec['label']
            if label not in self.ready_events:
                self.ready_events[label] = 0
            self.ready_events[label] += rec['start'].elapsed_time(rec['end']
        
        self.events = []

    def upload(self, conf):
        path = '/pyparsa/.git'
        self.close()
        print('uploading...')
        with open(os.environ['MONGO_USR']) as usr:
            with open(os.environ['MONGO_PWD']) as pwd:
                client = MongoClient('mongodb://iccluster095.iccluster.epfl.ch:32396', username=usr.read(), password=pwd.read())
                data = {
                    '_id': uuid.uuid4().hex,
                    'elapsed_time': self.elapsed_time, 
                    'clock': self.clock,
                    'events': self.ready_events,
                    'name': self.name,
                    'world_size': dist.get_world_size(),
                    'rank': dist.get_rank(),
                    'backend': dist.get_backend(),
                    'arch': conf.arch,
                    'optimizer': conf.optimizer,
                    'lr': conf.lr,
                    'data': conf.data,
                    'batch_size': conf.batch_size,
                    'num_epochs': conf.num_epochs,
                    'aggregator': conf.aggregator,
                    'tracking': self.tracking,
                }
                client['admin']['eval'].insert_one(data)
"""

                    'branch': check_output(['git', '--git-dir', path, 'branch']).decode('utf-8').split(' ')[1].split('\n')[0],
                    'commit': check_output(['git', '--git-dir', path, 'show', '--summary']).decode("utf-8").split(' ')[1].split('\n')[0],
"""

class CUDATimer(TimerBase):

    def record(self, label):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        #self.timestamps += [{'label': label, 'stamp': time.perf_counter()}]
        return event

    @contextmanager
    def __call__(self, label, epoch=0):
        start = self.record(label+'_start')
        yield
        end = self.record(label+'_end')
        self.events += [{'label': label, 'start': start, 'end': end}]

    def wait(self, event, handle):
        pass
        
    def track(self, handle):
        pass

