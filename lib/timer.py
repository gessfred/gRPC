import torch
from contextlib import contextmanager
import torch.distributed as dist
import datetime
import time
from pymongo import MongoClient
import uuid
import os
from subprocess import Popen, PIPE, check_output

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
        self.ready_events = {}
        self.epoch_idx = 0
        self.ts = {}
        self.stack = []
        self.ec = {}

    @contextmanager
    def __call__(self, label, epoch=0):
        pass

    def record(self, label):
        pass

    def dump(self):
        if not self.closed:
            self.close()
        print('--------------------timer--------------------')
        print('events: {}'.format(self.ready_events))
        print('elapsed_time: {}'.format(self.elapsed_time))
        print('------------------------------------------------')

    def summary(arg):
        return 'no summary'

    def wait(self, event, handle):
        pass
        
    def track(self, handle):
        pass

    def close(self):
        self.aggregate()
        self.close_epoch()
        self.closed = True
        self.elapsed_time = time.time() - self.start

    def upload(self, conf):
        path = '/pyparsa/.git'
        self.close()
        print('uploading...')
        git = {
            'branch': os.environ['VCS_BRANCH'],
            'commit': os.environ['VCS_COMMIT'],
        }
        client = MongoClient('mongodb://178.128.35.255:27017', username=os.environ['MONGO_USR'], password=os.environ['MONGO_PWD'])
        data = {
            '_id': uuid.uuid4().hex,#unique __record__ id
            'uuid': os.environ['UUID'],#unique "deployment id"
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
            'n_sub_process': conf.n_sub_process,
            'git': git,
            'time_stamps': self.ts,
        }
        client['coltrain']['benchmarking'].insert_one(data)
    def aggregate(self):
        torch.cuda.synchronize()
        for rec in self.events:
            label = rec['label']
            if label not in self.ready_events:
                self.ready_events[label] = 0
            self.ready_events[label] += rec['start'].elapsed_time(rec['end'])
        del self.events
        self.events = []

    def close_epoch(self):
        pass
        """self.epoch_idx += 1
        if int(self.epoch_idx) == 10:
            self.ts[str(self.epoch_idx)] = self.timestamps
            del self.timestamps
            self.timestamps = []"""

#class 

class CUDATimer(TimerBase):

    def record(self):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    @contextmanager
    def __call__(self, label, epoch=0):
        start = self.record()
        self.stack.append(label)
        yield
        end = self.record()
        id = '/'.join(self.stack)
        self.events += [{'label': id, 'start': start, 'end': end}]
        self.stack.pop()
        #self.ec[id] = self.ec.get(id, 0)
    def wait(self, event, handle):
        pass
        
    def track(self, handle):
        pass