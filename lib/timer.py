import torch
from contextlib import contextmanager
import datetime
import time
class Timer(object):
    def __init__(self):
        super().__init__()
        self.profile = {}
        self.timestamps = {}
        self.events = {}
        self.start = time.time()
        self.elapsed_time = 0
        self.closed = False
        self.elapsed_times = []

    @contextmanager
    def __call__(self, label):
        start = self.record_cuda(label+'_start')
        yield
        end = self.record_cuda(label+'_end')
        self.events[label] = [start, end]

    def record_cuda(self, label):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        #self.timestamps[label] = time.monotonic()
        return event

    def dump(self):
        if not self.closed:
            self.close()
        print('--------------------timeline--------------------')
        print('profile: {}'.format(self.profile))
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