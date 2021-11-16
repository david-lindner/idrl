import random

import numpy as np


class SampleBuffer:
    def __init__(self, observation_type):
        self.buffer = {}
        self.labels = {}
        self.cur_size = 0
        self.observation_type = observation_type

    def __len__(self):
        return self.cur_size

    def add_single(self, item, label):
        self.buffer[self.cur_size] = item
        self.labels[self.cur_size] = label
        self.cur_size += 1

    def add(self, items, labels):
        assert len(items) == len(labels)
        idx = 0
        while idx < len(items):
            self.add_single(items[idx], labels[idx])
            idx += 1
        assert len(self.buffer) == self.cur_size

    def get_batch(self, batch_size):
        assert batch_size < len(self)
        idx = np.random.choice(np.arange(self.cur_size), batch_size)
        return [self.buffer[idx] for idx in idxs], [self.labels[idx] for idx in idxs]

    def get_all_batches(self, batch_size):
        assert batch_size < len(self)
        idx = np.arange(self.cur_size)
        np.random.shuffle(idx)
        buffer_shuffled = np.array([self.buffer[i] for i in idx])
        labels_shuffled = np.array([self.labels[i] for i in idx])
        N = self.cur_size // batch_size
        L = batch_size * N
        return np.split(buffer_shuffled[:L], N), np.split(labels_shuffled[:L], N)

    def get_all(self):
        idx = np.arange(self.cur_size)
        return [self.buffer[i] for i in idx], [self.labels[i] for i in idx]

    def bootstrap(self):
        idx = np.random.randint(0, self.cur_size, size=(self.cur_size,))
        buffer = SampleBuffer(self.observation_type)
        buffer.buffer = dict([(i, self.buffer[idx[i]]) for i in range(self.cur_size)])
        buffer.labels = dict([(i, self.labels[idx[i]]) for i in range(self.cur_size)])
        buffer.cur_size = self.cur_size
        return buffer

    def copy(self):
        other = SampleBuffer(self.observation_type)
        items, labels = self.get_all()
        other.add(items, labels)
        return other
