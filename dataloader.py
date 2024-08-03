import math
import random
import torch


class DataIterator:

    def __init__(self, data_file, batch_size, PAD_TOKEN):
        super(DataIterator, self).__init__()
        self.batch_size = batch_size
        self.data_lis = self.read_file(data_file)
        self.data_num = len(self.data_lis)
        self.indices = range(self.data_num)
        self.num_batches = math.ceil(self.data_num / self.batch_size)
        self.idx = 0
        self.pad_token = PAD_TOKEN
        self.reset()

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        return self

    def __getitem__(self, idx):
        index = self.indices[idx: idx + self.batch_size]
        data = torch.tensor([self.data_lis[i] for i in index])
        target = torch.cat([data[:, 1:], torch.full(
            (len(index), 1), self.pad_token, dtype=torch.int64)], dim=1)

        self.idx += self.batch_size
        return data, target

    def get_data_num(self):
        return self.data_num

    def __next__(self):
        if self.idx >= self.data_num:
            raise StopIteration
        index = self.indices[self.idx: self.idx + self.batch_size]
        data = torch.tensor([self.data_lis[i] for i in index])
        target = torch.cat([data[:, 1:], torch.full(
            (len(index), 1), self.pad_token, dtype=torch.int64)], dim=1)

        self.idx += self.batch_size
        return data, target

    def reset(self):
        self.idx = 0
        random.shuffle(self.data_lis)

    def read_file(self, data_file):
        with open(data_file, 'r') as file:
            lines = file.readlines()
        lis = [[int(float(s)) for s in list(line.strip().split())]
               for line in lines]
        return lis
