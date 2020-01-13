import torch
import torch.utils.data as data
import numpy as np


def read_problem(file_name):
    """
    Returns [y, x] -> y: list of label, x: list of dictionary of data instances
    """
    prob_y = []
    prob_x = []
    row_ptr = [0]
    col_idx = []
    for i, line in enumerate(open(file_name)):
        line = line.split(None, 1)
        if len(line) == 1:
            line += ['']
        label, features = line
        prob_y += [float(label)]
        xi = []
        for e in features.split():
            ind, val = e.split(":")
            xi += [float(val)]
        prob_x += [xi]
    return (prob_y, prob_x)

class MLDataset(data.Dataset):

    def __init__(self, root, train=True):
        super().__init__()
        self.train = train
        targets, datas = read_problem(root)
        length = len(datas)
        ratio = 0.2
        k = int(float(length) * ratio)
        if self.train:
            self.data = datas[:-k]
            self.target = targets[:-k]
        else:
            self.data = datas[-k:]
            self.target = targets[-k:]
        self.data = np.array(self.data, dtype='float32')
        self.target = np.array(self.target, dtype='float32')
        self.target = np.reshape(self.target, (self.target.size, 1))
        self.data = torch.from_numpy(self.data)
        self.target = torch.from_numpy(self.target)

    def __getitem__(self, index):
       return self.data[index], self.target[index]

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        head = "Dataset " + self.__class__.__name__
        lines = [head] + [" " * 4 + str(self.data)] + [" " * 4 + str(self.target)]
        return "\n".join(lines)

def abalone(rootPath, train):
    path = rootPath + "/abalone.txt"
    return MLDataset(path, train)

def bodyfat(rootPath, train):
    path = rootPath + "/bodyfat.txt"
    return MLDataset(path, train)

def housing(rootPath, train):
    path = rootPath + "/housing.txt"
    return MLDataset(path, train)

if __name__ == '__main__':
    path = "../datasets"
    dataset = abalone(path, True)
    print(dataset)
