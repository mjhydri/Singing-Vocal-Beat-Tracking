from torch.utils.data import Dataset
from random import shuffle as sh


class split(Dataset):
    def __init__(self, base_dir=None, splits=None, train_test_ratio="all_for_train", shuffle=None,):
        self.base_dir = base_dir
        self.split = splits
        self.shuffle = shuffle
        self.train_test_ratio = train_test_ratio


    if splits is not None:
        splits = self.available_splits()
    self.splits = splits

    def