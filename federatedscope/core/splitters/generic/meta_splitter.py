import random
import numpy as np
import logging

from federatedscope.core.splitters import BaseSplitter
from federatedscope.core.splitters.generic import IIDSplitter

logger = logging.getLogger(__name__)


class MetaSplitter(BaseSplitter):
    """
    This splitter split dataset with meta information with LLM dataset.

    Args:
        client_num: the dataset will be split into ``client_num`` pieces
    """
    def __init__(self, client_num, **kwargs):
        super(MetaSplitter, self).__init__(client_num)
        # Create an IID spliter in case that num_client < categories
        self.iid_spliter = IIDSplitter(client_num)

    def __call__(self, dataset, prior=None, **kwargs):
        from torch.utils.data import Dataset, Subset

        tmp_dataset = [ds for ds in dataset]
        if isinstance(tmp_dataset[0], tuple):
            label = np.array([y for x, y in tmp_dataset])
        elif isinstance(tmp_dataset[0], dict):
            label = np.array([x['categories'] for x in tmp_dataset])
        else:
            raise TypeError(f'Unsupported data formats {type(tmp_dataset[0])}')

        # Split by categories
        categories = set(label)
        idx_slice = []
        for cat in categories:
            idx_slice.append(np.where(np.array(label) == cat)[0].tolist())
        random.shuffle(idx_slice)

        # print the size of each categories
        tot_size = 0
        for i, cat in enumerate(categories):
            logger.info(f'Index: {i}\t'
                        f'Category: {cat}\t'
                        f'Size: {len(idx_slice[i])}')
            tot_size += len(idx_slice[i])
        logger.info(f'Total size: {tot_size}')

        if len(categories) < self.client_num:
            logger.warning(
                f'The number of clients is {self.client_num}, which is '
                f'smaller than a total of {len(categories)} catagories, '
                'use iid splitter instead.')
            return self.iid_spliter(dataset)

        # Merge to client_num pieces
        new_idx_slice = []
        for i in range(len(categories)):
            if i < self.client_num:
                new_idx_slice.append(idx_slice[i])
            else:
                new_idx_slice[i % self.client_num] += idx_slice[i]

        if isinstance(dataset, Dataset):
            data_list = [Subset(dataset, idxs) for idxs in idx_slice]
        else:
            data_list = [[dataset[idx] for idx in idxs] for idxs in idx_slice]
        return data_list
