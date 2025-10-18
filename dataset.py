from base.dataset import GenericDataArranger, GenericDataset
from torch.utils.data import Dataset as PytorchDataset

import numpy as np
from PIL import Image


class Dataset(GenericDataset):
    def __init__(self,context_path, data_list, continuous_label_dim, modality, multiplier, feature_dimension, window_length, mode, mean_std=None,
                 time_delay=0, feature_extraction=0, context_feature_model=None ):
        super().__init__(data_list,context_path, continuous_label_dim, modality, multiplier, feature_dimension, window_length, mode, mean_std=mean_std,
                 time_delay=time_delay, feature_extraction=feature_extraction, context_feature_model=context_feature_model) 


class DataArranger(GenericDataArranger):
    def __init__(self, dataset_info, dataset_path, debug):
        super().__init__(dataset_info, dataset_path, debug)

    @staticmethod
    def get_feature_list():
        # feature_list = ['vggish', 'bert', 'egemaps']
        # feature_list = ['vggish', 'logmel','video128']
        feature_list = ['vggish','video']
        return feature_list

    def partition_range_fn(self):
        # partition_range = {
        #     'train': [np.arange(0, 71), np.arange(71, 142), np.arange(142, 213), np.arange(213, 284), np.arange(284, 356)],
        #     'validate': [np.arange(356, 432)],
        #     'test': [],
        #     'extra': [np.arange(432, 594)]}

        partition_range = {
            'train': [np.arange(0, 61), np.arange(61, 122), np.arange(122, 183), np.arange(183, 244)],
            'validate': [np.arange(244, 311)],
            'test': [],
            'extra': []}


        if self.debug == 1:
            partition_range = {
                'train': [np.arange(0, 10), np.arange(10, 20), np.arange(20, 30), np.arange(30, 40) ],
                'validate': [np.arange(60, 70)],
                'test': [],
                'extra': []}
            
        if self.debug == 2:
            partition_range = {
                'train': [np.arange(0,1), np.arange(1, 2), np.arange(2, 3), np.arange(3, 4) ],
                'validate': [np.arange(8, 10)],
                'test': [],
                'extra': []}

        return partition_range

    @staticmethod
    def assign_fold_to_partition():
        fold_to_partition = {'train': 4, 'validate': 1, 'test': 0, 'extra': 0}
        return fold_to_partition


