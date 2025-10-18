import os
from operator import itemgetter

from collections import OrderedDict

from torch.utils.data import Dataset
from torchvision.transforms import transforms

from base.transforms3D import *
from base.utils import load_npy
import pandas as pd

from transformers import RobertaModel, RobertaTokenizer
from sentence_transformers import SentenceTransformer

from tqdm import tqdm


class GenericDataArranger(object):
    def __init__(self, dataset_info, dataset_path, debug):
        self.dataset_info = dataset_info
        self.debug = debug
        self.trial_list = self.generate_raw_trial_list(dataset_path)
        self.partition_range = self.partition_range_fn()
        self.fold_to_partition = self.assign_fold_to_partition()

    def generate_iterator(self):
        iterator = self.dataset_info['partition']
        return iterator

    def generate_partitioned_trial_list(self, window_length, hop_length, fold, windowing=True):
        
        # # Double check if all videos are present 
        # path_train_annotations = '/projets/AS84330/Datasets2/Abaw6_EXPR_contextual/6th_ABAW_Annotations/EXPR/train'
        # path_val_annotations = '/projets/AS84330/Datasets2/Abaw6_EXPR_contextual/6th_ABAW_Annotations/EXPR/validate'
        # video_files_train = os.listdir(path_train_annotations)
        # video_files_val = os.listdir(path_val_annotations)
            
        train_validate_range = self.partition_range['train'] + self.partition_range['validate']
        assert  len(train_validate_range) == self.fold_to_partition['train'] + self.fold_to_partition['validate']
        train_validate_range = np.asarray(train_validate_range, dtype="object")
        partition_range = list(np.roll(train_validate_range, fold))
        # partition_range += self.partition_range['test'] + self.partition_range['extra']
        partitioned_trial = {}

        for partition, num_fold in self.fold_to_partition.items():
            partitioned_trial[partition] = []

            for i in range(num_fold):
                index = partition_range.pop(0)
                trial_of_this_fold = list(itemgetter(*index)(self.trial_list))

                if len(index) == 1:
                    trial_of_this_fold = [trial_of_this_fold]

                for path, trial, length in trial_of_this_fold:
                    if not windowing:
                        window_length = length
                        
                    video_name = os.path.split(path)[-1]


                    windowed_indices = self.windowing(np.arange(length), window_length=window_length,
                                                      hop_length=hop_length)
                    
                    
                    path_to_partitionned_video = os.path.join('/projets2/AS84330/Datasets/Abaw6_EXPR/partitioned_trial_videos', video_name)
                    if os.path.exists(path_to_partitionned_video):
                        video_files = [f for f in os.listdir(path_to_partitionned_video)]
                        
                        if (len(video_files) != len(windowed_indices)):
                            print(f"File name: : {path_to_partitionned_video}")
                            print("difference: ", len(video_files) - len(windowed_indices))
                    

                    for indice,index in enumerate(windowed_indices):
                        partitioned_trial[partition].append([path, trial, length, index, indice])

        return partitioned_trial

    def calculate_mean_std(self, partitioned_trial):
        feature_list = self.get_feature_list()
        mean_std_dict = {partition: {feature: {'mean': None, 'std': None} for feature in feature_list} for partition in partitioned_trial.keys()}

        # Calculate the mean
        for feature in feature_list:
            for partition, trial_of_a_partition in partitioned_trial.items():
                lengths = 0
                sums = 0
                for path, _, _, _, _ in trial_of_a_partition:
                    data = load_npy(path, feature)
                    data = data.flatten()
                    lengths += len(data)
                    sums += data.sum()
                mean_std_dict[partition][feature]['mean'] = sums / (lengths + 1e-10)

        # Then calculate the standard deviation.
        for feature in feature_list:
            for partition, trial_of_a_partition in partitioned_trial.items():
                lengths = 0
                x_minus_mean_square = 0
                mean = mean_std_dict[partition][feature]['mean']
                for path, _, _, _, _ in trial_of_a_partition:
                    data = load_npy(path, feature)
                    data = data.flatten()
                    lengths += len(data)
                    x_minus_mean_square += np.sum((data - mean) ** 2)
                x_minus_mean_square_divide_N_minus_1 = x_minus_mean_square / (lengths - 1)
                mean_std_dict[partition][feature]['std'] = np.sqrt(x_minus_mean_square_divide_N_minus_1)

        return mean_std_dict

    @staticmethod
    def partition_range_fn():
        raise NotImplementedError

    @staticmethod
    def assign_fold_to_partition():
        raise NotImplementedError

    @staticmethod
    def get_feature_list():
        feature_list = ['landmark', 'action_unit', 'mfcc', 'egemaps', 'vggish', 'bert']
        return feature_list

    def generate_raw_trial_list(self, dataset_path):
        trial_path = os.path.join(dataset_path, self.dataset_info['data_folder'])

        trial_dict = OrderedDict({'train': [], 'validate': [], 'extra': [], 'test': []})
        for idx, partition in enumerate(self.generate_iterator()):

            if partition == "unused":
                continue

            trial = self.dataset_info['trial'][idx]
            path = os.path.join(trial_path, trial)
            length = self.dataset_info['length'][idx]

            trial_dict[partition].append([path, trial, length])

        trial_list = []
        for partition, trials in trial_dict.items():
            trial_list.extend(trials)

        return trial_list

    @staticmethod
    def windowing(x, window_length, hop_length):
        length = len(x)

        if length >= window_length:
            steps = (length - window_length) // hop_length + 1

            sampled_x = []
            for i in range(steps):
                start = i * hop_length
                end = start + window_length
                sampled_x.append(x[start:end])

            if sampled_x[-1][-1] < length - 1:
                sampled_x.append(x[-window_length:])
        else:
            sampled_x = [x]

        return sampled_x


class GenericDataset(Dataset):
    def __init__(self, context_path, data_list, continuous_label_dim, modality, multiplier, feature_dimension, window_length, mode, mean_std=None,
                 time_delay=0, feature_extraction=0, context_feature_model=None):
        self.data_list = data_list
        self.mode = mode
        self.continuous_label = self.init_continuous_label()
        self.features_video = self.init_features_videos()
        self.continuous_label_dim = continuous_label_dim
        self.mean_std = mean_std
        self.mean_std_info = 0
        self.time_delay = time_delay
        self.modality = modality
        self.multiplier = multiplier
        self.feature_dimension = feature_dimension
        self.feature_extraction = feature_extraction
        self.window_length = window_length
        self.transform_dict = {}
        self.get_3D_transforms()
        self.context_path = context_path
        self.context_feature_model = context_feature_model
        
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True, clean_up_tokenization_spaces=True, model_max_length=512)

    def init_continuous_label(self):
        path = '/projets/AS84330/Datasets2/Abaw6_EXPR_contextual/compacted_48/'
        videos = []
        for _, name, _,_,_ in self.data_list:
            videos.append(name)
        videos = np.unique(videos)
        
        annotations = {}
        for video in videos: 
            video = str(video)
            annotations[video] = np.load(os.path.join(path, video, 'EXPR_continuous_label.npy'))
        
        return annotations
    
    def init_features_videos(self):
        path = os.path.join('/projets/AS84330/Datasets2/Abaw6_EXPR_contextual/features/raw_video_features', self.mode)
        
        features_videos = {}
        for video in tqdm(os.listdir(path), total=len(os.listdir(path)), desc=f'loading features ... [{self.mode}]'): 
            video = str(video).replace('.npy', '')
            features_videos[video] = np.load(os.path.join(path, video + '.npy'))
        
        return features_videos
        
    
    def get_index_given_emotion(self):
        raise NotImplementedError

    def get_3D_transforms(self):
        normalize = GroupNormalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        if "video" in self.modality:
            if self.mode == 'train':
                self.transform_dict['video'] = transforms.Compose([
                    GroupNumpyToPILImage(0),
                    GroupRandomCrop(48, 40),
                    GroupRandomHorizontalFlip(),
                    Stack(),
                    ToTorchFormatTensor(),
                    normalize
                ])
            else:
                self.transform_dict['video'] = transforms.Compose([
                    GroupNumpyToPILImage(0),
                    GroupCenterCrop(40),
                    Stack(),
                    ToTorchFormatTensor(),
                    normalize
                ])

        for feature in self.modality:
            if "continuous_label" not in feature and "video" not in feature:
                self.transform_dict[feature] = self.get_feature_transform(feature)

    def get_feature_transform(self, feature):
        if  "logmel" or "context" in feature:
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[self.mean_std[feature]['mean']],
                                     std=[self.mean_std[feature]['std']])
            ])
        return transform
    
    def partition_per_class_distribution(self):
        all_labels = [0,0,0,0,0,0,0,0]
        for data in self.data_list:
            path, trial, length, index, indice = data
            examples = {}
            example = self.load_data(path, index, 'EXPR_continuous_label', indice)
            
            for label in example:
                label = int(label[0])
                all_labels[label] +=1 
        
        # print(all_labels)
        total = 0 
        for num in all_labels:
            total += num
        print(f"neutral:   {round((all_labels[0]/total)*100, 1)}%, {all_labels[0]}")
        print(f"anger:     {round((all_labels[1]/total)*100, 1)}%, {all_labels[1]}")
        print(f"disgust:   {round((all_labels[2]/total)*100, 1)}%, {all_labels[2]}")
        print(f"fear:      {round((all_labels[3]/total)*100, 1)}%, {all_labels[3]}")
        print(f"happiness: {round((all_labels[4]/total)*100, 1)}%, {all_labels[4]}")
        print(f"sadness:   {round((all_labels[5]/total)*100, 1)}%, {all_labels[5]}")
        print(f"surprise:  {round((all_labels[6]/total)*100, 1)}%, {all_labels[6]}")
        print(f"other:     {round((all_labels[7]/total)*100, 1)}%, {all_labels[7]}")
        
        

    def __getitem__(self, index):
        path, trial, length, index, indice = self.data_list[index]
        examples = {}

        # for feature in self.modality:
        #     examples[feature] = self.get_example(path, length, index, feature, indice)
            

        examples['EXPR_continuous_label'] = self.get_features('EXPR_continuous_label', trial, length, index)
        features_video = self.get_features('features_video', trial, length, index)
        examples['video'] = torch.zeros([300, 3, 40, 40])
        examples['context'] = self.get_example(path, length, index, 'context', indice)
        examples['video'] = self.get_example(path, length, index, 'video', indice)
        
        # features_video = self.get_features_video(trial, length, index)
        
        # features_video = self.features_video[trial][index]
        # features_video = torch.zeros([300, 512, 5, 5])
        if len(index) < self.window_length:
            index = np.arange(self.window_length)
        return examples, trial, length, index, features_video

    def __len__(self):
        return len(self.data_list)

    
    
    def get_features(self, modal, trial, length, indices):
        
        if modal == 'features_video':
            data = self.features_video[trial][indices]
        elif modal == 'EXPR_continuous_label':
            data = self.continuous_label[trial][indices]
        
        if length < self.window_length:
            dtype = np.float32
            shape = (300, ) + data.shape[1:]
            example = np.zeros(shape=shape, dtype=dtype)
            example[indices] = data
            data = example
            
        return data
    def get_example(self, path, length, index, feature, indice):


        x = random.randint(0, self.multiplier[feature] - 1)
        random_index = index * self.multiplier[feature] + x

        # Probably, a trial may be shorter than the window, so the zero padding is employed.
        # if (feature == "context"):
        #     example = self.load_data(path, random_index, feature, indice)
        if length < self.window_length:
            shape = (self.window_length,) + self.feature_dimension[feature]
            dtype = np.float32
            if feature == "video":
                dtype = np.int8
            example = np.zeros(shape=shape, dtype=dtype)
            if feature == 'context':
                shape = (self.window_length,) + (4096,)
                example = np.zeros(shape=shape, dtype=dtype)
            example[index] = self.load_data(path, random_index, feature, indice)
            example[index[-1]: self.window_length-1] = example[index[-1]]
        else:
            example = self.load_data(path, random_index, feature, indice)

        # Sometimes we may want to shift the label, so that
        # the ith label point  corresponds to the (i - time_delay)-th data point.
        if "continuous_label" in feature and self.time_delay != 0:
            example = np.concatenate(
                (example[self.time_delay:, :],
                 np.repeat(example[-1, :][np.newaxis], repeats=self.time_delay, axis=0)), axis=0)

        if ("continuous_label" not in feature) and ("context" not in feature):
            example = self.transform_dict[feature](np.asarray(example, dtype=np.float32))
        
        if "context" in feature:
            example = example
        return example

    def load_data(self, path, indices, feature, indice):
        if feature == "context":
            # TODO: add the context processing 
            return self.load_context_features(path, indices)
        filename = os.path.join(path, feature + ".npy")

        # For the test set, labels of zeros are generated as dummies.
        data = np.zeros(((len(indices),) + self.feature_dimension[feature]), dtype=np.float32)

        if os.path.isfile(filename):
            if self.feature_extraction:
                data = np.load(filename, mmap_mode='c')
            else:
                data = np.load(filename, mmap_mode='c')[indices]

            if "continuous_label" in feature:
                data = self.processing_label(data)
        return data
    
        
    def processing_label(self, label):
        label = label[:, self.continuous_label_dim]
        if label.ndim == 1:
            label = label[:, None]
        return label

    def load_context(self, path, indices):
        # Load the context data
        context_path = os.path.join(path, 'context.tsv')
        context_data = pd.read_csv(context_path, sep='\t', header=None)
        context_data = context_data.values
        context_data = context_data[indices]
        
        
        # if len(context_data) != 300:
        #     raise ValueError(f"Expected 300 context data points, but got {len(context_data)}")
        
        context_data = [item[1] for item in context_data]
        sentence_embeddings = self.create_sentence_embeddings(context_data)
        
        
        return sentence_embeddings
    
    def load_context_features(self, path, indices):
        # Load the context data
        context_path_folder = os.path.join(path, 'context', 'features')
        filename = os.path.join(context_path_folder, f'{self.context_feature_model}.npy')
        data = np.load(filename, mmap_mode='c')
        data = data[indices]
        
        return data
        
        
        

    def create_sentence_embeddings(self, context_data):
        # model = SentenceTransformer("all-MiniLM-L6-v2")
        model = SentenceTransformer("bert-base-nli-mean-tokens")
        sentence_embeddings = model.encode(context_data)
        return sentence_embeddings
    
    