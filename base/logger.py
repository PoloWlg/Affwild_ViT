from base.utils import ensure_dir, save_to_pickle, sigmoid
import os

import umap
import torch

import statistics
import numpy as np
import pandas as pd

from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score
from torcheval.metrics.functional import multiclass_f1_score, multiclass_confusion_matrix

from matplotlib.colors import ListedColormap
import seaborn as sns

import matplotlib.pyplot as plt

import logging

# Suppress only the specific warning from root logger
logging.getLogger("root").setLevel(logging.ERROR)


class PlotHandler(object):
    r"""
    A class to plot the output-label figures.
    """

    def __init__(self, metrics, emotion, epoch_result_dict,
                 trialwise_output_dict, trialwise_continuous_label_dict,
                 epoch=None, train_mode=None, directory_to_save_plot=None):
        self.metrics = metrics
        self.emotion = emotion
        self.epoch_result_dict = epoch_result_dict

        self.epoch = epoch
        self.train_mode = train_mode
        self.directory_to_save_plot = directory_to_save_plot

        self.trialwise_output_dict = trialwise_output_dict
        self.trialwise_continuous_label_dict = trialwise_continuous_label_dict

    def complete_directory_to_save_plot(self):
        r"""
        Determine the full path to save the plot.
        """
        if self.train_mode:
            exp_folder = "train"
        else:
            exp_folder = "validate"

        if self.epoch is None:
            exp_folder = "test"

        directory = os.path.join(self.directory_to_save_plot, "plot", exp_folder, "epoch_" + str(self.epoch))
        if self.epoch == "test":
            directory = os.path.join(self.directory_to_save_plot, "plot", exp_folder)

        os.makedirs(directory, exist_ok=True)
        return directory

    def save_output_vs_continuous_label_plot(self):
        r"""
        Plot the output versus continuous label figures for each session.
        """

        for (trial, output_record), (_, label_record) in zip(self.trialwise_output_dict.items(), self.trialwise_continuous_label_dict.items()):

            complete_directory = self.complete_directory_to_save_plot()

            plot_filename = trial
            full_plot_filename = os.path.join(complete_directory, plot_filename + ".jpg")

            self.plot_and_save(full_plot_filename, trial, output_record, label_record)

    def plot_and_save(self, full_plot_filename, trial, output, continuous_label):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1)

        result_list = []

        for metric in self.metrics:
            result = self.epoch_result_dict[trial][metric]
            # The pcc usually have two output, one for value and one for confidence. So
            # here we only read and the value and discard the confidence.
            if metric == "pcc":
                result = self.epoch_result_dict[trial][metric][0]
            result_list.append(result)

        ax.plot(output, "r-", label="Output")
        ax.plot(continuous_label, "g-", label="Label")
        ax.set_xlabel("Sample")
        ax.set_ylabel("Value")
        ax.legend(loc="upper right", framealpha=0.2)
        ax.title.set_text(
            "{}: rmse={:.3f}, pcc={:.3f}, ccc={:.3f}".format(self.emotion, *result_list))
        fig.tight_layout()
        plt.savefig(full_plot_filename)
        plt.close()


class ContinuousOutputHandler(object):
    def __init__(self):
        self.seen_trials = []
        self.trialwise_records = {}
        self.partition_records = []

    def update_output_for_seen_trials(self, output, trials, indices, lengths):

        for i, trial in enumerate(trials):

            # If this is the first time to record the output for trial
            if trial not in self.seen_trials:
                self.seen_trials.append(trial)
                self.trialwise_records[trial] = self.create_list_for_a_trial(lengths[i])

            index = indices[i]
            for k, data in enumerate(output[i, :, 0]):
                if k == lengths[i]:
                    break
                self.trialwise_records[trial][index[k]].append(output[i, k, :])
                

    def average_trial_wise_records(self):

        for trial in self.seen_trials:
            length = len(self.trialwise_records[trial])

            for i in range(length):
                # self.trialwise_records[trial][i] = statistics.mean(
                #     np.float64(self.trialwise_records[trial][i]))
                
                self.trialwise_records[trial][i] = [sum(elements) / len(elements) for elements in zip(*self.trialwise_records[trial][i])]

            self.trialwise_records[trial] = np.asarray(self.trialwise_records[trial])
            
    def non_average_trial_wise_records(self):

        for trial in self.seen_trials:
            length = len(self.trialwise_records[trial])

            for i in range(length):
                # self.trialwise_records[trial][i] = statistics.mean(
                #     np.float64(self.trialwise_records[trial][i]))
                
                self.trialwise_records[trial][i] =  self.trialwise_records[trial][i][0]

            self.trialwise_records[trial] = np.asarray(self.trialwise_records[trial])

    def concat_records(self):
        for trial in self.seen_trials:
            self.partition_records.extend(self.trialwise_records[trial])

        self.partition_records = np.asarray(self.partition_records)

    def create_list_for_a_trial(self, length):
        trial_record = [[] for i in range(length)]
        return trial_record

class FeatureMapsOutputHandler(object):
    def __init__(self):
        self.feature_maps_handler = []
        for i in range (8):
            self.feature_maps_handler.append(ContinuousOutputHandler())
            
        self.features_res50_handler = ContinuousOutputHandler()
    
    def update_output_for_seen_trials(self, trials, indices, lengths, model):
        umaps = model.get_feature_maps().permute(2,0,1,3).detach().cpu().numpy()
        
        for i, feature in enumerate(umaps):
            self.feature_maps_handler[i].update_output_for_seen_trials(feature, trials, indices, lengths)
            
        self.features_res50_handler.update_output_for_seen_trials(model.get_res50_features().transpose(1,2).detach().cpu().numpy(), trials, indices, lengths)
        
    def average_trial_wise_records(self):
        print('average trial wise records')
        self.features_res50_handler.non_average_trial_wise_records()
        for i in range (len(self.feature_maps_handler)):
            self.feature_maps_handler[i].non_average_trial_wise_records()
            
    def concat_records(self):
        print('Concat records')
        self.features_res50_handler.concat_records()
        for i in range (len(self.feature_maps_handler)):
            self.feature_maps_handler[i].concat_records()
            
    def get_feature_maps(self, all_labels, save_path):
        
        print('process and save umaps')
        umaps = self.features_res50_handler.partition_records
        self.process_and_save_umaps(umaps, all_labels, 'test_polo', save_path)
        
        for i in range (len(self.feature_maps_handler)):
            umaps = self.feature_maps_handler[i].partition_records
            self.process_and_save_umaps(umaps, all_labels, i, save_path)
            
    def process_and_save_umaps(self, features, labels, name, save_path):
        
        labels = labels.squeeze()
        
        # Paths
        path_to_save_umaps_features = os.path.join(save_path,f'umap_result_{name}.npy' )
        path_to_save_fig = os.path.join(save_path, f'{name}.png')
        
        # Save features
        np.save(path_to_save_umaps_features, features)
        
        # Define emotions and corresponding colors
        emotions = ['Neutral', 'Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Other']
        colors = sns.color_palette("hsv", len(emotions))  # HSV palette to represent emotions

        # Create a UMAP object to reduce dimensions to 2D for visualization
        umap_model = umap.UMAP(n_jobs=90, n_components=2, random_state=42, verbose=True, low_memory = False)

        # Fit and transform the features
        embedding = umap_model.fit_transform(features)

        # Plot the UMAP projection with different colors for each emotion
        plt.figure(figsize=(10, 8), dpi=300)
        for i, emotion in enumerate(emotions):
            plt.scatter(embedding[labels == i, 0], embedding[labels == i, 1], 
                        color=colors[i], label=emotion, s=0.1)

        plt.legend(markerscale=30, loc='best', title='Emotions')
        plt.title('UMAP Projection of Emotions', fontsize=16)
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.savefig(path_to_save_fig)
            
            
            
class ConcordanceCorrelationCoefficient(object):
    """
    A class for performing concordance correlation coefficient (CCC) centering. Basically, when multiple continuous labels
    are available, it is not a good choice to perform a direct average. Formally, a Lin's CCC centering has to be done.

    This class is a Pythonic equivalence of CCC centering to the Matlab scripts ("run_gold_standard.m")
        from the AVEC2016 dataset.

    Ref:
        "Lawrence I-Kuei Lin (March 1989).  A concordance correlation coefficient to evaluate reproducibility".
            Biometrics. 45 (1): 255–268. doi:10.2307/2532051. JSTOR 2532051. PMID 2720055.
    """

    def __init__(self, data):
        self.data = data
        if data.shape[0] > data.shape[1]:
            self.data = data.T
        self.rator_number = self.data.shape[0]
        self.combination_list = self.generate_combination_pair()
        self.cnk_matrix = self.generate_cnk_matrix()
        self.ccc = self.calculate_paired_ccc()
        self.agreement = self.calculate_rator_wise_agreement()
        self.mean_data = self.calculate_mean_data()
        self.weight = self.calculate_weight()
        self.centered_data = self.perform_centering()

    def perform_centering(self):
        """
        The centering is done by directly average the shifted and weighted data.
        :return: (ndarray), the centered  data.
        """
        centered_data = self.data - np.repeat(self.mean_data[:, np.newaxis], self.data.shape[1], axis=1) + self.weight
        return centered_data

    def calculate_weight(self):
        """
        The weight of the m continuous labels. It will be used to weight (actually translate) the data when
            performing the final step.
        :return: (float), the weight of the given m continuous labels.
        """
        weight = np.sum((self.mean_data * self.agreement) / np.sum(self.agreement))
        return weight

    def calculate_mean_data(self):
        """
        A directly average of data.
        :return: (ndarray), the averaged data.
        """
        mean_data = np.mean(self.data, axis=1)
        return mean_data

    def generate_combination_pair(self):
        """
        Generate all possible combinations of Cn2.
        :return: (ndarray), the combination list of Cn2.
        """
        n = self.rator_number
        combination_list = []

        for boy in range(n - 1):
            for girl in np.arange(boy + 1, n, 1):
                combination_list.append([boy, girl])

        return np.asarray(combination_list)

    def generate_cnk_matrix(self):
        """
        Generate the Cn2 matrix. The j-th column of the matrix records all the possible candidate
            to the j-th rater. So that for the j-th column, we can acquire all the possible unrepeated
            combination for the j-th rater.
        :return:
        """
        total = self.rator_number
        cnk_matrix = np.zeros((total - 1, total))

        for column in range(total):
            cnk_matrix[:, column] = np.concatenate((np.where(self.combination_list[:, 0] == column)[0],
                                                    np.where(self.combination_list[:, 1] == column)[0]))

        return cnk_matrix.astype(int)

    @staticmethod
    def calculate_ccc(x, y):
        """
        Calculate the CCC.
        :param x: (ndarray), an 1xn array.
        :param y: (ndarray), another 1xn array.
        :return: the CCC.
        """
        is_multitask = 0
        if len(x.shape) == 2:
            is_multitask = 1

        x_mean = np.nanmean(x)
        y_mean = np.nanmean(y)

        if is_multitask:
            x_mean = np.nanmean(x, axis=1)[:, None]
            y_mean = np.nanmean(y, axis=1)[:, None]

        covariance = np.nanmean((x - x_mean) * (y - y_mean))
        if is_multitask:
            covariance = np.nanmean((x - x_mean) * (y - y_mean), axis=1)[:, None]

        # Make it consistent with Matlab's nanvar (division by len(x)-1, not len(x)))
        x_var = 1.0 / (len(x) - 1) * np.nansum((x - x_mean) ** 2)
        y_var = 1.0 / (len(y) - 1) * np.nansum((y - y_mean) ** 2)
        if is_multitask:
            x_var = 1.0 / (len(x[-1]) - 1) * np.nansum((x - x_mean) ** 2, axis=1)[:, None]
            y_var = 1.0 / (len(y[-1]) - 1) * np.nansum((y - y_mean) ** 2, axis=1)[:, None]

        concordance_correlation_coefficient = \
            (2 * covariance) / (x_var + y_var + (x_mean - y_mean) ** 2 + 1e-100)

        return concordance_correlation_coefficient

    def calculate_paired_ccc(self):
        """
        Calculate the CCC for all the pairs from the combination list.
        :return: (ndarray), the CCC for each combinations.
        """
        ccc = np.zeros((self.combination_list.shape[0]))
        for index in range(len(self.combination_list)):
            ccc[index] = self.calculate_ccc(self.data[self.combination_list[index, 0], :],
                                            self.data[self.combination_list[index, 1], :])

        return ccc

    def calculate_rator_wise_agreement(self):
        """
        Calculate the inter-rater CCC agreement.
        :return: (ndarray), a array recording the CCC agreement of each single rater to all the rest raters.
        """

        ccc_agreement = np.zeros(self.rator_number)

        for index in range(self.rator_number):
            ccc_agreement[index] = np.mean(self.ccc[self.cnk_matrix[:, index]])

        return ccc_agreement


class ContinuousMetricsCalculator(object):
    r"""
        A class to calculate the metrics, usually rmse, pcc, and ccc for continuous regression.
        """

    def __init__(
            self,
            metrics,
            emotion,
            output_handler,
            continuous_label_handler,
    ):

        # What metrics to calculate.
        self.metrics = metrics

        # What emotional dimensions to consider.
        self.emotion = emotion

        # The instances saving the data for evaluation.
        self.output_handler = output_handler
        self.continuous_label_handler = continuous_label_handler

        # Initialize the dictionary for saving the metric results.
        self.metric_record_dict = self.init_metric_record_dict()

    def get_partitionwise_output_and_continuous_label(self):
        return self.output_handler.partition_records, \
               self.continuous_label_handler.partition_records

    def get_trialwise_output_and_continuous_label(self):
        return self.output_handler.trialwise_records, \
               self.continuous_label_handler.trialwise_records

    def init_metric_record_dict(self):
        trialwise_dict, _ = self.get_trialwise_output_and_continuous_label()
        metric_record_dict = {key: [] for key in trialwise_dict}
        return metric_record_dict

    def calculator(self, output, label, metric):
        if metric == "rmse":
            result = np.sqrt(((output - label) ** 2).mean())
        elif metric == "pcc":
            # result = pearsonr(output, label)
            result = multiclass_f1_score(torch.tensor(output.tolist()), torch.tensor(label.flatten()), num_classes=8)
        elif metric == "f1_score_average":
            result = multiclass_f1_score(torch.tensor(output.tolist()), torch.tensor(label.flatten(), dtype=torch.int64), num_classes=8, average='macro').item()
            result = np.round(result,4)
        elif metric == "f1_score":
            result_per_class = multiclass_f1_score(torch.tensor(output.tolist()), torch.tensor(label.flatten(), dtype=torch.int64), num_classes=8, average=None).numpy()
            result_per_class = np.round(result_per_class*100, 1)
            formated_result_per_class = {
                'neutral': result_per_class[0],
                'anger': result_per_class[1],
                'disgust': result_per_class[2],
                'fear': result_per_class[3],
                'happiness': result_per_class[4],
                'sadness': result_per_class[5],
                'surprise': result_per_class[6],
                'other': result_per_class[7],
            }
            clean_emotions = {k: round(float(v), 4) for k, v in formated_result_per_class.items()}
            result = clean_emotions
        
        elif metric == "class_distribution":
            predicted_class = np.argmax(output.tolist(), axis=1) 
            predicted_class = torch.tensor(predicted_class)
            conf_matrix = multiclass_confusion_matrix(predicted_class, torch.tensor(label.flatten(), dtype=torch.int64), num_classes=8)
            
            # Calculate the sum of each row (sum across true labels)
            row_sums = conf_matrix.sum(dim=1, keepdim=True)
            col_sums = conf_matrix.sum(dim=0, keepdim=True)
            total_sum = conf_matrix.sum().unsqueeze(0).unsqueeze(1)
            
            conf_matrix_with_sums = torch.cat([conf_matrix, row_sums], dim=1)  # Add row sums
            col_sums_with_total = torch.cat([col_sums, total_sum], dim=1)       # Add total sum to column sums
            conf_matrix_with_sums = torch.cat([conf_matrix_with_sums, col_sums_with_total], dim=0)  # Add column sums
                        
            result = conf_matrix_with_sums.numpy()
            
        elif metric == "class_distribution_percent":
            predicted_class = np.argmax(output.tolist(), axis=1) 
            predicted_class = torch.tensor(predicted_class)
            conf_matrix = multiclass_confusion_matrix(predicted_class, torch.tensor(label.flatten(), dtype=torch.int64), num_classes=8)
            conf_matrix_percent = conf_matrix.float() / conf_matrix.sum(dim=1, keepdim=True) * 100
            conf_matrix_percent = conf_matrix_percent.numpy()
            result = conf_matrix_percent
            
        else:
            raise ValueError("Metric {} is not defined.".format(metric))
        return result

    def calculate_metrics(self):

        # Load the data for three scenarios.
        # They will all be evaluated.
        trialwise_output, trialwise_continuous_label = self.get_trialwise_output_and_continuous_label()
        partitionwise_output, partitionwise_continuous_label = self.get_partitionwise_output_and_continuous_label()

        for (trial_id, output), (_, label) in zip(
                trialwise_output.items(), trialwise_continuous_label.items()):

            result_dict = {metric: [] for metric in self.metrics}
            for metric in self.metrics:
                result_dict[metric] = self.calculator(output, label, metric)
                # if metric == "pcc":
                #     result_dict[metric] = [result]
            trial_record_dict = result_dict
            self.metric_record_dict[trial_id] = trial_record_dict

        # Partition-wise evaluation

        partitionwise_dict = {metric: [] for metric in self.metrics}

        for metric in self.metrics:
            result = self.calculator(partitionwise_output, partitionwise_continuous_label, metric)
            partitionwise_dict[metric] = result

        self.metric_record_dict['overall'] = partitionwise_dict

    def save_trial_wise_records(self, save_path, train_mode, epoch):
        save_path = self.get_save_path(save_path, train_mode, epoch)

        trialwise_output, trialwise_continuous_label = self.get_trialwise_output_and_continuous_label()
        pkl_to_save = {'output': trialwise_output, 'continuous_label': trialwise_continuous_label, 'metrics': self.metric_record_dict}
        save_to_pickle(save_path, pkl_to_save)

    def get_save_path(self, save_path, train_mode, epoch):

        save_folder = "validate"
        if train_mode:
            save_folder = "train"

        if epoch is None:
            save_folder = "test"


        if epoch is None:
            save_path = os.path.join(save_path, "dict", self.emotion, save_folder + ".pkl")
        else:
            save_path = os.path.join(save_path, "dict", self.emotion, save_folder, "epoch_" + str(epoch) + ".pkl")

        ensure_dir(save_path)
        return save_path




class FeaturesHandler(object):
    def __init__(self):
        self.seen_trials = []
        self.trialwise_records = {}
        self.partition_records = []

    def update_output_for_seen_trials(self, output, trials, indices, lengths):

        for i, trial in enumerate(trials):

            # If this is the first time to record the output for trial
            if trial not in self.seen_trials:
                self.seen_trials.append(trial)
                self.trialwise_records[trial] = self.create_list_for_a_trial(lengths[i])

            index = indices[i]
            for k, data in enumerate(output[i, :, :]):
                if k == lengths[i]:
                    break
                self.trialwise_records[trial][index[k]] = output[i, k, :].cpu().detach().numpy()
        
    def create_list_for_a_trial(self, length):
        # trial_record = [[] for i in range(length)]
        trial_record = np.zeros((length, 512, 5, 5))
        return trial_record