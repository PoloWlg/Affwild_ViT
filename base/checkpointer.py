from base.utils import load_pickle, save_to_pickle

import os
import time

import numpy as np
import pandas as pd

import json
import sys



class GenericCheckpointer(object):
    r"""
    Save the trainer and parameter controller at runtime, and load them
        in another run if resume = 1.
    """
    def __init__(self, path, trainer, parameter_controller, resume):
        self.checkpoint = {}
        self.path = path
        self.trainer = trainer
        self.parameter_controller = parameter_controller
        self.resume = resume

    def load_checkpoint(self):
        # If checkpoint file exists, then read it.
        if os.path.isfile(self.path):
            print("Loading checkpoint. Are you sure it is intended?")
            self.checkpoint = {**self.checkpoint, **load_pickle(self.path)}
            print("Checkpoint loaded!")

            self.trainer = self.checkpoint['trainer']
            self.trainer.resume = True
            self.parameter_controller = self.checkpoint['param_control']
            self.parameter_controller.trainer = self.trainer
        else:
            raise ValueError("Checkpoint not exists!!")
        return self.trainer, self.parameter_controller
    
    def load_weigths(self, path_weights):
        # If checkpoint file exists, then read it.
        if os.path.isfile(path_weights):
            print("Loading model weights. Are you sure it is intended?")
            self.checkpoint = {**self.checkpoint, **load_pickle(path_weights)}
            self.trainer.model = self.checkpoint['trainer'].model
            print("model weights loaded!")
        else:
            raise ValueError("Model weights not exists!!")
        return self.trainer

    def save_checkpoint(self, trainer, parameter_controller, path):
        self.checkpoint['trainer'] = trainer
        self.checkpoint['param_control'] = parameter_controller

        print("Saving checkpoint.")
        path = os.path.join(path, "checkpoint.pkl")
        save_to_pickle(path, self.checkpoint, replace=True)
        print("Checkpoint saved.")


class Checkpointer(GenericCheckpointer):
    def __init__(self, path, trainer, parameter_controller, resume):
        super().__init__(path, trainer, parameter_controller, resume)
        self.columns = []

    def save_log_to_csv(self, epoch=None, mean_train_record=None, mean_validate_record=None, test_record=None):

        if epoch is not None:
            num_layers_to_update = len(self.trainer.optimizer.param_groups[0]['params'])
            csv_records = [time.time(), epoch, int(self.trainer.best_epoch_info['epoch']), num_layers_to_update,
                           self.trainer.optimizer.param_groups[0]['lr'], self.trainer.train_losses[-1], self.trainer.validate_losses[-1],
                           mean_train_record['f1_score_average'], mean_train_record['f1_score']['neutral'], mean_train_record['f1_score']['anger'],mean_train_record['f1_score']['disgust'], mean_train_record['f1_score']['fear'], mean_train_record['f1_score']['happiness'], mean_train_record['f1_score']['sadness'], mean_train_record['f1_score']['surprise'], mean_train_record['f1_score']['other'],
                           mean_validate_record['f1_score_average'], mean_validate_record['f1_score']['neutral'], mean_validate_record['f1_score']['anger'],mean_validate_record['f1_score']['disgust'], mean_validate_record['f1_score']['fear'], mean_validate_record['f1_score']['happiness'], mean_validate_record['f1_score']['sadness'], mean_validate_record['f1_score']['surprise'], mean_validate_record['f1_score']['other']
                         ]
        else:
            csv_records = ["Test results:", "rmse: ", test_record['rmse'],
                           "pcc: ", test_record['pcc'][0], test_record['pcc'][1],
                           "ccc: ", test_record['ccc']]

        row_df = pd.DataFrame(data=csv_records)
        row_df.T.to_csv(self.trainer.csv_filename, mode='a', index=False, header=False)

    def init_csv_logger(self, args, config):

        self.trainer.csv_filename = os.path.join(self.trainer.save_path, "training_logs.csv")

        # Record the arguments.
        arguments_dict = vars(args)
        arguments_dict = pd.json_normalize(arguments_dict, sep='_')

        df_args = pd.DataFrame(data=arguments_dict)
        df_args.to_csv(self.trainer.csv_filename, index=False)

        config = pd.json_normalize(config, sep='_')
        df_config = pd.DataFrame(data=config)
        df_config.to_csv(self.trainer.csv_filename, mode='a', index=False)

        self.columns = ['time', 'epoch', 'best_epoch', 'layer_to_update', 'lr',
                        'tr_loss', 'val_loss', 
                        'tr_f1','tr_neutral', 'tr_anger', 'tr_disgust', 'tr_fear', 'tr_happiness', 'tr_sadness', 'tr_surprise', 'tr_other',
                        'val_f1','tr_neutral', 'val_anger', 'val_disgust', 'val_fear', 'val_happiness', 'val_sadness', 'val_surprise', 'val_other',
                        ]

        df = pd.DataFrame(columns=self.columns)
        df.to_csv(self.trainer.csv_filename, mode='a', index=False)
        
    def save_config_to_json(self, args):
        config = vars(args)
        config_path = os.path.join(self.trainer.save_path, 'config.json')

        with open(config_path, 'w') as config_file:
            json.dump(config, config_file, indent=4)
        sys.path.insert(0, args.python_package_path)
        
    def save_confusion_matrix(self, train_conf_matrix, val_conf_matrix, train_conf_matrix_percent, val_conf_matrix_percent):
        save_path_train = os.path.join(self.trainer.save_path, 'conf_matrix_train.csv')
        save_path_val = os.path.join(self.trainer.save_path, 'conf_matrix_val.csv')
        save_path_train_percent = os.path.join(self.trainer.save_path, 'conf_matrix_train_percent.csv')
        save_path_val_percent = os.path.join(self.trainer.save_path, 'conf_matrix_val_percent.csv')
        
        df_conf_matrix_train = pd.DataFrame(train_conf_matrix, 
                              columns=['Predicted neutral', 'Predicted anger', 'Predicted disgust', 'Predicted fear', 'Predicted happiness', 'Predicted sadness', 'Predicted surprise', 'Predicted other', 'Total'],
                              index=['Actual neutral', 'Actual anger', 'Actual disgust', 'Actual fear', 'Actual happiness', 'Actual sadness', 'Actual surprise', 'Actual other', 'Total'],)
        
        df_conf_matrix_val = pd.DataFrame(val_conf_matrix, 
                              columns=['Predicted neutral', 'Predicted anger', 'Predicted disgust', 'Predicted fear', 'Predicted happiness', 'Predicted sadness', 'Predicted surprise', 'Predicted other', 'Total'],
                              index=['Actual neutral', 'Actual anger', 'Actual disgust', 'Actual fear', 'Actual happiness', 'Actual sadness', 'Actual surprise', 'Actual other', 'Total'],)
        df_conf_matrix_train_percent = pd.DataFrame(train_conf_matrix_percent, 
                              columns=['Predicted neutral', 'Predicted anger', 'Predicted disgust', 'Predicted fear', 'Predicted happiness', 'Predicted sadness', 'Predicted surprise', 'Predicted other'],
                              index=['Actual neutral', 'Actual anger', 'Actual disgust', 'Actual fear', 'Actual happiness', 'Actual sadness', 'Actual surprise', 'Actual other'],)
        
        df_conf_matrix_val_percent = pd.DataFrame(val_conf_matrix_percent, 
                              columns=['Predicted neutral', 'Predicted anger', 'Predicted disgust', 'Predicted fear', 'Predicted happiness', 'Predicted sadness', 'Predicted surprise', 'Predicted other'],
                              index=['Actual neutral', 'Actual anger', 'Actual disgust', 'Actual fear', 'Actual happiness', 'Actual sadness', 'Actual surprise', 'Actual other'],)
        # Save as CSV
        df_conf_matrix_train.to_csv(save_path_train, index=True)
        df_conf_matrix_val.to_csv(save_path_val, index=True)      
        
        df_conf_matrix_train_percent.to_csv(save_path_train_percent, index=True)
        df_conf_matrix_val_percent.to_csv(save_path_val_percent, index=True)      


class ClassificationCheckpointer(GenericCheckpointer):
    r"""
    Write training logs into csv files.
    """
    def __init__(self, path, trainer, parameter_controller, resume):
        super().__init__(path, trainer, parameter_controller, resume)
        self.columns = []

    def save_log_to_csv(self, epoch=None):
        np.set_printoptions(suppress=True)
        num_layers_to_update = len(self.trainer.optimizer.param_groups[0]['params'])

        if epoch is None:
            csv_records = ["Test results: ", "accuracy: ", self.trainer.test_accuracy, "kappa: ", self.trainer.test_kappa, "conf_mat: ", self.trainer.test_confusion_matrix]
        else:
            csv_records = [time.time(), epoch, int(self.trainer.best_epoch_info['epoch']), num_layers_to_update,
                           self.trainer.optimizer.param_groups[0]['lr'], self.trainer.train_losses[-1],
                           self.trainer.validate_losses[-1], self.trainer.train_accuracies[-1], self.trainer.validate_accuracies[-1],
                           self.trainer.train_kappas[-1], self.trainer.validate_kappas[-1],
                           self.trainer.train_confusion_matrices[-1], self.trainer.validate_confusion_matrices[-1]]

        row_df = pd.DataFrame(data=csv_records)
        row_df.T.to_csv(self.trainer.csv_filename, mode='a', index=False, header=False)
        np.set_printoptions()

    def init_csv_logger(self, args, config):

        self.trainer.csv_filename = os.path.join(self.trainer.save_path, "training_logs.csv")

        # Record the arguments.
        arguments_dict = vars(args)
        self.print_dict(arguments_dict)
        self.print_dict(config)

        self.columns = ['time', 'epoch', 'best_epoch', 'layer_to_update', 'lr',
                        'tr_loss', 'val_loss', 'tr_acc', 'val_acc', 'tr_kappa', 'val_kappa', 'tr_conf_mat', 'val_conf_mat']

        df = pd.DataFrame(columns=self.columns)
        df.to_csv(self.trainer.csv_filename, mode='a', index=False)

    def print_dict(self, data_dict):
        for key, value in data_dict.items():
            csv_records = [str(key) + " = " + str(value)]
            row_df = pd.DataFrame(data=csv_records)
            row_df.T.to_csv(self.trainer.csv_filename, mode='a', index=False, header=False, sep=' ')

