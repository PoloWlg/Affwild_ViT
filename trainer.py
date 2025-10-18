from base.trainer import GenericVideoTrainer
from base.scheduler import GradualWarmupScheduler, MyWarmupScheduler

from torch import optim
import torch

import time
import copy
import os
import wandb

from pprint import pprint

import numpy as np


class Trainer(GenericVideoTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.best_epoch_info = {
            'model_weights': copy.deepcopy(self.model.state_dict()),
            'loss': 1e10,
            'f1_score_average': -1e10,
            'acc': -1,
            'p_r_f1': 0,
            'kappa': 0,
            'epoch': 0,
            'metrics': {
                'train_loss': -1,
                'val_loss': -1,
                'train_acc': -1,
                'val_acc': -1,
            }
        }

    def init_optimizer_and_scheduler(self, epoch=0):
        self.optimizer = optim.AdamW(self.get_parameters(), lr=self.learning_rate, weight_decay=0.1)


        self.scheduler = MyWarmupScheduler(
            optimizer=self.optimizer, lr = self.learning_rate, min_lr=self.min_learning_rate,
            best=self.best_epoch_info['f1_score_average'], mode="max", patience=self.patience,
            factor=self.factor, num_warmup_epoch=self.min_epoch, init_epoch=epoch)

    def fit(self, dataloader_dict, checkpoint_controller, parameter_controller):

        if self.verbose:
            print("------")
            print("Starting training, on device:", self.device)

        self.time_fit_start = time.time()
        start_epoch = self.start_epoch

        if self.best_epoch_info is None:
            self.best_epoch_info = {
                'model_weights': copy.deepcopy(self.model.state_dict()),
                'loss': 1e10,
                'ccc': -1e10
            }
            

        for epoch in np.arange(start_epoch, self.max_epoch):
            if not self.resume:
                if self.fit_finished:
                    if self.verbose:
                        print("\nEarly Stop!\n")
                    break

            improvement = False

            if epoch in self.milestone or (parameter_controller.get_current_lr() < self.min_learning_rate and epoch >= self.min_epoch and self.scheduler.relative_epoch > self.min_epoch):
                parameter_controller.release_param(self.model.spatial, epoch)
                
                if not parameter_controller.gradual_release: 
                    parameter_controller.trainer.init_optimizer_and_scheduler(epoch=epoch)
                    
                if parameter_controller.early_stop:
                    break

                # self.model.load_state_dict(self.best_epoch_info['model_weights'])

            if epoch == 0 and self.load_weights: 
                print('load weights for the model ...')
                self.model.load_state_dict(torch.load(self.load_weights, map_location=self.device))
                
            if epoch == 0 and self.load_weights_res50: 
                print('load weights for the resnet50 model ...')
                checkpoint = torch.load(self.load_weights_res50, map_location=self.device)
                filtered_state_dict = {k.replace('spatial.', ''): v for k, v in checkpoint.items() if k.startswith('spatial.')}
                self.model.spatial.load_state_dict(filtered_state_dict)
            time_epoch_start = time.time()

            if self.verbose:
                print("There are {} layers to update.".format(len(self.optimizer.param_groups[0]['params'])))

            # Get the losses and the record dictionaries for training and validation.
            train_kwargs = {"dataloader_dict": dataloader_dict, "epoch": epoch}
            train_loss, train_record_dict = self.train(**train_kwargs)

            validate_kwargs = {"dataloader_dict": dataloader_dict, "epoch": epoch}
            validate_loss, validate_record_dict = self.validate(**validate_kwargs)

            # if epoch % 1 == 0:
            #     test_kwargs = {"dataloader_dict": dataloader_dict, "epoch": None, "train_mode": 0}
            #     validate_loss, test_record_dict = self.test(checkpoint_controller=checkpoint_controller, feature_extraction=0, **test_kwargs)
            #     print(test_record_dict['overall']['ccc'])

            if validate_loss < 0:
                raise ValueError('validate loss negative')

            self.train_losses.append(train_loss)
            self.validate_losses.append(validate_loss)

            validate_ccc = validate_record_dict['overall']['f1_score_average']

            self.scheduler.best = self.best_epoch_info['f1_score_average']


            if validate_ccc > self.best_epoch_info['f1_score_average']:
                torch.save(self.model.state_dict(), os.path.join(self.save_path, "model_state_dict" + str(validate_ccc) + ".pth"))

                improvement = True
                self.best_epoch_info = {
                    'model_weights': copy.deepcopy(self.model.state_dict()),
                    'loss': validate_loss,
                    'f1_score_average': validate_ccc,
                    'epoch': epoch,
                }
                
                # Save confusion matrix at the best epoch
            checkpoint_controller.save_confusion_matrix(
                train_record_dict['overall']['class_distribution'], validate_record_dict['overall']['class_distribution'],
                train_record_dict['overall']['class_distribution_percent'], validate_record_dict['overall']['class_distribution_percent']
                )

            if self.verbose:
                print(
                    "\n Fold {:2} Epoch {:2} in {:.0f}s || Train loss={:.3f} | Val loss={:.3f} | LR={:.1e} | Release_count={} | best={} | "
                    "improvement={}-{}".format(
                        self.fold,
                        epoch + 1,
                        time.time() - time_epoch_start,
                        train_loss,
                        validate_loss,
                        self.optimizer.param_groups[0]['lr'],
                        parameter_controller.release_count,
                        int(self.best_epoch_info['epoch']) + 1,
                        improvement,
                        self.early_stopping_counter))
                

                print('--- Train ---')
                print('f1 score average: ', train_record_dict['overall']['f1_score_average'])
                pprint(train_record_dict['overall']['f1_score'])
                print('--- Validate ---')
                print('f1 score average: ', validate_record_dict['overall']['f1_score_average'])
                pprint(validate_record_dict['overall']['f1_score'])
                print("------")
                
                wandb.log({
                    "train_accuracy": train_record_dict['overall']['f1_score_average'],
                    "val_accuracy": validate_record_dict['overall']['f1_score_average'],
                    "epoch": epoch
                })

            checkpoint_controller.save_log_to_csv(
                epoch, train_record_dict['overall'], validate_record_dict['overall'])
            

            # Early stopping controller.
            if self.early_stopping and self.scheduler.relative_epoch > self.min_epoch:
                if improvement:
                    self.early_stopping_counter = self.early_stopping
                else:
                    self.early_stopping_counter -= 1

                if self.early_stopping_counter <= 0:
                    self.fit_finished = True


            self.scheduler.step(metrics=validate_ccc, epoch=epoch)


            self.start_epoch = epoch + 1

            if self.load_best_at_each_epoch:
                self.model.load_state_dict(self.best_epoch_info['model_weights'])

            checkpoint_controller.save_checkpoint(self, parameter_controller, self.save_path)

        self.fit_finished = True
        checkpoint_controller.save_checkpoint(self, parameter_controller, self.save_path)

        self.model.load_state_dict(self.best_epoch_info['model_weights'])
