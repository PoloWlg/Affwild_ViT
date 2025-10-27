from base.experiment import GenericExperiment
from base.utils import load_pickle
from base.loss_function import CCCLoss
from trainer import Trainer

from dataset import DataArranger, Dataset
from base.checkpointer import Checkpointer
from models.model import LFAN, CAN
from models.model_proposed import CAN2, Video_only, Proposed

from base.parameter_control import ResnetParamControl

import torch
import os
import wandb


def check_bool(argument):
    if type(argument) == bool:
        return argument
    else:
        if argument.lower() == 'true':
            return True
        if argument.lower() == 'false':
            return False
        else:
            raise ValueError('Value not valid')

class Experiment(GenericExperiment):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.release_count = args.release_count
        self.gradual_release = args.gradual_release
        self.milestone = args.milestone
        self.backbone_mode = "ir"
        self.min_num_epochs = args.min_num_epochs
        self.num_epochs = args.num_epochs
        self.early_stopping = args.early_stopping
        self.load_best_at_each_epoch = args.load_best_at_each_epoch
        self.fixed_lr = args.fixed_lr
        self.load_weights = args.load_weights
        self.load_weights_res50 = args.load_weights_res50
        
        self.num_heads = args.num_heads
        self.modal_dim = args.modal_dim
        self.tcn_kernel_size = args.tcn_kernel_size
        
        self.semantic_context_path = args.semantic_context_path
        self.frozen_resnet50 = args.frozen_resnet50
        self.compute_att_maps = args.compute_att_maps
        self.weighted_ce_loss = args.weighted_ce_loss
        
        self.context_feature_model = args.context_feature_model
        
        self.save_feature_maps = args.save_feature_maps
        self.save_tsne_pcc_inter_connexions = args.save_tsne_pcc_inter_connexions
        self.unfreeze_all_clip = args.unfreeze_all_clip
        

    
    def prepare(self):
        self.config = self.get_config()

        self.feature_dimension = self.get_feature_dimension(self.config)
        self.multiplier = self.get_multiplier(self.config)
        self.time_delay = self.get_time_delay(self.config)

        
        self.get_modality()
        self.continuous_label_dim = self.get_selected_continuous_label_dim()

        self.dataset_info = load_pickle(os.path.join(self.dataset_path, "dataset_info.pkl"))
        self.data_arranger = self.init_data_arranger()
        if self.calc_mean_std:
            self.calc_mean_std_fn()
        self.mean_std_dict = load_pickle(os.path.join(self.dataset_path, "mean_std_info.pkl"))

    def init_data_arranger(self):
        arranger = DataArranger(self.dataset_info, self.dataset_path, self.debug)
        return arranger

    def run(self):

        # criterion = CCCLoss()
        criterion = torch.nn.CrossEntropyLoss()
        if self.weighted_ce_loss:
            class_weights = 1 / torch.tensor([179503, 17153, 10978, 9110, 94344, 81054, 30639, 171407]).to(self.device)
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

        for fold in iter(self.folds_to_run):
            import gc

            # Clear unused GPU memory
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.ipc_collect()
            
            wandb.init(
                project=f"Finetune", 
                name=f"lr_{self.args.learning_rate}-bs_{self.args.batch_size}-self.args.unfreeze_all_clip_{self.args.unfreeze_all_clip}",
                config={
                    "gpu": self.args.gpu,
                    "epochs": self.args.num_epochs,
                    "batch_size": self.args.batch_size,
                    "learning_rate": self.args.learning_rate
                })

            save_path = os.path.join(self.save_path,
                                     self.experiment_name + "_" + self.model_name + "_" + self.stamp + "_fold" + str(
                                         fold) + "_" + self.emotion +  "_seed" + str(self.seed))
            self.save_path = save_path
            self.args.save_path = save_path
            
            if not self.resume:
                if self.experiment_name == 'test':
                    os.makedirs(save_path, exist_ok=True)
                else:
                    os.makedirs(save_path, exist_ok=False)
            
            checkpoint_filename = os.path.join(save_path, "checkpoint.pkl")

            model = self.init_model()

            dataloaders = self.init_dataloader(fold)

            trainer_kwards = {'device': self.device, 'emotion': self.emotion, 'model_name': self.model_name,
                              'models': model, 'save_path': save_path, 'fold': fold,
                              'min_epoch': self.min_num_epochs, 'max_epoch': self.num_epochs,
                              'early_stopping': self.early_stopping, 'scheduler': self.scheduler,
                              'learning_rate': self.learning_rate, 'fixed_lr':self.fixed_lr,'compute_att_maps':self.compute_att_maps,  'min_learning_rate': self.min_learning_rate,
                              'patience': self.patience, 'batch_size': self.batch_size,
                              'criterion': criterion, 'factor': self.factor, 'verbose': True,
                              'milestone': self.milestone, 'metrics': self.config['metrics'],
                              'load_best_at_each_epoch': self.load_best_at_each_epoch,
                              'save_plot': self.config['save_plot'], 'load_weights': self.load_weights, 'load_weights_res50': self.load_weights_res50, 'save_feature_maps': self.save_feature_maps,}

            trainer = Trainer(**trainer_kwards)

            parameter_controller = ResnetParamControl(trainer, gradual_release=self.gradual_release,
                                                      release_count=self.release_count,
                                                      backbone_mode=["visual", "audio"])

            checkpoint_controller = Checkpointer(checkpoint_filename, trainer, parameter_controller, resume=self.resume)

            if self.resume:
                trainer, parameter_controller = checkpoint_controller.load_checkpoint()
                trainer.model.to(self.device)
                trainer.device = torch.device('cuda')
                trainer.optimizer = torch.optim.Adam(trainer.get_parameters(), lr=trainer.learning_rate, weight_decay=0.001)
                
                # for param_group in trainer.optimizer.param_groups:
                #     for param in param_group['params']:
                #         print(param.device)
            else:
                # if self.load_weights:
                    # checkpoint_filename = self.load_weights
                    # checkpoint_controller = Checkpointer(checkpoint_filename, trainer, parameter_controller, resume=self.resume)
                    # trainer, _ = checkpoint_controller.load_checkpoint()
                    # trainer.save_path = save_path
                    # trainer.fit_finished = False
                    # trainer.start_epoch = 0
                    # trainer.model.to(self.device)
                    # trainer.device = torch.device('cuda')
                    # trainer.optimizer = torch.optim.Adam(trainer.get_parameters(), lr=trainer.learning_rate, weight_decay=0.001)
                checkpoint_controller.init_csv_logger(self.args, self.config)
                checkpoint_controller.save_config_to_json(self.args)

            if not trainer.fit_finished or trainer.resume:
                trainer.fit(dataloaders, parameter_controller=parameter_controller,
                            checkpoint_controller=checkpoint_controller)

            test_kwargs = {'dataloader_dict': dataloaders, 'epoch': None, 'partition': 'validate'}
            trainer.test(checkpoint_controller, predict_only=1, **test_kwargs)

    def init_dataset(self, data, continuous_label_dim, mode, fold):
        dataset = Dataset(data, self.args.context_path, continuous_label_dim, self.modality, self.multiplier,
                          self.feature_dimension, self.window_length,
                          mode, mean_std=self.mean_std_dict[fold][mode], time_delay=self.time_delay, context_feature_model= self.context_feature_model)
        return dataset

    def init_model(self):
        self.init_randomness()
        modality = [modal for modal in self.modality if "continuous_label" not in modal]

        if self.model_name == "LFAN":
            model = LFAN(backbone_settings=self.config['backbone_settings'],
                                                   modality=modality, example_length=self.window_length,
                                                   kernel_size=self.tcn_kernel_size,
                                                   tcn_channel=self.config['tcn']['channels'], modal_dim=self.modal_dim, num_heads=self.num_heads,
                                                   root_dir=self.load_path, device=self.device)
            model.init()
        elif self.model_name == "CAN":
            model = CAN(root_dir=self.load_path, fusion_method=self.args.fusion_method , modalities=modality, tcn_settings=self.config['tcn_settings'], backbone_settings=self.config['backbone_settings'], output_dim=len(self.continuous_label_dim), device=self.device, frozen_resnet50 = self.frozen_resnet50)
        elif self.model_name == "CAN2":
            model = CAN2(root_dir=self.load_path, fusion_method=self.args.fusion_method , modalities=modality, tcn_settings=self.config['tcn_settings'], backbone_settings=self.config['backbone_settings'], output_dim=len(self.continuous_label_dim), device=self.device, semantic_context_path=self.semantic_context_path, compute_att_maps=self.compute_att_maps, frozen_resnet50 = self.frozen_resnet50, args = self.args)
        elif self.model_name == "Video_only":    
            model = Video_only(root_dir=self.load_path, device=self.device, backbone_settings=self.config['backbone_settings'], frozen_resnet50=self.frozen_resnet50)
        elif self.model_name == "Proposed":    
            model = Proposed(root_dir=self.load_path, device=self.device, backbone_settings=self.config['backbone_settings'], frozen_resnet50=self.frozen_resnet50, args=self.args)
    
        return model

    def get_modality(self):
        pass

    def get_config(self):
        from configs import config
        return config

    def get_selected_continuous_label_dim(self):
        if self.emotion == "arousal":
            dim = [1]
        elif self.emotion == "valence":
            dim = [0]
        elif self.emotion == "expr":
            dim = [0, 1, 2, 3, 4, 5, 6, 7]
            
        else:
            raise ValueError("Unknown emotion!")
        return dim
