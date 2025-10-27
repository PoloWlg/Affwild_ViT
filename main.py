import sys
import argparse
import torch
from torch import cuda
import json
import os
import wandb

# Set the number of threads for OpenMP (used by libraries like NumPy, SciPy)
os.environ["OMP_NUM_THREADS"] = "90"

# Set the number of threads for MKL (used by NumPy, etc.)
os.environ["MKL_NUM_THREADS"] = "90"

if __name__ == '__main__':
    frame_size = 48
    crop_size = 40

    parser = argparse.ArgumentParser(description='Say hello')
    
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Experiment Setting
    # 1.1. Server
    parser.add_argument('-gpu', default=2, type=int, help='Which gpu to use?')
    parser.add_argument('-cpu', default=5, type=int, help='How many threads are allowed?')
    parser.add_argument('-high_performance_cluster', default=1, type=int, help='On high-performance server or not?'
                                                                               'If set to 1, then the gpu and cpu settings will be ignored.'
                                                                               'It should be set to 1 if the user has no right to specify cpu and gpu usage, '
                                                                               'e.g., on Google Colab or NSCC.')

    with open(os.path.join(current_dir, 'path.json'), 'r') as file:
        paths = json.load(file)
    # 1.2. Paths
    parser.add_argument('-dataset_path', default=paths['dataset_path'], type=str,
                        help='The root directory of the preprocessed dataset.')  # /scratch/users/ntu/su012/dataset/mahnob
    parser.add_argument('-dataset_path2', default=paths['dataset_path2'], type=str,
                        help='The root directory of the preprocessed dataset2.') 
    parser.add_argument('-load_path', default=paths['load_path'], type=str,
                        help='The path to load the trained models, such as the backbone.')  # /scratch/users/ntu/su012/pretrained_model
    parser.add_argument('-save_path', default=paths['save_path'], type=str,
                        help='The path to save the trained models ')  # /scratch/users/ntu/su012/trained_model
    parser.add_argument('-python_package_path', default=paths['python_package_path'], type=str,
                        help='The path to the entire repository.')
    parser.add_argument('-context_path', default='/projets2/AS84330/Datasets/Abaw6_EXPR/context', type=str,
                        help='The root directory of the preprocessed context dataset.')
    parser.add_argument('-semantic_context_path', default=paths['semantic_context_path'], type=str,
                        help='Path of the semantic context')

    # 1.3. Experiment name, and stamp, will be used to name the output files.
    # Stamp is used to add a string to the outpout filename, so that instances with different setting will not overwride.
    parser.add_argument('-experiment_name', default="test", type=str, help='The experiment name.')
    parser.add_argument('-stamp', default='', type=str, help='To indicate different experiment instances')

    # 1.4. Load checkpoint or not?
    parser.add_argument('-resume', default=0, type=int, help='Resume from checkpoint?')
    
    # Load Weights or not ?
    parser.add_argument('-load_weights', default='',type=str, help='Path of the weights to load')
    parser.add_argument('-load_weights_res50', default='', type=str, help='Path of the weights to load')

    # 1.5. Debug or not?
    parser.add_argument('-debug', default=0, type=int, help='The number of trials to load for debugging. Set to 0 for non-debugging execution.')

    # 1.6. What modality to use?
    #  Set to ['frame'] for unimodal and ['frame', 'mfcc', 'vggish' for multimodal. Using other features may cause bugs.
    # parser.add_argument('-modality', default=['video', 'logmel','mfcc', "VA_continuous_label"], nargs="*")
    # parser.add_argument('-modality', default=['video', 'vggish', "VA_continuous_label"], nargs="*")
    parser.add_argument('-modality', default=["EXPR_continuous_label", "video", "context"], nargs="*")
    
    # Context
    parser.add_argument('-context_feature_model', default="qwen3", type=str)
    
    # Calculate mean and std for each modality?
    parser.add_argument('-calc_mean_std', default=0,  type=int,
                        help='Calculate the mean and std and save to a pickle file')

    # 1.7. What emotion to train?
    # If choose both, then the multi-headed will be automatically enabled, meaning, the models will predict both the Valence
    #   and Arousal.
    # If choose valence or arousal, the output dimension can be 1 for single-headed, or 2 for multi-headed.
    # For the latter, a weight will be applied to the output to favor the selected emotion.
    parser.add_argument('-emotion', default="valence",
                        help='The emotion dimension to focus when updating gradient: arousal, valence, both, expr')

    # 1.8. Whether to save the models?
    parser.add_argument('-save_model', default=1, type=int, help='Whether to save the models?')

    # 2. Training settings.
    parser.add_argument('-num_heads', default=2, type=int)
    parser.add_argument('-modal_dim', default=32, type=int)
    parser.add_argument('-tcn_kernel_size', default=5, type=int,
                        help='The size of the 1D kernel for temporal convolutional networks.')

    # 2.1. Overall settings
    parser.add_argument('-model_name', default="Proposed", help='LFAN, CAN, CAN2, Video_only, Proposed')
    parser.add_argument('-fusion_method', default="Video_only", help='concat, attention, proposed1, proposed2_orthogonal')
    parser.add_argument('-frozen_resnet50',type=int, default=1, help='True for frozen False for unfrozen')
    parser.add_argument('-compute_att_maps',type=int, default=0, help='Computing attention maps')
    parser.add_argument('-compute_tam',type=int, default=1, help='Computing TAM module (compute attention maps must be set to 1)')

    parser.add_argument('-weighted_ce_loss',type=int, default=1, help='Weighting cross entropy loss or not')


    parser.add_argument('-cross_validation', default=1, type=int)
    parser.add_argument('-num_folds', default=6, type=int)
    parser.add_argument('-folds_to_run', default=[0], nargs="+", type=int, help='Which fold(s) to run? Each fold may take 1-2 days.')

    # 2.2. Epochs and data
    parser.add_argument('-num_epochs', default=20, type=int, help='The total of epochs to run during training.')
    parser.add_argument('-min_num_epochs', default=1, type=int, help='The minimum epoch to run at least.')
    parser.add_argument('-early_stopping', default=50, type=int,
                        help='If no improvement, the number of epoch to run before halting the training')
    parser.add_argument('-window_length', default=300, type=int, help='The length in point number to windowing the data.')
    parser.add_argument('-hop_length', default=200, type=int, help='The step size or stride to move the window.')
    parser.add_argument('-batch_size', default=2, type=int)

    # 2.1. Scheduler and Parameter Control
    parser.add_argument('-seed', default=4, type=int)
    parser.add_argument('-scheduler', default='plateau', type=str, help='plateau, cosine')
    parser.add_argument('-learning_rate', default=1e-5, type=float, help='The initial learning rate.')
    parser.add_argument('-fixed_lr', default=True, type=bool, help='Whether or not to fix the learning rate ')
    parser.add_argument('-min_learning_rate', default=1.e-7, type=float, help='The minimum learning rate.')
    parser.add_argument('-patience', default=2, type=int, help='Patience for learning rate changes.')
    parser.add_argument('-factor', default=0.1, type=float, help='The multiplier to decrease the learning rate.')
    parser.add_argument('-gradual_release', default=0, type=int, help='Whether to gradually release some layers?')
    parser.add_argument('-release_count', default=3, type=int, help='How many layer groups to release?')
    parser.add_argument('-milestone', default=[0], nargs="+", type=int, help='The specific epochs to do something.')
    parser.add_argument('-load_best_at_each_epoch', default=0, type=int,
                        help='Whether to load the best models state at the end of each epoch?')


    # 2.2. Groundtruth settings
    parser.add_argument('-time_delay', default=0, type=float,
                        help='For time_delay=n, it means the n-th label points will be taken as the 1st, and the following ones will be shifted accordingly.'
                             'The rear point will be duplicated to meet the original length.'
                             'This is used to compensate the human labeling delay.')
    # parser.add_argument('-metrics', default=["rmse", "pcc", "ccc"], nargs="*", help='The evaluation metrics.')
    parser.add_argument('-metrics', default=["ccc"], nargs="*", help='The evaluation metrics.')
    parser.add_argument('-save_plot', default=0, type=int,
                        help='Whether to plot the session-wise output/target or not?')

    #  2.3. Logs 
    parser.add_argument('-save_feature_maps', default=0, type=int,
                    help='Whether to save features umaps at 3 of the location ?')
    parser.add_argument('-save_tsne_pcc_inter_connexions', default=0, type=int,
                    help='Whether to save features tsne before and after TAM ?')
    
    parser.add_argument('-unfreeze_all_clip', default=0, type=int,
                    help='')
    args = parser.parse_args()
    

    from experiment import Experiment
    cuda.set_device(args.gpu)

    
    exp = Experiment(args)
    exp.prepare()
    print('RUNNING EXPERIMENT...')
    exp.run()