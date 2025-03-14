
import os
import argparse


def parse_arguments(is_training: bool = True):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # CosPlace Groups parameters
    parser.add_argument("--M", type=int, default=10, help="_")
    parser.add_argument("--alpha", type=int, default=30, help="_")
    parser.add_argument("--N", type=int, default=5, help="_")
    parser.add_argument("--L", type=int, default=2, help="_")
    parser.add_argument("--groups_num", type=int, default=8, help="_")
    parser.add_argument("--min_images_per_class", type=int, default=10, help="_")
    # Model parameters
    parser.add_argument("--backbone", type=str, default="ResNet18",
                        choices=["VGG16", "ResNet18", "ResNet50", "ResNet101", "ResNet152","mobilenet_v3_small","mobilenet_v3_large","efficientnet_v2_s","efficientnet_b0","efficientnet_b3","regnet_y_1_6gf", "convnext_small", "swin_t","swin_v2_t","convnext_base", "regnet_y_1_6gf"], help="_")
    parser.add_argument("--fc_output_dim", type=int, default=512,
                        help="Output dimension of final fully connected layer")
    # Training parameters
    parser.add_argument("--use_amp16", action="store_true",
                        help="use Automatic Mixed Precision")
    parser.add_argument("--augmentation_device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="on which device to run data augmentation")
    parser.add_argument("--batch_size", type=int, default=32, help="_")
    parser.add_argument("--epochs_num", type=int, default=50, help="_")
    parser.add_argument("--iterations_per_epoch", type=int, default=10000, help="_")
    parser.add_argument("--lr", type=float, default=0.00001, help="_")
    parser.add_argument("--classifiers_lr", type=float, default=0.01, help="_")
    # Data augmentation
    parser.add_argument("--brightness", type=float, default=0.7, help="_")
    parser.add_argument("--contrast", type=float, default=0.7, help="_")
    parser.add_argument("--hue", type=float, default=0.5, help="_")
    parser.add_argument("--saturation", type=float, default=0.7, help="_")

    # Pre Processing Augmentation
    parser.add_argument("--preprocessing", action="store_true", help="_")
    parser.add_argument("--gamma", type=float, default=0.7, help="_")
    parser.add_argument("--random_resized_crop", type=float, default=0.5, help="_")

    # Validation / test parameters
    parser.add_argument("--infer_batch_size", type=int, default=16,
                        help="Batch size for inference (validating and testing)")
    parser.add_argument("--positive_dist_threshold", type=int, default=25,
                        help="distance in meters for a prediction to be considered a positive")
    # Resume parameters
    parser.add_argument("--resume_train", type=str, default=None,
                        help="path to checkpoint to resume, e.g. logs/.../last_checkpoint.pth")
    parser.add_argument("--resume_model", type=str, default=None,
                        help="path to model to resume, e.g. logs/.../best_model.pth")
    # Other parameters
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"], help="_")
    parser.add_argument("--seed", type=int, default=0, help="_")
    parser.add_argument("--num_workers", type=int, default=8, help="_")
    # Paths parameters
    parser.add_argument("--dataset_folder", type=str, default=None,
                        help="path of the folder with train/val/test sets")
    parser.add_argument("--save_dir", type=str, default="default",
                        help="name of directory on which to save the logs, under logs/save_dir")
    #DOMAIN ADAPTATION
    parser.add_argument("--grl", action="store_true", help="_")
    parser.add_argument("--dataset_root", type=str, default=None, help="_")
    parser.add_argument("--grl_datasets", type=str, default=None, help="_")

    #RERANKING
    parser.add_argument("--warping_module",action="store_true", help="_") 
    parser.add_argument("--num_reranked_predictions", type=int, default=5, help="_")

    #MODEL SOUP
    parser.add_argument("--model_soupe_greedy", action="store_true", help="_") 
    parser.add_argument("--model_soupe_uniform", action="store_true", help="_") 
    
    #FDA
    parser.add_argument("--fda", action="store_true",help="_") 
    parser.add_argument("--fda_weight", type=float, default=0.01,help="_") 


    #LOSSES
    parser.add_argument("--loss", type=str, default="cosface",
                        choices=["cosface","arcface", "sphereface", "elasticface", "new_elasticface"], help="_")


    
    args = parser.parse_args()
    
    if args.dataset_folder is None:
        try:
            args.dataset_folder = os.environ['SF_XL_PROCESSED_FOLDER']
        except KeyError:
            raise Exception("You should set parameter --dataset_folder or export " +
                            "the SF_XL_PROCESSED_FOLDER environment variable as such \n" +
                            "export SF_XL_PROCESSED_FOLDER=/path/to/sf_xl/processed")
    
    if not os.path.exists(args.dataset_folder):
        raise FileNotFoundError(f"Folder {args.dataset_folder} does not exist")
    
    if is_training:
        args.train_set_folder = os.path.join(args.dataset_folder, "train")
        if not os.path.exists(args.train_set_folder):
            raise FileNotFoundError(f"Folder {args.train_set_folder} does not exist")
        
        args.val_set_folder = os.path.join(args.dataset_folder, "val")
        if not os.path.exists(args.val_set_folder):
            raise FileNotFoundError(f"Folder {args.val_set_folder} does not exist")
    
    args.test_set_folder = os.path.join(args.dataset_folder, "test")
    if not os.path.exists(args.test_set_folder):
        raise FileNotFoundError(f"Folder {args.test_set_folder} does not exist")
    
    return args
