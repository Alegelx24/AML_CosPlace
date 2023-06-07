
# Taking Cosplace to the Next Level: Improvements to the Visual-Geolocalization model

This is the pyTorch implementation of Advanced Machine Learning course project at Politecnico di Torino. It consists in an improvement of the existing Cosplace visual-geolocalization model [[ArXiv](https://arxiv.org/abs/2204.02287)]. You can find the corresponding paper here.

## Authors
The authors of this project are:
- Gelsi Alessandro S303525 - [Github](https://github.com/Alegelx24)
- Ferraro Luca S301843 - [Github](https://github.com/LucaFerraro00)

## Link to Datasets 

- [SF-XS](https://drive.google.com/file/d/1tQqEyt3go3vMh4fj_LZrRcahoTbzzH-y/view?usp=drive_link)
- [Tokyo-XS](https://drive.google.com/file/d/15QB3VNKj93027UAQWv7pzFQO1JDCdZj2/view?usp=drive_link) 
- [Tokyo-night](https://drive.google.com/drive/folders/1ji55oNPm8wyQe86kereDDoFG4gtgjCRL?usp=sharing) 

## How to run the code: Train

- After downloading the dataset, to train the model you can simply run 

  `$ python train.py --dataset_folder path/to/sf-xs`

- To set groups number and epochs number: 

  `$ python train.py --dataset_folder path/to/sf-xs --groups_num 1 --epochs_num 3`

- To change the backbone for example you can simply run 

  `$ python train.py --dataset_folder path/to/sf-xs --backbone efficientnet_v2_s --groups_num 1 --epochs_num 3`

- To change the loss function you can simply run 

  `$ python train.py --dataset_folder path/to/sf-xs --loss arcface --groups_num 1 --epochs_num 3`

- To perform the improved preprocessing (RandomPerspective+AdjustGamma) you can simply run 

  `$ python train.py --dataset_folder path/to/sf-xs --preprocessing --groups_num 1 --epochs_num 3`

- Gamma preprocessing value is settable simply running 

  `$ python train.py --dataset_folder path/to/sf-xs --preprocessing --gamma 1.5 --groups_num 1 --epochs_num 3`

- To perform the improved preprocessing (RandomPerspective+AdjustGamma) you can simply run 

  `$ python train.py --dataset_folder path/to/sf-xs --preprocessing --groups_num 1 --epochs_num 3`

- You can perform domain adaptation through grl running

  `$ python train.py --dataset_folder path/to/sf-xs --groups_num 1 --epochs_num 3  --grl --dataset_root path/to/target_dataset --grl_datasets target_dataset_subfolder_name `

- You can also speed up your training with Automatic Mixed Precision 

  `$ python train.py --dataset_folder path/to/sf-xs/processed --use_amp16`

Run `$ python train.py -h` to have a look at all the hyperparameters that you can change. You will find all hyperparameters mentioned in the paper.

## How to run the code: Test

- You can test a trained model with

  `$ python eval.py --dataset_folder path/to/sf-xs --backbone efficientnet_v2_s  --resume_model path/to/best_model.pth`

- You can test a trained model using GeoWarp as reranking method with the following script. To set custom encoder and pooling you should  manually change the paths inside test.py file.

  `$ python eval.py --dataset_folder path/to/sf-xs --groups_num 1  --epochs_num 3 --backbone efficientnet_v2_s --resume_model path/to/best_model.pth --warping_module  --num_reranked_predictions 5`

- You can test a trained model using Model soup as model ensemble method with the following script. The path folder containing .pth files of ingredients is defined inside model_soup.py files. 

  `$ python eval.py --dataset_folder path/to/sf-xs  --groups_num 1  --epochs_num 3   --model_soupe_greedy `

- You can test a trained model performing FDA between images inside test queries and database with the following script. 

  `$ python eval.py --dataset_folder path/to/sf-xs --groups_num 1 --epochs_num 3  --resume_model path/to/best_model.pth --fda --fda_weight 0.0075 `

Run `$ python eval.py -h` to have a look at all the hyperparameters that you can change. You will find all hyperparameters mentioned in the paper.

You can download plenty of trained models below.

## Trained Models

All the trained model are downloadable at this [[link](https://drive.google.com/drive/folders/1mtALaGvLLRjGLgJgfIe7HeCcUaG_YDiQ?usp=sharing)]
