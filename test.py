
import faiss
import torch
import logging
import numpy as np
from tqdm import tqdm
from typing import Tuple
from argparse import Namespace
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader, Dataset
import warping_model.warping_dataset as warp_dataset
import warping_model.warp_network as warp_network



# Compute R@1, R@5, R@10, R@20
RECALL_VALUES = [1, 5, 10, 20]
test_batch_size = 8


def test(args: Namespace, eval_ds: Dataset, model: torch.nn.Module, num_reranked_predictions=None, ) -> Tuple[np.ndarray, str]:
    """Compute descriptors of the given dataset and compute the recalls."""
    
    model = model.eval()
    with torch.no_grad():
        logging.debug("Extracting database descriptors for evaluation/testing")
        database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
        database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                        batch_size=args.infer_batch_size, pin_memory=(args.device == "cuda"))
        all_descriptors = np.empty((len(eval_ds), args.fc_output_dim), dtype="float32")
        for images, indices in tqdm(database_dataloader, ncols=100):
            descriptors = model(images.to(args.device))
            descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors
        
        logging.debug("Extracting queries descriptors for evaluation/testing using batch size 1")
        queries_infer_batch_size = 1
        queries_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num, eval_ds.database_num+eval_ds.queries_num)))
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                        batch_size=queries_infer_batch_size, pin_memory=(args.device == "cuda"))
        for images, indices in tqdm(queries_dataloader, ncols=100):
            descriptors = model(images.to(args.device))
            descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors
    
    queries_descriptors = all_descriptors[eval_ds.database_num:]
    database_descriptors = all_descriptors[:eval_ds.database_num]
    
    # Use a kNN to find predictions
    faiss_index = faiss.IndexFlatL2(args.fc_output_dim)
    faiss_index.add(database_descriptors)
    del database_descriptors, all_descriptors
    
    logging.debug("Calculating recalls")
    _, predictions = faiss_index.search(queries_descriptors, max(RECALL_VALUES))


    ##Re-ranking with geo warp
    if args.warping_module:
        reranked_predictions = predictions.copy()
        num_reranked_predictions=args.num_reranked_predictions
        with torch.no_grad():
            for num_q in tqdm(range(eval_ds.queries_num), desc="Testing", ncols=100):
                dot_prods_wqp = np.zeros((num_reranked_predictions))
                query_path = eval_ds.queries_paths[num_q]
                for i1 in range(0, num_reranked_predictions, test_batch_size):
                    batch_indexes = list(range(num_reranked_predictions))[i1:i1+test_batch_size]
                    current_batch_size = len(batch_indexes)
                    query = warp_dataset.open_image_and_apply_transform(query_path)
                    query_repeated_twice = torch.repeat_interleave(query.unsqueeze(0), current_batch_size, 0)
                    
                    preds = []
                    for i in batch_indexes:
                        pred_path = eval_ds.database_paths[predictions[num_q, i]]
                        preds.append(warp_dataset.open_image_and_apply_transform(pred_path))
                    preds = torch.stack(preds)
                    
                    ##LOAD WARP MODEL
                    warp_backbone="alexnet"
                    #choices for warp_backbone=["alexnet", "vgg16", "resnet50"]
                    warp_pooling = "gem" # or warp_pooling= "netvlad"
                    kernel_sizes = [7, 5, 5, 5, 5, 5]         
                    channels=[225, 128, 128, 64, 64, 64, 64]           

                    features_extractor = warp_network.FeaturesExtractor(warp_backbone, warp_pooling)
                   # global_features_dim = warp_dataset.get_output_dim(features_extractor, warp_pooling)
                    homography_regression = warp_network.HomographyRegression(kernel_sizes=kernel_sizes, channels=channels, padding=1)
                    
                    baseline_path= "warping_model/data/pretrained_baselines/alexnet_gem.pth"
                    if baseline_path is not None:
                        state_dict = torch.load(baseline_path)
                        features_extractor.load_state_dict(state_dict)
                        del state_dict
                    else:
                        logging.warning("WARNING: --resume_fe is set to None, meaning that the "
                                        "Feature Extractor is not initialized!")

                    home_reg_path= "warping_model/data/trained_homography_regressions/alexnet_gem.pth"  
                    if home_reg_path is not None:
                        state_dict = torch.load(home_reg_path)
                        homography_regression.load_state_dict(state_dict)
                        del state_dict
                    else:
                        logging.warning("WARNING: --resume_hr is set to None, meaning that the "
                                        "Homography Regression is not initialized!")
                    
                    warp_model = warp_network.Network(features_extractor, homography_regression).cuda().eval()
                    warp_model = torch.nn.DataParallel(warp_model)

                    warped_pair = warp_dataset.compute_warping(warp_model, query_repeated_twice.cuda(), preds.cuda())
                    q_features = warp_model("features_extractor", [warped_pair[0], "local"])
                    p_features = warp_model("features_extractor", [warped_pair[1], "local"])
                    # Sum along all axes except for B. wqp stands for warped query-prediction
                    dot_prod_wqp = (q_features * p_features).sum(list(range(1, len(p_features.shape)))).cpu().numpy()
                    
                    dot_prods_wqp[i1:i1+test_batch_size] = dot_prod_wqp
                
                reranking_indexes = dot_prods_wqp.argsort()[::-1]
                reranked_predictions[num_q, :num_reranked_predictions] = predictions[num_q][reranking_indexes]
        
        predictions = reranked_predictions
    
    #### For each query, check if the predictions are correct
    positives_per_query = eval_ds.get_positives()
    recalls = np.zeros(len(RECALL_VALUES))
    for query_index, preds in enumerate(predictions):
        for i, n in enumerate(RECALL_VALUES):
            if np.any(np.in1d(preds[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                break
    # Divide by queries_num and multiply by 100, so the recalls are in percentages
    recalls = recalls / eval_ds.queries_num * 100
    recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(RECALL_VALUES, recalls)])
    return recalls, recalls_str
