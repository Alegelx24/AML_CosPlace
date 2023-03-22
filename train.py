if __name__ == '__main__' :

    import sys
    import torch
    import logging
    import numpy as np
    from tqdm import tqdm
    import multiprocessing
    from datetime import datetime
    import torchvision.transforms as T
    import test
    import util
    import parser_1
    import commons
    import cosface_loss
    import cosface_loss_ArcFace
    import cosface_loss_SphereFace
    import elastic_face
    import model_soup
    import GradientReversalLayer as GRL
    import augmentations
    from model import network
    from datasets.test_dataset import TestDataset
    from datasets.train_dataset import TrainDataset
    from datasets.grl_dataset import GrlDataset
    from warping_model.warping_dataset import HomographyDataset
    from datasets.dataset_qp import DatasetQP
    import warping_model.warping_dataset as WDS

    from torchvision.transforms.functional import hflip
    from torch.utils.data.dataloader import DataLoader


    torch.backends.cudnn.benchmark = True  # Provides a speedup

    args = parser_1.parse_arguments()
    start_time = datetime.now()
    output_folder = f"logs/{args.save_dir}/{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
    commons.make_deterministic(args.seed)
    commons.setup_logging(output_folder, console="debug")
    logging.info(" ".join(sys.argv))
    logging.info(f"Arguments: {args}")
    logging.info(f"The outputs are being saved in {output_folder}")

    encoder_dim=512

    if args.grl:
        grl_discriminator = GRL.get_discriminator(encoder_dim, len(args.grl_datasets.split("+")))
    else:
        grl_discriminator = None

    #### Model
    model = network.GeoLocalizationNet(args.backbone, args.fc_output_dim, grl_discriminator, attention=args.attention)

    logging.info(f"There are {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs.")

    if args.resume_model is not None:
        logging.debug(f"Loading model from {args.resume_model}")
        model_state_dict = torch.load(args.resume_model)
        model.load_state_dict(model_state_dict)

    model = model.to(args.device).train()

    #### Warping module
    if args.warping_module:

        kernel_sizes= [7, 5, 5, 5, 5, 5]
        channels = [225, 128, 128, 64, 64, 64, 64]
        homography_regression = network.HomographyRegression(kernel_sizes=kernel_sizes, channels=channels, padding=1)
        model = network.GeoLocalizationNet(args.backbone, args.fc_output_dim, grl_discriminator, homography_regression=homography_regression).cuda().eval()
        model = torch.nn.DataParallel(model)

        consistency_w=0.1
        features_wise_w=10
        ss_w=1
        batch_size_consistency=16
        batch_size_features_wise=16
        num_reranked_preds= 5

    
    def to_cuda(list_):
        """Move to cuda all items of the list."""
        return [item.cuda() for item in list_]
    
    def hor_flip(points):
        """Flip points horizontally.
        
        Parameters
        ----------
        points : torch.Tensor of shape [B, 8, 2]
        """
        new_points = torch.zeros_like(points)
        new_points[:, 0::2, :] = points[:, 1::2, :]
        new_points[:, 1::2, :] = points[:, 0::2, :]
        new_points[:, :, 0] *= -1
        return new_points
    
    mse = torch.nn.MSELoss()

    def compute_loss(loss, weight):
        """Compute loss and gradients separately for each loss, and free the
        computational graph to reduce memory consumption.
        """
        loss *= weight
        loss.backward()
        return loss.item()


    #### Optimizer
    criterion = torch.nn.CrossEntropyLoss()
    model_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    #### Datasets
    groups = [TrainDataset(args, args.train_set_folder, M=args.M, alpha=args.alpha, N=args.N, L=args.L,
                        current_group=n, min_images_per_class=args.min_images_per_class) for n in range(args.groups_num)]
    # Each group has its own classifier, which depends on the number of classes in the group
    classifiers = [cosface_loss.MarginCosineProduct(args.fc_output_dim, len(group)) for group in groups]
    #classifiers = [elastic_face.ElasticArcFace(args.fc_output_dim, len(group)) for group in groups]

    #classifiers = [cosface_loss_SphereFace.MarginCosineProduct(args.fc_output_dim, len(group)) for group in groups]
    #classifiers = [cosface_loss_ArcFace.MarginCosineProduct(args.fc_output_dim, len(group)) for group in groups]
    classifiers_optimizers = [torch.optim.Adam(classifier.parameters(), lr=args.classifiers_lr) for classifier in classifiers]

    logging.info(f"Using {len(groups)} groups")
    logging.info(f"The {len(groups)} groups have respectively the following number of classes {[len(g) for g in groups]}")
    logging.info(f"The {len(groups)} groups have respectively the following number of images {[g.get_images_num() for g in groups]}")

    val_ds = TestDataset(args.val_set_folder, positive_dist_threshold=args.positive_dist_threshold)
    test_ds = TestDataset(args.test_set_folder, queries_folder="queries_v1",
                        positive_dist_threshold=args.positive_dist_threshold)
    logging.info(f"Validation set: {val_ds}")
    logging.info(f"Test set: {test_ds}")

    if args.warping_module:
        train_ds_onegroup=TrainDataset(args, args.train_set_folder, M=args.M, alpha=args.alpha, N=args.N, L=args.L,
                        current_group=0, min_images_per_class=args.min_images_per_class)
        ss_dataset = HomographyDataset(root_path=f"{args.dataset_folder}/train", k=0.6)
        dataset_qp = DatasetQP(model, args.fc_output_dim, groups, qp_threshold=1.2)
        dataloader_qp = commons.InfiniteDataLoader(dataset_qp, shuffle=True,
                                                    batch_size=max(batch_size_consistency=16, batch_size_features_wise=16),
                                                    num_workers=2, pin_memory=True, drop_last=True)
        data_iter_qp = iter(dataloader_qp)


    #### Resume
    if args.resume_train:
        model, model_optimizer, classifiers, classifiers_optimizers, best_val_recall1, start_epoch_num = \
            util.resume_train(args, output_folder, model, model_optimizer, classifiers, classifiers_optimizers)
        model = model.to(args.device)
        epoch_num = start_epoch_num - 1
        logging.info(f"Resuming from epoch {start_epoch_num} with best R@1 {best_val_recall1:.1f} from checkpoint {args.resume_train}")
    else:
        best_val_recall1 = start_epoch_num = 0

    #### Train / evaluation loop
    logging.info("Start training ...")
    logging.info(f"There are {len(groups[0])} classes for the first group, " +
                f"each epoch has {args.iterations_per_epoch} iterations " +
                f"with batch_size {args.batch_size}, therefore the model sees each class (on average) " +
                f"{args.iterations_per_epoch * args.batch_size / len(groups[0]):.1f} times per epoch")


    if args.augmentation_device == "cuda":
        if args.preprocessing == True:
            gpu_augmentation = T.Compose([
                    augmentations.DeviceAgnosticColorJitter(brightness=args.brightness,
                                                            contrast=args.contrast,
                                                            saturation=args.saturation,
                                                            hue=args.hue),
                    augmentations.DeviceAgnosticRandomResizedCrop([512, 512],
                                                                scale=[1-args.random_resized_crop, 1]),
                   augmentations.DeviceAgnosticRandomPerspective(),
                   augmentations.DeviceAgnosticAdjustGamma(gamma=args.gamma, gain=1.2),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
        else: 
             gpu_augmentation = T.Compose([
                    augmentations.DeviceAgnosticColorJitter(brightness=args.brightness,
                                                            contrast=args.contrast,
                                                            saturation=args.saturation,
                                                            hue=args.hue),
                    augmentations.DeviceAgnosticRandomResizedCrop([512, 512],
                                                                scale=[1-args.random_resized_crop, 1]),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

    if args.use_amp16:
        scaler = torch.cuda.amp.GradScaler()

    if args.grl:   
        grl_dataset = GrlDataset(args.dataset_root, args.grl_datasets.split("+"))
        grl_loss_weight=0.3
    else:
        grl_dataset=None
    
    for epoch_num in range(start_epoch_num, args.epochs_num):
        
        #### Train
        epoch_start_time = datetime.now()
        # Select classifier and dataloader according to epoch
        current_group_num = epoch_num % args.groups_num
        classifiers[current_group_num] = classifiers[current_group_num].to(args.device)
        util.move_to_device(classifiers_optimizers[current_group_num], args.device)

        if args.grl:
            epoch_grl_loss = 0
            cross_entropy_loss = torch.nn.CrossEntropyLoss()
            grl_dataloader = DataLoader(dataset=grl_dataset, num_workers=args.num_workers,
                                        batch_size=args.batch_size, shuffle=True, pin_memory=(args.device == "cuda"))
        
        dataloader = commons.InfiniteDataLoader(groups[current_group_num], num_workers=args.num_workers,
                                                batch_size=args.batch_size, shuffle=True,
                                                pin_memory=(args.device == "cuda"), drop_last=True)
        
        dataloader_iterator = iter(dataloader)


        if args.warping_module:
             homography_regression = homography_regression.train()


        model = model.train()
        
        epoch_losses = np.zeros((0, 1), dtype=np.float32)
        for iteration in tqdm(range(args.iterations_per_epoch), ncols=100):
            images, targets, _ = next(dataloader_iterator)
            images, targets = images.to(args.device), targets.to(args.device)
            
            if args.augmentation_device == "cuda":
                images = gpu_augmentation(images)
            
            model_optimizer.zero_grad()
            classifiers_optimizers[current_group_num].zero_grad()
            
            if not args.use_amp16:
                descriptors = model(images)
                output = classifiers[current_group_num](descriptors, targets)
                loss = criterion(output, targets)
                loss.backward()
                epoch_losses = np.append(epoch_losses, loss.item())
                del loss, output, images
                model_optimizer.step()
                classifiers_optimizers[current_group_num].step()
            else:  # Use AMP 16
                with torch.cuda.amp.autocast():
                    descriptors = model(images)
                    output = classifiers[current_group_num](descriptors, targets)
                    loss = criterion(output, targets)
                scaler.scale(loss).backward()
                epoch_losses = np.append(epoch_losses, loss.item())
                del loss, output, images
                scaler.step(model_optimizer)
                scaler.step(classifiers_optimizers[current_group_num])
                scaler.update()

            if args.grl:
                images, labels = next(iter(grl_dataloader))
                images, labels = images.to(args.device), labels.to(args.device)
                outputs = model(images, grl=True)
                loss_grl = cross_entropy_loss(outputs, labels)
                (loss_grl *grl_loss_weight ).backward()
                epoch_grl_loss += loss_grl.item()
                del images, labels, outputs, loss_grl

            if args.warping_module:
                warped_img_1, warped_img_2, warped_intersection_points_1, warped_intersection_points_2 = to_cuda(next(dataloader_iterator))
                queries, positives = to_cuda(next(data_iter_qp))
                
                with torch.no_grad():
                    similarity_matrix_1to2, similarity_matrix_2to1 = model("similarity", [warped_img_1, warped_img_2])
                    if  consistency_w != 0:
                        queries_cons = queries[:batch_size_consistency]
                        positives_cons = positives[:batch_size_consistency]
                        similarity_matrix_q2p, similarity_matrix_p2q = model("similarity", [queries_cons, positives_cons])
                        fl_similarity_matrix_q2p, fl_similarity_matrix_p2q = model("similarity", [hflip(queries_cons), hflip(positives_cons)])
                        del queries_cons, positives_cons
                
                # ss_loss
                pred_warped_intersection_points_1 = model("regression", similarity_matrix_1to2)
                pred_warped_intersection_points_2 = model("regression", similarity_matrix_2to1)
                ss_loss = (mse(pred_warped_intersection_points_1[:, :4], warped_intersection_points_1) +
                        mse(pred_warped_intersection_points_1[:, 4:], warped_intersection_points_2) +
                        mse(pred_warped_intersection_points_2[:, :4], warped_intersection_points_2) +
                        mse(pred_warped_intersection_points_2[:, 4:], warped_intersection_points_1))
                ss_loss = compute_loss(ss_loss, ss_w)
                del pred_warped_intersection_points_1, pred_warped_intersection_points_2
           
                
                # consistency_loss
                pred_intersection_points_q2p = model("regression", similarity_matrix_q2p)
                pred_intersection_points_p2q = model("regression", similarity_matrix_p2q)
                fl_pred_intersection_points_q2p = model("regression", fl_similarity_matrix_q2p)
                fl_pred_intersection_points_p2q = model("regression", fl_similarity_matrix_p2q)
                four_predicted_points = [
                    torch.cat((pred_intersection_points_q2p[:, 4:], pred_intersection_points_q2p[:, :4]), 1),
                    pred_intersection_points_p2q,
                    hor_flip(torch.cat((fl_pred_intersection_points_q2p[:, 4:], fl_pred_intersection_points_q2p[:, :4]), 1)),
                    hor_flip(fl_pred_intersection_points_p2q)
                ]
                four_predicted_points_centroids = torch.cat([p[None] for p in four_predicted_points]).mean(0).detach()
                consistency_loss = sum([mse(pred, four_predicted_points_centroids) for pred in four_predicted_points])
                consistency_loss = compute_loss(consistency_loss, consistency_w)
                del pred_intersection_points_q2p, pred_intersection_points_p2q
                del fl_pred_intersection_points_q2p, fl_pred_intersection_points_p2q
                del four_predicted_points
         
                
                # features_wise_loss
                queries_fw = queries[:batch_size_features_wise]
                positives_fw = positives[:batch_size_features_wise]
                # Add random weights to avoid numerical instability
                random_weights = (torch.rand(batch_size_features_wise, 4)**0.1).cuda()
                w_queries, w_positives, _, _ = WDS.compute_warping(model, queries_fw, positives_fw, weights=random_weights)
                f_queries = model("features_extractor", [w_queries, "local"])
                f_positives = model("features_extractor", [w_positives, "local"])
                features_wise_loss = compute_loss(mse(f_queries, f_positives), features_wise_w)
                del queries, positives, queries_fw, positives_fw, w_queries, w_positives, f_queries, f_positives

                epoch_losses = np.concatenate((epoch_losses, np.array([[ss_loss, consistency_loss, features_wise_loss]])))

            

        classifiers[current_group_num] = classifiers[current_group_num].cpu()
        util.move_to_device(classifiers_optimizers[current_group_num], "cpu")
        
        logging.debug(f"Epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
                    f"loss = {epoch_losses.mean():.4f}")
        
        #### Evaluation
        recalls, recalls_str = test.test(args, val_ds, model)
        logging.info(f"Epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, {val_ds}: {recalls_str[:20]}")
        is_best = recalls[0] > best_val_recall1
        best_val_recall1 = max(recalls[0], best_val_recall1)
        # Save checkpoint, which contains all training parameters
        util.save_checkpoint({
            "epoch_num": epoch_num + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": model_optimizer.state_dict(),
            "classifiers_state_dict": [c.state_dict() for c in classifiers],
            "optimizers_state_dict": [c.state_dict() for c in classifiers_optimizers],
            "best_val_recall1": best_val_recall1
        }, is_best, output_folder)


    logging.info(f"Trained for {epoch_num+1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")
    

    #### Test best model on test set v1

    
    '''
    if args.model_soupe_uniform:
        state_dicts = model_soup.load_models()
        model = model_soup.get_model_soup()
    else:
        if args.model_soupe_greedy:
            #.....
            
        else:
            best_model_state_dict = torch.load(f"{output_folder}/best_model.pth")
            model.load_state_dict(best_model_state_dict)
    '''
 

    if args.warping_module:
        homography_regression = homography_regression.eval()
        test_baseline_recalls, test_baseline_recalls_pretty_str, test_baseline_predictions, _, _ = \
        util.compute_features(test_ds, model, args.fc_output_dim)
        logging.info(f"baseline test: {test_baseline_recalls_pretty_str}")
        _, reranked_test_recalls_pretty_str = test.test(model, test_baseline_predictions, test_ds,
                                                    num_reranked_predictions=num_reranked_preds)
        logging.info(f"test after warping - {reranked_test_recalls_pretty_str}")
    
    best_model_state_dict = torch.load(f"{output_folder}/best_model.pth")
    model.load_state_dict(best_model_state_dict)
    

    logging.info(f"Now testing on the test set: {test_ds}")
    recalls, recalls_str = test.test(args, test_ds, model)
    logging.info(f"{test_ds}: {recalls_str}")

    logging.info("Experiment finished (without any errors)")
