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
    import arcface_loss
    import sphereface_loss
    import elastic_face
    import new_elastic_face
    import GradientReversalLayer as GRL
    import augmentations
    from model import network
    from datasets.test_dataset import TestDataset
    from datasets.train_dataset import TrainDataset
    from datasets.grl_dataset import GrlDataset
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
    model = network.GeoLocalizationNet(args.backbone, args.fc_output_dim, grl_discriminator)

    logging.info(f"There are {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs.")

    if args.resume_model is not None:
        logging.debug(f"Loading model from {args.resume_model}")
        model_state_dict = torch.load(args.resume_model)
        model.load_state_dict(model_state_dict)

    model = model.to(args.device).train()


    #### Optimizer
    criterion = torch.nn.CrossEntropyLoss()
    model_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    #### Datasets
    groups = [TrainDataset(args, args.train_set_folder, M=args.M, alpha=args.alpha, N=args.N, L=args.L,
                        current_group=n, min_images_per_class=args.min_images_per_class) for n in range(args.groups_num)]
    # Each group has its own classifier, which depends on the number of classes in the group
   
    #Loss selection
    if args.loss =="cosface":
        classifiers = [cosface_loss.MarginCosineProduct(args.fc_output_dim, len(group)) for group in groups]
    elif args.loss =="arcface":
        classifiers = [arcface_loss.ArcFace(args.fc_output_dim, len(group)) for group in groups]
    elif args.loss =="sphereface":
        classifiers = [sphereface_loss.SphereFace(args.fc_output_dim, len(group)) for group in groups]
    elif args.loss =="elasticface":
        classifiers = [elastic_face.ElasticCosFace(args.fc_output_dim, len(group)) for group in groups]
    elif args.loss =="new_elasticface":
        classifiers = [new_elastic_face.ElasticCosFace(args.fc_output_dim, len(group)) for group in groups]

    classifiers_optimizers = [torch.optim.Adam(classifier.parameters(), lr=args.classifiers_lr) for classifier in classifiers]

    logging.info(f"Using {len(groups)} groups")
    logging.info(f"The {len(groups)} groups have respectively the following number of classes {[len(g) for g in groups]}")
    logging.info(f"The {len(groups)} groups have respectively the following number of images {[g.get_images_num() for g in groups]}")

    val_ds = TestDataset(args.val_set_folder, positive_dist_threshold=args.positive_dist_threshold)
    test_ds = TestDataset(args.test_set_folder, queries_folder="queries_v1",
                        positive_dist_threshold=args.positive_dist_threshold)
    logging.info(f"Validation set: {val_ds}")
    logging.info(f"Test set: {test_ds}")


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
                  # augmentations.DeviceAgnosticRandomPerspective(),
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
        
        if args.grl:
            grl_dataloader_iterator=iter(grl_dataloader)

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
                
                if args.grl:
                    grl_images, labels = next(grl_dataloader_iterator)
                    grl_images, labels = grl_images.to(args.device), labels.to(args.device)
                    outputs = model(grl_images, grl=True)
                    loss_grl = cross_entropy_loss(outputs, labels)
                    (loss_grl *grl_loss_weight ).backward()
                    epoch_grl_loss += loss_grl.item()
                    del grl_images, labels, outputs, loss_grl

                epoch_losses = np.append(epoch_losses, loss.item())
                del loss, output, images
                model_optimizer.step()
                classifiers_optimizers[current_group_num].step()
            else:  # Use AMP 16
                with torch.cuda.amp.autocast():
                    descriptors = model(images)
                    output = classifiers[current_group_num](descriptors, targets)
                    loss = criterion(output, targets)

                    if args.grl:
                        grl_images, labels = next(grl_dataloader_iterator)
                        grl_images, labels = grl_images.to(args.device), labels.to(args.device)
                        outputs = model(grl_images, grl=True)
                        loss_grl = cross_entropy_loss(outputs, labels)
                        (loss_grl *grl_loss_weight ).backward()
                        epoch_grl_loss += loss_grl.item()
                        del grl_images, labels, outputs, loss_grl
                
                scaler.scale(loss).backward()
                epoch_losses = np.append(epoch_losses, loss.item())
                del loss, output, images
                scaler.step(model_optimizer)
                scaler.step(classifiers_optimizers[current_group_num])
                scaler.update()

            if args.grl:
                grl_images, labels = next(grl_dataloader_iterator)
                grl_images, labels = grl_images.to(args.device), labels.to(args.device)
                outputs = model(grl_images, grl=True)
                loss_grl = cross_entropy_loss(outputs, labels)
                (loss_grl *grl_loss_weight ).backward()
                epoch_grl_loss += loss_grl.item()
                del grl_images, labels, outputs, loss_grl

              

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
    
    best_model_state_dict = torch.load(f"{output_folder}/best_model.pth")
    model.load_state_dict(best_model_state_dict)
    

    logging.info(f"Now testing on the test set: {test_ds}")
    recalls, recalls_str = test.test(args, test_ds, model)
    logging.info(f"{test_ds}: {recalls_str}")

    logging.info("Experiment finished (without any errors)")
