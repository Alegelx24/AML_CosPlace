if __name__ == '__main__':
    import sys
    import torch
    import logging
    import multiprocessing
    from datetime import datetime

    import test
    import parser_1
    import commons
    import FDA as fda
    from model import network
    from datasets.test_dataset import TestDataset
    import model_soup
    import os


    torch.backends.cudnn.benchmark = True  # Provides a speedup

    args = parser_1.parse_arguments(is_training=False)
    start_time = datetime.now()
    output_folder = f"logs/{args.save_dir}/{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
    commons.make_deterministic(args.seed)
    commons.setup_logging(output_folder, console="info")
    logging.info(" ".join(sys.argv))
    logging.info(f"Arguments: {args}")
    logging.info(f"The outputs are being saved in {output_folder}")

    #### Model
    model = network.GeoLocalizationNet(args.backbone, args.fc_output_dim)

    logging.info(f"There are {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs.")

    if args.resume_model is not None:
        logging.info(f"Loading model from {args.resume_model}")
        model_state_dict = torch.load(args.resume_model)
        
        if args.grl:
            del model_state_dict["grl_discriminator.1.weight"]
            del model_state_dict["grl_discriminator.1.bias"]
            del model_state_dict["grl_discriminator.3.weight"]
            del model_state_dict["grl_discriminator.3.bias"]
            del model_state_dict["grl_discriminator.5.weight"]
            del model_state_dict["grl_discriminator.5.bias"]
        model.load_state_dict(model_state_dict)
    else:
        logging.info("WARNING: You didn't provide a path to resume the model (--resume_model parameter). " +
                    "Evaluation will be computed using randomly initialized weights.")

    model = model.to(args.device)

    #model soup
    if args.model_soupe_uniform:
        print("loading state dicts for model soup uniform...")
        state_dicts = model_soup.load_models()
        alphal = [1 / len(state_dicts) for i in range(len(state_dicts))]
        model = model_soup.get_model_soup(model, state_dicts, alphal)
    
    if args.model_soupe_greedy:
        state_dicts = model_soup.load_models()
        args.val_set_folder = os.path.join(args.dataset_folder, "val")
        if not os.path.exists(args.val_set_folder):
            raise FileNotFoundError(f"Folder {args.val_set_folder} does not exist")
        val_ds = TestDataset(args.val_set_folder, positive_dist_threshold=args.positive_dist_threshold)

        val_results = []

        for i in range(len(state_dicts)):
          recalls,_ = test.test(args, val_ds, model)
          r1=recalls[0]/100
          val_results.append(r1)

        ranked_candidates = [i for i in range(len(state_dicts))]
        ranked_candidates.sort(key=lambda x: -val_results[x])

        current_best = val_results[ranked_candidates[0]]
        best_ingredients = ranked_candidates[:1]
        for i in range(1, len(state_dicts)):
            # add current index to the ingredients
            ingredient_indices = best_ingredients \
                + [ranked_candidates[i]]
            alphal = [0 for i in range(len(state_dicts))]
            for j in ingredient_indices:
                alphal[j] = 1 / len(ingredient_indices)
            model = model_soup.get_model_soup(model, state_dicts, alphal)
    #fda on test
    if args.fda:
        fda.FDA_database_transform(args.test_set_folder+"/database",args.test_set_folder+"/queries_v1",args.test_set_folder+"/database_trasformed", args.fda_weight)
        test_ds = TestDataset(args.test_set_folder,database_folder="database_trasformed", queries_folder="queries_v1",
                                positive_dist_threshold=args.positive_dist_threshold)
    else:
        test_ds = TestDataset(args.test_set_folder, queries_folder="queries_v1",
                        positive_dist_threshold=args.positive_dist_threshold)

    recalls, recalls_str = test.test(args, test_ds, model)
    logging.info(f"{test_ds}: {recalls_str}")
