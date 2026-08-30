import os
import sys
import h5py
import json
import logging
import argparse
import subprocess
import numpy as np
from easydict import EasyDict

DP_SRC_ROOT = os.path.join(os.path.dirname(__file__), "../src")
print("diffusion policy source root:", DP_SRC_ROOT)
sys.path.insert(0, DP_SRC_ROOT)

from extract_demo import extract_demo
from robomimic_dataset_conversion import convert_rel_actions_to_abs
from extract_useful_data import extract_useful_data_v2
from dataset_combine import combine_dataset

def generate_training_cfg(task_config, used_demo, dataset_path_dict, abs_output_dir, seeds, prefix, extra_params_for_dataset=None, finetune=False):
    config_template_path = os.path.join("config_template", task_config + ".json")
    assert os.path.exists(config_template_path)
    with open(config_template_path, "r") as f:
        config_template = json.load(f)
    config_template = EasyDict(config_template)
    for seed in seeds:
        config = config_template.copy()
        config = EasyDict(config)
        config.seed = seed
        config.dataset.params.path = dataset_path_dict[seed]
        config.dataset.params.train_filter_key = used_demo
        if extra_params_for_dataset is not None:
            config.dataset.params.update(extra_params_for_dataset)
        config.log_dir = os.path.join(abs_output_dir, "logs/{}_seed_{}".format(prefix, seed))
        if finetune:
            # if finetune, we need to resume from the last checkpoint, instead of predefined checkpoint provided by the config
            # we assume dataset and ckpt are in the same log dir
            config.resume_ckpt = os.path.join("/".join(dataset_path_dict[seed].split("/")[:-2]), "ckpt", "policy_last.ckpt")
            config.num_epochs = config.finetune_epochs if hasattr(config, "finetune_epochs") else config.num_epochs
            config.optimizer.params.lr = config.finetune_lr if hasattr(config, "finetune_lr") else config.optimizer.params.lr
        else:
            if not hasattr(config, "resume_ckpt"):
                config.resume_ckpt = None
        if not hasattr(config, "resume_epoch"):
            config.resume_epoch = -1
        config_path = os.path.join(abs_output_dir, "configs/{}_seed_{}.json".format(prefix, seed))
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
        with h5py.File(dataset_path_dict[seed], "r") as f:
            if "mask/{}".format(used_demo) not in f:
                logging.info(
                    "Generated training config {}_seed_{}.json".format(prefix, seed)
                )
                logging.warning("Mask {} not found in dataset {}".format(used_demo, dataset_path_dict[seed]))
            else:
                num_demos = len(f["mask/{}".format(used_demo)])
                logging.info(
                    "Generated training config {}_seed_{}.json".format(prefix, seed) + " " +
                    "Number of demos in {}: {}".format(used_demo, num_demos)
                )

def start_training(abs_output_dir, seeds, prefix, cuda_device=None, resume=False):
    processes = []
    for i, seed in enumerate(seeds):
        config_name = "{}_seed_{}.json".format(prefix, seed)
        if resume and os.path.exists(os.path.join(abs_output_dir, "logs/{}_seed_{}".format(prefix, seed))):
            logging.info("Skip training {}_seed_{} because it already exists".format(prefix, seed))
            continue
        visible_devices = 0 if cuda_device is None else cuda_device[i%len(cuda_device)]
        # cmd_prefix = "torchrun --nnodes 1 --rdzv_backend c10d --rdzv_endpoint localhost:0 --nproc_per_node=gpu train.py "
        cmd_prefix = "python train_single_gpu.py "
        cmd = "cd {} && CUDA_VISIBLE_DEVICES={} ".format(DP_SRC_ROOT, visible_devices) + \
              cmd_prefix + \
              "--config {}".format(os.path.join(abs_output_dir, "configs/{}".format(config_name)))
        process = subprocess.Popen(cmd, shell=True)
        processes.append(process)
        logging.info("Started training {}".format(config_name))
    # spin until all training is done
    for process in processes:
        process.wait()
    logging.info("All training is done")

def start_evaluating(abs_output_dir, seeds, prefix, eval_specific_args="", eval_output_dir_name="eval", count_only=False, cuda_device=None):
    global n_rollouts
    processes = []
    eval_log_dir_dict = {}
    for i, seed in enumerate(seeds):
        config_name = "{}_seed_{}.json".format(prefix, seed)
        trys = os.listdir(os.path.join(abs_output_dir, "logs/{}_seed_{}".format(prefix, seed)))
        trys = sorted(trys) # use the last try
        eval_dataset_id = "{}_seed_{}".format(prefix, seed)
        eval_log_id = trys[-1]
        eval_log_dir = os.path.join(abs_output_dir, "logs", eval_dataset_id, eval_log_id)
        eval_log_dir_dict[eval_dataset_id] = eval_log_dir
        if not count_only and not os.path.exists(os.path.join(eval_log_dir, eval_output_dir_name)):
            eval_ckpt = "policy_last"
            rollout_args = f"--n_rollouts {n_rollouts} --seed 233 --try_times 5 --abs_action"
            extra_args = "--dataset_obs --render_traj" #--return_intermediate"
            cmd = "CUDA_VISIBLE_DEVICES={} python run.py --agent {} --config {} {} --dataset_path {}/demo.hdf5 --video_dir {} {} {}".format(
                0 if cuda_device is None else cuda_device[i%len(cuda_device)],
                os.path.join(abs_output_dir, "logs", eval_dataset_id, eval_log_id, "ckpt", eval_ckpt + ".ckpt"),
                os.path.join(abs_output_dir, "logs", eval_dataset_id, eval_log_id, "config.json"),
                rollout_args,
                os.path.join(eval_log_dir, eval_output_dir_name),
                os.path.join(eval_log_dir, eval_output_dir_name),
                extra_args,
                eval_specific_args
            )
            process = subprocess.Popen(cmd, shell=True)
            processes.append(process)
            logging.info("Started evaluating {} {}".format(eval_dataset_id, eval_specific_args))
    # spin until all evaluation is done
    for process in processes:
        process.wait()
    logging.info("All evaluation is done")
    # calculate success rate
    success_rate_list = []
    exploration_total_num_list = []
    exploration_success_num_list = []
    for seed in seeds:
        eval_demo_path = os.path.join(eval_log_dir_dict["{}_seed_{}".format(prefix, seed)], eval_output_dir_name, "demo.hdf5")
        eval_demo_plus_core_path = os.path.join(eval_log_dir_dict["{}_seed_{}".format(prefix, seed)], eval_output_dir_name, "demo_plus_core.hdf5")
        if not os.path.exists(eval_demo_path) and os.path.exists(eval_demo_plus_core_path):
            logging.info("Skip evaluating {}_seed_{} because demo.hdf5 is deleted but demo_plus_core.hdf5 exists".format(prefix, seed))
            continue
        success_num = 0
        exploration_success_num = 0
        exploration_total_num = 0
        with h5py.File(eval_demo_path, "r") as f:
            for ep in f["data"]:
                is_success = np.any(f["data/{}/dones".format(ep)][()])
                if int(ep[5:]) >= n_rollouts:
                    exploration_total_num += 1
                    if is_success:
                        exploration_success_num += 1
                else:
                    if is_success:
                        success_num += 1
            success_rate = success_num / n_rollouts
            success_rate_list.append(success_rate)
            exploration_success_num_list.append(exploration_success_num)
            exploration_total_num_list.append(exploration_total_num)
        logging.info("Success rate of {}_seed_{}/{}/{}: {}".format(prefix, seed, eval_log_id, eval_output_dir_name, success_rate))
        logging.info("- Exploration success rate of {}_seed_{}/{}/{}: {}/{}, {}".format(
            prefix, seed, eval_log_id, eval_output_dir_name, exploration_success_num, exploration_total_num, 
            exploration_success_num / exploration_total_num if exploration_total_num > 0 else None
        ))
    if success_rate_list:
        logging.info("Success rate mean of {} {}: {}".format(prefix, eval_output_dir_name, np.mean(success_rate_list)))
        logging.info("Success rate std of {} {}: {}".format(prefix, eval_output_dir_name, np.std(success_rate_list)))
    return eval_log_dir_dict

def start_extracting_useful_data(eval_log_dir_dict, seeds, prefix, eval_output_dir_name="eval"):
    global n_rollouts
    for seed in seeds:
        eval_demo_path = os.path.join(eval_log_dir_dict["{}_seed_{}".format(prefix, seed)], eval_output_dir_name, "demo.hdf5")
        eval_demo_plus_core_path = os.path.join(eval_log_dir_dict["{}_seed_{}".format(prefix, seed)], eval_output_dir_name, "demo_plus_core.hdf5")
        if os.path.exists(eval_demo_plus_core_path):
            logging.info("Skip extracting useful data from {}_seed_{} because demo_plus_core.hdf5 already exists".format(prefix, seed))
            continue
        assert os.path.exists(eval_demo_path)
        extract_useful_data_v2(eval_demo_path, "success", start_index=n_rollouts)
        logging.info("Extracted useful data from {}_seed_{}".format(prefix, seed))

def start_combining_dataset(used_demo, core_dataset_path, eval_log_dir_dict, seeds, prefix, eval_output_dir_name="eval"):
    for seed in seeds:
        eval_demo_path = os.path.join(eval_log_dir_dict["{}_seed_{}".format(prefix, seed)], eval_output_dir_name, "demo.hdf5")
        output_demo_path = os.path.join(eval_log_dir_dict["{}_seed_{}".format(prefix, seed)], eval_output_dir_name, "demo_plus_core.hdf5")
        if os.path.exists(output_demo_path):
            logging.info("Skip combining dataset {}_seed_{} because demo_plus_core.hdf5 already exists".format(prefix, seed))
            continue
        dataset_combine_cfg = EasyDict({
            "output": output_demo_path,
            "input": [
                {
                    "path": eval_demo_path,
                    "map":[
                        [None, "all"],
                        ["success", "train_success"],
                        [None, "from_eval"]
                    ]
                }, 
                {
                    "path": core_dataset_path,
                    "map":[
                        [used_demo, "all"],
                        [used_demo, "train_success"],
                        [used_demo, "from_prev"]
                    ]
                }
            ]
        })
        combine_dataset(dataset_combine_cfg)
        logging.info("Combined dataset {}_seed_{} and core".format(prefix, seed))

def start_combining_dataset_for_multi_round(used_demo, eval_log_dir_dict, seeds, old_prefix, prefix, eval_output_dir_name="eval"):
    for seed in seeds:
        core_dataset_path = os.path.join(eval_log_dir_dict["{}_seed_{}".format(old_prefix, seed)], eval_output_dir_name, "demo_plus_core.hdf5")
        eval_demo_path = os.path.join(eval_log_dir_dict["{}_seed_{}".format(prefix, seed)], eval_output_dir_name, "demo.hdf5")
        output_demo_path = os.path.join(eval_log_dir_dict["{}_seed_{}".format(prefix, seed)], eval_output_dir_name, "demo_plus_core.hdf5")
        if os.path.exists(output_demo_path):
            logging.info("Skip combining dataset {}_seed_{} because demo_plus_core.hdf5 already exists".format(prefix, seed))
            continue
        dataset_combine_cfg = EasyDict({
            "output": output_demo_path,
            "input": [
                {
                    "path": eval_demo_path,
                    "map":[
                        [None, "all"],
                        ["success", "train_success"],
                        [None, "from_eval"]
                    ]
                }, 
                {
                    "path": core_dataset_path,
                    "map":[
                        [used_demo, "all"],
                        [used_demo, "train_success"],
                        [used_demo, "from_prev"]
                    ]
                }
            ]
        })
        combine_dataset(dataset_combine_cfg)
        logging.info("Combined dataset {}_seed_{} and core".format(prefix, seed))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="path to xxx_v141.hdf5 dataset file, xxx can be low_dim, image, etc.",
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="task config name",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="output directory for saving results",
    )

    parser.add_argument(
        "--used_demo",
        type=str,
        required=True,
        help="e.g. core_5, core_10, core_20, core_40, core_50, core_100, core_200",
    )

    parser.add_argument(
        "--seeds",
        default=None,
        type=int,
        nargs="+",
        help="every given seed will be tested",
    )

    parser.add_argument(
        "--cuda_device",
        default=None,
        type=int,
        nargs="+",
        help="use which cuda device to train",
    )

    parser.add_argument(
        "--noise_scale",
        default=0.5,    # 0.5 for image, 0.01 for low_dim
        type=float,
        help="noise scale",
    )

    parser.add_argument(
        "--finetune",
        action="store_true",
        help="whether to finetune the model from the last checkpoint",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="whether to resume the last running",
    )

    parser.add_argument(
        "--no_baseline",
        action="store_true",
        help="whether to skip the baseline training, i.e. only run OURS",
    )

    args = parser.parse_args()

    if args.seeds is None:
        seeds = [233, 2333, 23333, 233333]
    else:
        seeds = args.seeds

    n_rollouts = 100

    # set up logging
    abs_output_dir = os.path.abspath(args.output_dir)
    os.makedirs(abs_output_dir, exist_ok=True)
    log_file = os.path.join(abs_output_dir, "full_run_multi_round.log")
    logging.basicConfig(
        filename=log_file, level=logging.INFO, filemode="w",
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # ensure actions are in absolute form
    dataset_path = args.dataset
    dataset_dir = os.path.dirname(dataset_path)
    dataset_name = os.path.basename(dataset_path)
    assert dataset_name[-5:] == ".hdf5"
    dataset_path_of_abs_actions = os.path.join(dataset_dir, dataset_name[:-5] + "_abs_6drot.hdf5")
    if not os.path.exists(dataset_path_of_abs_actions):
        # back up the original dataset
        backup_dataset_path = os.path.join(dataset_dir, dataset_name[:-5] + ".bak.hdf5")
        os.system("cp {} {}".format(dataset_path, backup_dataset_path))
        logging.info("Backed up the original dataset to {}".format(backup_dataset_path))

        # count the number of demos
        f = h5py.File(dataset_path, "r")
        num_demos = len(list(f["data"].keys()))
        f.close()
        logging.info("Number of demos: {}".format(num_demos))
        assert num_demos == 200

        # extract demos
        assert args.used_demo in ["core_5", "core_10", "core_20", "core_40", "core_50", "core_100", "core_200"]
        extract_demo(dataset_path, "core_5", num_demos//5)
        extract_demo(dataset_path, "core_10", num_demos//10)
        extract_demo(dataset_path, "core_20", num_demos//20)
        extract_demo(dataset_path, "core_40", num_demos//40)
        extract_demo(dataset_path, "core_50", num_demos//50)
        extract_demo(dataset_path, "core_100", num_demos//100)
        extract_demo(dataset_path, "core_200", num_demos//200)
        logging.info("Extracted core_5, core_10, core_20, core_40, core_50, core_100, core_200")

        # convert relative actions to absolute actions
        eval_dir = os.path.join(dataset_dir, dataset_name[:-5] + "_abs_6drot_eval")
        convert_rel_actions_to_abs(dataset_path, dataset_path_of_abs_actions, eval_dir, num_workers=None)
        logging.info("Converted relative actions to absolute actions")
    else:
        logging.info("Absolute action dataset already exist")
    core_path = dataset_path = dataset_path_of_abs_actions
    abs_core_path = os.path.abspath(core_path)

    # generate training configs
    dataset_path_dict = {seed: abs_core_path for seed in seeds}
    generate_training_cfg(args.config, args.used_demo, dataset_path_dict, abs_output_dir, seeds, "round_0")

    # start training
    start_training(abs_output_dir, seeds, "round_0", cuda_device=args.cuda_device, resume=args.resume)

    eval_log_dir_dict = {}

    # round 0 
    logging.info("----------------------- Round 0 -----------------------")

    # evaluate the trained models
    eval_log_dir_dict.update(start_evaluating(abs_output_dir, seeds, "round_0", cuda_device=args.cuda_device))#, eval_specific_args="--num_inference_steps 100")) # NOTE: remember to change this

    # extract useful data
    start_extracting_useful_data(eval_log_dir_dict, seeds, "round_0")

    # combine dataset
    start_combining_dataset(args.used_demo, abs_core_path, eval_log_dir_dict, seeds, "round_0")

    # ours
    eval_output_dir_name = "ours"

    # evaluate the trained models
    eval_log_dir_dict.update(start_evaluating(abs_output_dir, seeds, "round_0", eval_specific_args="--enable_exploration --tau1 0.0 --tau2 1.0 --noise_scale {}".format(args.noise_scale), eval_output_dir_name=eval_output_dir_name, cuda_device=args.cuda_device))

    # extract useful data
    start_extracting_useful_data(eval_log_dir_dict, seeds, "round_0", eval_output_dir_name=eval_output_dir_name)

    # combine dataset
    start_combining_dataset(args.used_demo, abs_core_path, eval_log_dir_dict, seeds, "round_0", eval_output_dir_name=eval_output_dir_name)


    if not args.no_baseline:

        # round 1
        logging.info("----------------------- Round 1 -----------------------")
        
        # generate training configs for the second round
        dataset_path_dict = {
            seed: os.path.join(eval_log_dir_dict["round_0_seed_{}".format(seed)], "eval", "demo_plus_core.hdf5") 
            for seed in seeds
        }
        generate_training_cfg(args.config, "train_success", dataset_path_dict, abs_output_dir, seeds, "round_1", finetune=args.finetune)

        # start training for the second round
        start_training(abs_output_dir, seeds, "round_1", cuda_device=args.cuda_device, resume=args.resume)

        # evaluate the trained models for the second round
        eval_log_dir_dict.update(start_evaluating(abs_output_dir, seeds, "round_1", eval_output_dir_name="eval", cuda_device=args.cuda_device))

        # extract useful data
        start_extracting_useful_data(eval_log_dir_dict, seeds, "round_1", eval_output_dir_name="eval")

        # combine dataset
        start_combining_dataset_for_multi_round("train_success", eval_log_dir_dict, seeds, "round_0", "round_1", eval_output_dir_name="eval")
    


    # round 1 ours
    logging.info("----------------------- Round 1 OURS -----------------------")

    # generate training configs for the second round
    dataset_path_dict = {
        seed: os.path.join(eval_log_dir_dict["round_0_seed_{}".format(seed)], "ours", "demo_plus_core.hdf5") 
        for seed in seeds
    }
    generate_training_cfg(args.config, "train_success", dataset_path_dict, abs_output_dir, seeds, "round_1_{}".format("ours"), finetune=args.finetune)

    # start training for the second round
    start_training(abs_output_dir, seeds, "round_1_{}".format("ours"), cuda_device=args.cuda_device, resume=args.resume)

    # evaluate the trained models for the second round but ours
    eval_log_dir_dict.update(start_evaluating(abs_output_dir, seeds, "round_1_{}".format("ours"), eval_output_dir_name="ours", cuda_device=args.cuda_device, eval_specific_args="--enable_exploration --tau1 0.0 --tau2 1.0 --noise_scale {}".format(args.noise_scale)))

    # extract useful data
    start_extracting_useful_data(eval_log_dir_dict, seeds, "round_1_{}".format("ours"), eval_output_dir_name="ours")

    # combine dataset
    start_combining_dataset_for_multi_round("train_success", eval_log_dir_dict, seeds, "round_0", "round_1_{}".format("ours"), eval_output_dir_name="ours")


    def round_i(i):

        if not args.no_baseline:

            # round i
            logging.info("----------------------- Round {} -----------------------".format(i))

            # generate training configs for the next round
            dataset_path_dict = {
                seed: os.path.join(eval_log_dir_dict["round_{}_seed_{}".format(i-1, seed)], "eval", "demo_plus_core.hdf5") 
                for seed in seeds
            }
            generate_training_cfg(args.config, "train_success", dataset_path_dict, abs_output_dir, seeds, "round_{}".format(i), finetune=args.finetune)

            # start training for the next round
            start_training(abs_output_dir, seeds, "round_{}".format(i), cuda_device=args.cuda_device, resume=args.resume)

            # evaluate the trained models for the next round
            eval_log_dir_dict.update(start_evaluating(abs_output_dir, seeds, "round_{}".format(i), eval_output_dir_name="eval", cuda_device=args.cuda_device))

            # extract useful data
            start_extracting_useful_data(eval_log_dir_dict, seeds, "round_{}".format(i), eval_output_dir_name="eval")

            # combine dataset
            start_combining_dataset_for_multi_round("train_success", eval_log_dir_dict, seeds, "round_{}".format(i-1), "round_{}".format(i), eval_output_dir_name="eval")



        # round i ours
        logging.info("----------------------- Round {} OURS -----------------------".format(i))

        # generate training configs for the next round
        dataset_path_dict = {
            seed: os.path.join(eval_log_dir_dict["round_{}_{}_seed_{}".format(i-1, "ours", seed)], "ours", "demo_plus_core.hdf5") 
            for seed in seeds
        }
        generate_training_cfg(args.config, "train_success", dataset_path_dict, abs_output_dir, seeds, "round_{}_{}".format(i, "ours"), finetune=args.finetune)

        # start training for the next round
        start_training(abs_output_dir, seeds, "round_{}_{}".format(i, "ours"), cuda_device=args.cuda_device, resume=args.resume)

        # evaluate the trained models for the next round but ours
        eval_log_dir_dict.update(start_evaluating(abs_output_dir, seeds, "round_{}_{}".format(i, "ours"), eval_output_dir_name="ours", cuda_device=args.cuda_device, eval_specific_args="--enable_exploration --tau1 0.0 --tau2 1.0 --noise_scale {}".format(args.noise_scale)))

        # extract useful data
        start_extracting_useful_data(eval_log_dir_dict, seeds, "round_{}_{}".format(i, "ours"), eval_output_dir_name="ours")

        # combine dataset
        start_combining_dataset_for_multi_round("train_success", eval_log_dir_dict, seeds, "round_{}_{}".format(i-1, "ours"), "round_{}_{}".format(i, "ours"), eval_output_dir_name="ours")

    for i in range(2, 6):
        round_i(i)

    logging.info("All done. This is the end of the script.")