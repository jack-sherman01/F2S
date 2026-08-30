import os
import h5py
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42

def decode_log(log_path):
    baseline_list = []
    ours_list = []

    log_dir_path = os.path.join(os.path.dirname(log_path), "logs")
    log_dirs = glob.glob(os.path.join(log_dir_path, "*"))
    log_dirs = sorted(log_dirs)
    for log_dir in log_dirs:
        is_ours = "_ours_seed_" in log_dir or "round_0_" in log_dir
        round_num = int(log_dir.split("round_")[-1].split("_")[0])
        is_baseline = "_ours_seed_" not in log_dir
        log_subdir = sorted(glob.glob(os.path.join(log_dir, "*")))[-1]
        if is_baseline:
            demo_path = os.path.join(log_subdir, "eval", "demo_plus_core.hdf5")
            # print(f"Processing {demo_path}, is_ours: {is_ours}, is_baseline: {is_baseline}, round_num: {round_num}")
            with h5py.File(demo_path, "r") as f:
                jerk_list = []
                for ch in f["mask/all"][:]:
                    id = int(ch.decode().split("_")[-1])
                    if id < 100 or id >= 600:
                        continue
                    actions = f["data"][ch.decode()]["actions"][:]
                    actions = np.array(actions)[:,:3]
                    jerk = np.diff(actions, 3, axis=0)
                    jerk_magnitude = np.mean(np.linalg.norm(jerk, axis=1))
                    # print(f"ID: {id}, Jerk magnitude: {jerk_magnitude}")
                    jerk_list.append(jerk_magnitude)
                jerk_mean = np.mean(jerk_list)
                print(f"Baseline Round {round_num}, Jerk mean: {jerk_mean}")
            if is_baseline:
                if len(baseline_list) <= round_num:
                    baseline_list.append([jerk_mean])
                else:
                    baseline_list[round_num].append(jerk_mean)
        
        if is_ours:
            demo_path = os.path.join(log_subdir, "ours", "demo_plus_core.hdf5")
            # print(f"Processing {demo_path}, is_ours: {is_ours}, is_baseline: {is_baseline}, round_num: {round_num}")
            with h5py.File(demo_path, "r") as f:
                jerk_list = []
                for ch in f["mask/all"][:]:
                    id = int(ch.decode().split("_")[-1])
                    if id < 100 or id >= 600:
                        continue
                    actions = f["data"][ch.decode()]["actions"][:]
                    actions = np.array(actions)[:,:3]
                    jerk = np.diff(actions, 3, axis=0)
                    jerk_magnitude = np.mean(np.linalg.norm(jerk, axis=1))
                    # print(f"ID: {id}, Jerk magnitude: {jerk_magnitude}")
                    jerk_list.append(jerk_magnitude)
                jerk_mean = np.mean(jerk_list)
                print(f"Ours Round {round_num}, Jerk mean: {jerk_mean}")
            if is_ours:
                if len(ours_list) <= round_num:
                    ours_list.append([jerk_mean])
                else:
                    ours_list[round_num].append(jerk_mean)


    print("baseline_list: ", baseline_list)
    print("ours_list: ", ours_list)

    return baseline_list, ours_list


if __name__ == "__main__":
    ours_config = ("Ours", '#1f77b4', '-')
    sime_config = ("SIME", '#ff7f0e', '--')
    dp_config = ("DP", '#2ca02c', '--')

    local_logs = [
        dict(
            log_dict = {
                "/home/jinyang/workspace/SIME-dev/simulation/out/can_image_delta_dp_ext_64to16_vib_klw-1e-3_ns-2_multi_round/full_run_multi_round.log": {
                    "ours": ours_config,
                },
                "/home/jinyang/workspace/SIME-dev/simulation/out/can_image_delta_sime_64_multi_round/full_run_multi_round.log": {
                    "ours": sime_config,
                    "baseline": dp_config,
                }
            },
            task_name = "Can-20",
            ylim_low = 0,
            ylim_high = 0.05,
        ),
        dict(
            log_dict = {
                "/home/jinyang/workspace/SIME-dev/simulation/out/lift_image_delta_dp_ext_64to16_vib_klw-1e-3_ns-2_multi_round/full_run_multi_round.log" : {
                    "ours": ours_config,
                },
                "/home/jinyang/workspace/SIME-dev/simulation/out/lift_image_delta_sime_64_multi_round/full_run_multi_round.log" : {
                    "ours": sime_config,
                    "baseline": dp_config,
                }
            },
            task_name = "Lift-10",
            ylim_low = 0,
            ylim_high = 0.05,
        ),
        dict(
            log_dict = {
                "/home/jinyang/workspace/SIME-dev/simulation/out/square_image_delta_dp_ext_64to16_vib_klw-1e-3_ns-2_multi_round/full_run_multi_round.log" : {
                    "ours": ours_config,
                },
                "/home/jinyang/workspace/SIME-dev/simulation/out/square_image_delta_sime_64_multi_round/full_run_multi_round.log" : {
                    "ours": sime_config,
                    "baseline": dp_config,
                }
            },
            task_name = "Square-20",
            ylim_low = 0,
            ylim_high = 0.05,
        ),
        dict(
            log_dict = {
                "/home/jinyang/workspace/SIME-dev/simulation/out/transport_image_delta_dp_ext_64to16_vib_klw-1e-4_ns-2_multi_round/full_run_multi_round.log" : {
                    "ours": ours_config,
                },
                "/home/jinyang/workspace/SIME-dev/simulation/out/transport_image_delta_sime_64_multi_round/full_run_multi_round.log.combined": {
                    "ours": sime_config,
                    "baseline": dp_config,
                }
            },
            task_name = "Transport-20",
            ylim_low = 0,
            ylim_high = 0.05,
        ),
    ]

    for local_log in local_logs:
        log_dict = local_log["log_dict"]
        task_name = local_log["task_name"]
        ylim_low = local_log["ylim_low"]
        ylim_high = local_log["ylim_high"]

        plt.figure(figsize=(7, 5))
        plt.title(f"{task_name}", fontsize=20)
        x_len = 0
        for log_path, name_dict in log_dict.items():
            baseline_list, ours_list = decode_log(log_path)
            baseline_mean = [np.mean(x) for x in baseline_list]
            baseline_std = [np.std(x) for x in baseline_list]
            baseline_upper = [np.ma.max(x) for x in baseline_list]
            baseline_lower = [np.ma.min(x) for x in baseline_list]
            ours_mean = [np.mean(x) for x in ours_list]
            ours_std = [np.std(x) for x in ours_list]
            ours_upper = [np.ma.max(x) for x in ours_list]
            ours_lower = [np.ma.min(x) for x in ours_list]
            if "baseline" in name_dict:
                plt.plot(
                    np.arange(len(baseline_mean)), np.array(baseline_mean), 
                    linestyle=name_dict['baseline'][2], color=name_dict['baseline'][1], label=f"{name_dict['baseline'][0]}"
                )
                plt.fill_between(
                    np.arange(len(baseline_mean)), np.array(baseline_upper), np.array(baseline_lower), 
                    # np.arange(len(baseline_mean)), np.array(baseline_mean) + np.array(baseline_std), np.array(baseline_mean) - np.array(baseline_std), 
                    color=name_dict['baseline'][1], alpha=0.1, edgecolor=None
                )
            if "ours" in name_dict:
                plt.plot(
                    np.arange(len(ours_mean)), np.array(ours_mean), 
                    linestyle=name_dict['ours'][2], color=name_dict['ours'][1], label=f"{name_dict['ours'][0]}"
                )
                plt.fill_between(
                    np.arange(len(ours_mean)), np.array(ours_upper), np.array(ours_lower), 
                    # np.arange(len(ours_mean)), np.array(ours_mean) + np.array(ours_std), np.array(ours_mean) - np.array(ours_std),
                    color=name_dict['ours'][1], alpha=0.1, edgecolor=None
                )
            x_len = max(x_len, len(baseline_mean), len(ours_mean))

        plt.xlabel("Round", fontsize=20)
        plt.ylabel("Jerk Magnitude", fontsize=20)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12, loc="lower right")
        plt.grid(axis='y')

        plt.xlim(0, x_len-1)
        plt.ylim(ylim_low, ylim_high)
        plt.xticks(np.arange(0, x_len, 1))
        plt.yticks(np.arange(ylim_low, ylim_high + 0.01, (ylim_high-ylim_low)/5))

        os.makedirs("figures", exist_ok=True)
        plt.savefig(f"figures/Jerk-{task_name}.pdf", format="pdf")
        plt.show()
