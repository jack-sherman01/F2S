import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42

def decode_log(log_path):
    with open(log_path, "r") as f:
        lines = f.readlines()

    baseline_mean = []
    baseline_upper = []
    baseline_lower = []
    baseline_std = []
    ours_mean = []
    ours_upper = []
    ours_lower = []
    ours_std = []
    for line in lines:
        if ("eval" not in line and "ours" not in line) or "Success rate" not in line:
            continue
        
        print(line.strip())
        round_num = int(line.split("_")[1].split(" ")[0])
        # print("round_num", round_num)
        update_baseline = "ours" not in line
        update_ours = "ours" in line
        val = float(line.strip().split(" ")[-1])
        print(f"update_baseline: {update_baseline}, update_ours: {update_ours}, val: {val}")
        if "mean" in line:
            if update_baseline:
                if len(baseline_mean) <= round_num:
                    baseline_mean.append(val)
                else:
                    baseline_mean[round_num] = val
            if update_ours:
                if len(ours_mean) <= round_num:
                    ours_mean.append(val)
                else:
                    ours_mean[round_num] = val
        elif "std" in line:
            if update_baseline:
                if len(baseline_std) <= round_num:
                    baseline_std.append(val)
                else:
                    baseline_std[round_num] = val
            if update_ours:
                if len(ours_std) <= round_num:
                    ours_std.append(val)
                else:
                    ours_std[round_num] = val
        else:
            if update_baseline:
                if len(baseline_upper) <= round_num:
                    baseline_upper.append(val)
                else:
                    baseline_upper[round_num] = max(val, baseline_upper[round_num])
            if update_ours:
                if len(ours_upper) <= round_num:
                    ours_upper.append(val)
                else:
                    ours_upper[round_num] = max(val, ours_upper[round_num])
            if update_baseline:
                if len(baseline_lower) <= round_num:
                    baseline_lower.append(val)
                else:
                    baseline_lower[round_num] = min(val, baseline_lower[round_num])
            if update_ours:
                if len(ours_lower) <= round_num:
                    ours_lower.append(val)
                else:
                    ours_lower[round_num] = min(val, ours_lower[round_num])

    print("baseline_mean: ", baseline_mean)
    print("baseline_upper: ", baseline_upper)
    print("baseline_lower: ", baseline_lower)
    print("baseline_std: ", baseline_std)
    print("ours_mean: ", ours_mean)
    print("ours_upper: ", ours_upper)
    print("ours_lower: ", ours_lower)
    print("ours_std: ", ours_std)

    return baseline_mean, baseline_upper, baseline_lower, baseline_std, ours_mean, ours_upper, ours_lower, ours_std

if __name__ == "__main__":

    local_logs = [
        dict(
            log_dict = {
                "/home/jinyang/workspace/SIME-dev/simulation/out/can_image_delta_dp_ext_64to16_vib_klw-1e-3_ns-2_multi_round/full_run_multi_round.log": {
                    "ours": ("Ours", '#1f77b4', '-'),
                },
                "/home/jinyang/workspace/SIME-dev/simulation/out/can_image_delta_dp_ext_64to64_no_kl_multi_round/full_run_multi_round.log": {
                    "ours": ("Ours w/o KL", '#17becf', '--'),
                }
            },
            task_name = None,
            save_name = "kl",
            ylim_low = 0.45,
            ylim_high = 0.95,
        ),
        dict(
            log_dict = {
                "/home/jinyang/workspace/SIME-dev/simulation/out/can_image_delta_dp_ext_64to16_vib_klw-1e-3_ns-2_multi_round/full_run_multi_round.log": {
                    "ours": ("$\\alpha$=2.0", '#1f77b4', '-'),
                },
                "/home/jinyang/workspace/SIME-dev/simulation/out/can_image_delta_dp_ext_64to16_vib_klw-1e-3_ns-3_multi_round/full_run_multi_round.log": {
                    "ours": ("$\\alpha$=3.0", '#17becf', '--'),
                },
                "/home/jinyang/workspace/SIME-dev/simulation/out/can_image_delta_dp_ext_64to16_vib_klw-1e-3_ns-1_multi_round/full_run_multi_round.log": {
                    "ours": ("$\\alpha$=1.0", '#e377c2', '--'),
                },
            },
            task_name = None,
            save_name = "ns",
            ylim_low = 0.45,
            ylim_high = 0.95,
        ),
        dict(
            log_dict = {
                "/home/jinyang/workspace/SIME-dev/simulation/out/can_image_delta_dp_ext_64to16_vib_klw-1e-3_ns-2_multi_round/full_run_multi_round.log": {
                    "ours": ("$\\beta$=1e-3", '#1f77b4', '-'),
                },
                "/home/jinyang/workspace/SIME-dev/simulation/out/can_image_delta_dp_ext_64to16_vib_klw-1e-4_ns-2_multi_round/full_run_multi_round.log": {
                    "ours": ("$\\beta$=1e-4", '#17becf', '--'),
                },
                "/home/jinyang/workspace/SIME-dev/simulation/out/can_image_delta_dp_ext_64to16_vib_klw-1e-2_ns-2_multi_round/full_run_multi_round.log": {
                    "ours": ("$\\beta$=1e-2", '#e377c2', '--'),
                },
            },
            task_name = None,
            save_name = "klw",
            ylim_low = 0.45,
            ylim_high = 0.95,
        ),
        dict(
            log_dict = {
                "/home/jinyang/workspace/SIME-dev/simulation/out/can_image_delta_dp_ext_64to16_vib_klw-1e-3_ns-2_multi_round/full_run_multi_round.log": {
                    "ours": ("$d$=16", '#1f77b4', '-'),
                },
                "/home/jinyang/workspace/SIME-dev/simulation/out/can_image_delta_dp_ext_64to32_vib_klw-1e-3_ns-2_multi_round/full_run_multi_round.log": {
                    "ours": ("$d$=32", '#17becf', '--'),
                },
                "/home/jinyang/workspace/SIME-dev/simulation/out/can_image_delta_dp_ext_64to64_vib_klw-1e-3_ns-2_multi_round/full_run_multi_round.log": {
                    "ours": ("$d$=64", '#e377c2', '--'),
                },
            },
            task_name = None,
            save_name = "ld",
            ylim_low = 0.45,
            ylim_high = 0.95,
        ),
    ]

    for local_log in local_logs:
        log_dict = local_log["log_dict"]
        task_name = local_log["task_name"]
        ylim_low = local_log["ylim_low"]
        ylim_high = local_log["ylim_high"]
        save_name = local_log["save_name"]

        plt.figure(figsize=(7, 5))
        if task_name is not None:
            plt.title(f"{task_name}", fontsize=30)
        x_len = 0
        for log_path, name_dict in log_dict.items():
            baseline_mean, baseline_upper, baseline_lower, baseline_std, ours_mean, ours_upper, ours_lower, ours_std = decode_log(log_path)
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

        plt.xlabel("Round", fontsize=23)
        plt.ylabel("Success Rate", fontsize=23)
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)
        plt.legend(fontsize=23, loc="lower right")
        plt.grid(axis='y')

        plt.xlim(0, x_len-1)
        plt.ylim(ylim_low, ylim_high)
        plt.xticks(np.arange(0, x_len, 1))
        plt.yticks(np.arange(ylim_low, ylim_high + 0.01, (ylim_high-ylim_low)/5))

        os.makedirs("figures", exist_ok=True)
        plt.savefig(f"figures/Ablation-{save_name}-{task_name}.pdf", format="pdf")
        plt.show()
