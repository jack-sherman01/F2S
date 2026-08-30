import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42

def decode_log(log_path):
    with open(log_path, "r") as f:
        lines = f.readlines()

    baseline_list = []
    ours_list = []
    for line in lines:
        if ("eval" not in line and "ours" not in line) or "Exploration success rate" not in line:
            continue
        
        print(line.strip())
        round_num = int(line.split("_")[1].split(" ")[0])
        # print("round_num", round_num)
        update_baseline = "ours" not in line
        update_ours = "ours" in line
        val = float(line.strip().split(",")[-2].split("/")[-1])
        # val = float(line.strip().split(" ")[-1])
        print(f"update_baseline: {update_baseline}, update_ours: {update_ours}, val: {val}")

        if update_baseline:
            if len(baseline_list) <= round_num:
                baseline_list.append([val])
            else:
                baseline_list[round_num].append(val)
        if update_ours:
            if len(ours_list) <= round_num:
                ours_list.append([val])
            else:
                ours_list[round_num].append(val)

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
            ylim_high = 300,
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
            ylim_high = 100,
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
            ylim_high = 500,
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
            ylim_high = 400,
        ),
    ]

    means = {}
    uppers = {}
    lowers = {}
    stds = {}
    configs = {}
    x_len_max = 0


    for local_log in local_logs:
        log_dict = local_log["log_dict"]
        task_name = local_log["task_name"]
        ylim_low = local_log["ylim_low"]
        ylim_high = local_log["ylim_high"]

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
                means[f"{name_dict['baseline'][0]}"] = [baseline_mean] if f"{name_dict['baseline'][0]}" not in means else means[f"{name_dict['baseline'][0]}"] + [baseline_mean]
                uppers[f"{name_dict['baseline'][0]}"] = [baseline_upper] if f"{name_dict['baseline'][0]}" not in uppers else uppers[f"{name_dict['baseline'][0]}"] + [baseline_upper]
                lowers[f"{name_dict['baseline'][0]}"] = [baseline_lower] if f"{name_dict['baseline'][0]}" not in lowers else lowers[f"{name_dict['baseline'][0]}"] + [baseline_lower]
                stds[f"{name_dict['baseline'][0]}"] = [baseline_std] if f"{name_dict['baseline'][0]}" not in stds else stds[f"{name_dict['baseline'][0]}"] + [baseline_std]
                configs[f"{name_dict['baseline'][0]}"] = [name_dict['baseline']] if f"{name_dict['baseline'][0]}" not in configs else configs[f"{name_dict['baseline'][0]}"] + [name_dict['baseline']]
            if "ours" in name_dict:
                means[f"{name_dict['ours'][0]}"] = [ours_mean] if f"{name_dict['ours'][0]}" not in means else means[f"{name_dict['ours'][0]}"] + [ours_mean]
                uppers[f"{name_dict['ours'][0]}"] = [ours_upper] if f"{name_dict['ours'][0]}" not in uppers else uppers[f"{name_dict['ours'][0]}"] + [ours_upper]
                lowers[f"{name_dict['ours'][0]}"] = [ours_lower] if f"{name_dict['ours'][0]}" not in lowers else lowers[f"{name_dict['ours'][0]}"] + [ours_lower]
                stds[f"{name_dict['ours'][0]}"] = [ours_std] if f"{name_dict['ours'][0]}" not in stds else stds[f"{name_dict['ours'][0]}"] + [ours_std]
                configs[f"{name_dict['ours'][0]}"] = [name_dict['ours']] if f"{name_dict['ours'][0]}" not in configs else configs[f"{name_dict['ours'][0]}"] + [name_dict['ours']]

            # if "baseline" in name_dict:
            #     plt.plot(
            #         np.arange(len(baseline_mean)), np.array(baseline_mean), 
            #         linestyle=name_dict['baseline'][2], color=name_dict['baseline'][1], label=f"{name_dict['baseline'][0]}"
            #     )
            #     plt.fill_between(
            #         np.arange(len(baseline_mean)), np.array(baseline_upper), np.array(baseline_lower), 
            #         # np.arange(len(baseline_mean)), np.array(baseline_mean) + np.array(baseline_std), np.array(baseline_mean) - np.array(baseline_std), 
            #         color=name_dict['baseline'][1], alpha=0.1, edgecolor=None
            #     )
            # if "ours" in name_dict:
            #     plt.plot(
            #         np.arange(len(ours_mean)), np.array(ours_mean), 
            #         linestyle=name_dict['ours'][2], color=name_dict['ours'][1], label=f"{name_dict['ours'][0]}"
            #     )
            #     plt.fill_between(
            #         np.arange(len(ours_mean)), np.array(ours_upper), np.array(ours_lower), 
            #         # np.arange(len(ours_mean)), np.array(ours_mean) + np.array(ours_std), np.array(ours_mean) - np.array(ours_std),
            #         color=name_dict['ours'][1], alpha=0.1, edgecolor=None
            #     )
            x_len = max(x_len, len(baseline_mean), len(ours_mean))
        x_len_max = max(x_len_max, x_len)



    plt.figure(figsize=(7, 5))
    # plt.title(f"{task_name}", fontsize=30)
    # ylim_low = 0.45
    ylim_high = 300

    plt.plot(
        np.arange(x_len_max), np.mean(np.array(means["Ours"]), axis=0), 
        linestyle=configs["Ours"][0][2], color=configs["Ours"][0][1], label=f"Ours"
    )
    plt.fill_between(
        np.arange(x_len_max), np.mean(np.array(uppers["Ours"]), axis=0), np.mean(np.array(lowers["Ours"]), axis=0), 
        # np.arange(len(ours_mean)), np.array(ours_mean) + np.array(ours_std), np.array(ours_mean) - np.array(ours_std),
        color=configs["Ours"][0][1], alpha=0.1, edgecolor=None
    )
    plt.plot(
        np.arange(x_len_max), np.mean(np.array(means["DP"]), axis=0), 
        linestyle=configs["DP"][0][2], color=configs["DP"][0][1], label=f"DP"
    )
    plt.fill_between(
        np.arange(x_len_max), np.mean(np.array(uppers["DP"]), axis=0), np.mean(np.array(lowers["DP"]), axis=0), 
        # np.arange(len(baseline_mean)), np.array(baseline_mean) + np.array(baseline_std), np.array(baseline_mean) - np.array(baseline_std), 
        color=configs["DP"][0][1], alpha=0.1, edgecolor=None
    )
    plt.plot(
        np.arange(x_len_max), np.mean(np.array(means["SIME"]), axis=0), 
        linestyle=configs["SIME"][0][2], color=configs["SIME"][0][1], label=f"SIME"
    )
    plt.fill_between(
        np.arange(x_len_max), np.mean(np.array(uppers["SIME"]), axis=0), np.mean(np.array(lowers["SIME"]), axis=0), 
        # np.arange(len(ours_mean)), np.array(ours_mean) + np.array(ours_std), np.array(ours_mean) - np.array(ours_std),
        color=configs["SIME"][0][1], alpha=0.1, edgecolor=None
    )
    x_len = x_len_max


    plt.xlabel("Round", fontsize=23)
    plt.ylabel("Rollout Num", fontsize=23)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.legend(fontsize=23, loc="lower right")
    plt.grid(axis='y')

    plt.xlim(0, x_len-1)
    plt.ylim(ylim_low, ylim_high)
    plt.xticks(np.arange(0, x_len, 1))
    plt.yticks(np.arange(ylim_low, ylim_high + 0.01, (ylim_high-ylim_low)/5))

    os.makedirs("figures", exist_ok=True)
    plt.savefig(f"figures/Rollout_Num-Average.pdf", format="pdf")
    plt.show()
