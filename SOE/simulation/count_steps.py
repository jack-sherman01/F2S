import h5py
import argparse

def count_steps(file_path, show_print=False):
    total_steps = 0
    with h5py.File(file_path, 'r') as f:
        for demo in f["data"].keys():
            action_len = f["data"][demo]["actions"].shape[0]
            if show_print:
                print(f"Demo {demo} has {action_len} actions.")
            total_steps += action_len
    if show_print:
        print(f"Total steps across all demos: {total_steps}")
    return total_steps

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to the h5 file")
    args = parser.parse_args()
    count_steps(args.file, show_print=True)