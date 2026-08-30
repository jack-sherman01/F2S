import glob
import os
import argparse

def clean_failure(dataset):
    demos = glob.glob(f'{dataset}/*')
    print(demos)
    rm_cnt = 0
    not_rm_cnt = 0
    for demo in demos:
        # print(demo)
        assert os.path.exists(os.path.join(demo, 'preference.txt'))
        with open(os.path.join(demo, 'preference.txt'), 'r') as f:
            preference = f.read()
        if preference == 'fail':
            print('Remove', demo)
            os.system(f'rm -rf {demo}')
            rm_cnt += 1
        elif preference == 'success':
            not_rm_cnt += 1
        else:
            print(f'Unknown preference: {preference} in {demo}')
            raise ValueError('Unknown preference')
    print("Remove count:", rm_cnt, "Not remove count:", not_rm_cnt)
    print('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Clean failure demos')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset')
    args = parser.parse_args()
    for dataset in glob.glob(args.dataset):
        clean_failure(dataset)