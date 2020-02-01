import logging
import argparse
import csv
import numpy as np
from pathlib import Path


def main(args):
    data_dir = args.data_dir

    paths = sorted(list(data_dir.glob('*')))

    csv_path = str(args.output_path)
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        train_list = []
        test_list = []

        for p in paths:
            if 'TRAIN' in p.parts[-1]:
                train_list.append(p)
            else:
                test_list.append(p)

        n_train = len(train_list)
        n_test = len(test_list)
        # print(n_train)
        # print(n_test)

        random_seed = 601985
        np.random.seed(int(random_seed))
        np.random.shuffle(train_list)
        np.random.shuffle(test_list)
       
        n_train = int(n_train * args.using_rate)
        n_test = int(n_test * args.using_rate)
        print(int(n_train * 0.9))
        print(int(n_train * 0.1))
        print(n_test)
        train_list = train_list[:n_train]
        test_list = test_list[:n_test]

        for idx, p in enumerate(train_list):
            if idx < int(n_train * 0.9):
                writer.writerow([str(p.parts[-1]), 'Training'])
            else:
                writer.writerow([str(p.parts[-1]), 'Validation'])
        for p in test_list:
            writer.writerow([str(p.parts[-1]), 'Testing'])


def _parse_args():
    parser = argparse.ArgumentParser(description="To create the split.csv.")
    parser.add_argument('using_rate', type=float, help='The rate of data be used')
    parser.add_argument('data_dir', type=Path,
                        help='The directory of the data.')
    parser.add_argument('output_path', type=Path,
                        help='The output path of the csv file.')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    args = _parse_args()
    main(args)
