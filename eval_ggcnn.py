import argparse
import logging
import os.path

import torch
import torch.utils.data as data

from models.common import post_process_output
from utils.dataset_processing import evaluation, grasp
from utils.data import get_dataset

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate GG-CNN')

    # Network
    parser.add_argument('--network', type=str, help='Path to saved network to evaluate')

    # Dataset & Data & Training
    parser.add_argument('--dataset', type=str, help='Dataset Name ("cornell" or "jaquard")')
    parser.add_argument('--dataset-path', type=str, help='Path to dataset')
    parser.add_argument('--use-depth', type=int, default=1, help='Use Depth image for evaluation (1/0)')
    parser.add_argument('--use-rgb', type=int, default=0, help='Use RGB image for evaluation (0/1)')
    parser.add_argument('--augment', action='store_true', help='Whether data augmentation should be applied')
    parser.add_argument('--split', type=float, default=0.9, help='Fraction of data for training (remainder is validation)')
    parser.add_argument('--ds-rotate', type=float, default=0.0,
                        help='Shift the start point of the dataset to use a different test/train split')
    parser.add_argument('--num-workers', type=int, default=8, help='Dataset workers')
    parser.add_argument('--output-size', type=int, default=300,
                        help='the output size of the network, determining the cropped size of dataset images')

    parser.add_argument('--n-grasps', type=int, default=1, help='Number of grasps to consider per image')
    parser.add_argument('--iou-eval', action='store_true', help='Compute success based on IoU metric.')
    parser.add_argument('--jacquard-output', action='store_true', help='Jacquard-dataset style output')
    parser.add_argument('--vis', action='store_true', help='Visualise the network output')

    args = parser.parse_args()

    if args.jacquard_output and args.dataset != 'jacquard':
        raise ValueError('--jacquard-output can only be used with the --dataset jacquard option.')
    if args.jacquard_output and args.augment:
        raise ValueError('--jacquard-output can not be used with data augmentation.')

    return args


if __name__ == '__main__':
    args = parse_args()

    # Load Network
    device = torch.device("cuda:0")
    # device = torch.device("cpu")
    net = torch.load(args.network, map_location=device)
    net.to(device)
    net.eval()
    # Load Dataset
    logging.info('Loading {} Dataset...'.format(args.dataset.title()))
    Dataset = get_dataset(args.dataset)
    test_dataset = Dataset(file_path=args.dataset_path, output_size=args.output_size,
                           start=args.split, end=1.0, ds_rotate=args.ds_rotate,
                           random_rotate=args.augment, random_zoom=args.augment,
                           include_depth=args.use_depth, include_rgb=args.use_rgb)
    test_data = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers
    )
    logging.info('Done')

    results = {'correct': 0, 'failed': 0}

    if args.jacquard_output:
        jo_fn = args.network + '_jacquard_output.txt'
        with open(jo_fn, 'w') as f:
            pass

    with torch.no_grad():
        for idx, (x, y, didx, rot, zoom) in enumerate(test_data):
            logging.info('Processing {}/{}'.format(idx+1, len(test_data)))
            xc = x.to(device)
            yc = [yi.to(device) for yi in y]
            lossd = net.compute_loss(xc, yc)

            q_img, ang_img, width_img = post_process_output(lossd['pred']['pos'], lossd['pred']['cos'],
                                                        lossd['pred']['sin'], lossd['pred']['width'])

            if args.iou_eval:
                s = evaluation.calculate_iou_match(q_img, ang_img, test_data.dataset.get_gtbb(didx, rot, zoom),
                                                   no_grasps=args.n_grasps,
                                                   grasp_width=width_img,
                                                   )
                if s:
                    results['correct'] += 1
                else:
                    results['failed'] += 1

            if args.jacquard_output:
                grasps = grasp.detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=1)
                with open(jo_fn, 'a') as f:
                    for g in grasps:
                        f.write(test_data.dataset.get_jname(didx) + '\n')
                        f.write(g.to_jacquard(scale=1024 / 300) + '\n')

            if args.vis:
                evaluation.plot_output(test_data.dataset.get_rgb(didx, rot, zoom, normalise=False),
                                       test_data.dataset.get_depth(didx, rot, zoom), q_img,
                                       ang_img, no_grasps=args.n_grasps, grasp_width_img=width_img)

    if args.iou_eval:
        result_info = 'IOU Results: %d/%d = %f' % (
        results['correct'], results['correct'] + results['failed'], results['correct'] / (results['correct'] + results['failed']))
        logging.info(result_info)
        network_name = os.path.basename(args.network)
        save_dir = os.path.join('eval_results', f'{network_name}.txt')
        with open(save_dir, 'w') as f:
            f.write(f'Evaluation result for network {args.network}\n')
            f.write(f'Evaluation on {args.dataset} dataset\n')
            f.write(f'Result info is below ===> \n')
            f.write(result_info)
            f.write('\n')

    if args.jacquard_output:
        logging.info('Jacquard output saved to {}'.format(jo_fn))