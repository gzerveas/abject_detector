from __future__ import print_function
from __future__ import division

import logging
logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Loading packages ...")

import os
import sys
import torch
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
import time
import traceback
import argparse
import warnings

# Project modules
import utils
import model
import training
import dataset


# Handle command line arguments
parser = argparse.ArgumentParser(description='Run a complete training pipeline. Optionally, a JSON configuration file can be used, to overwrite command-line arguments.')
# I/O
parser.add_argument('--config', dest='config_filepath',
                    help='Configuration .json file (optional). Overwrites existing command-line args!')
parser.add_argument('--output_dir', default='/users/gzerveas/data/gzerveas/pruned_med_img_seg/output',
                    help='Root output directory. Must exist. Time-stamped directories will be created inside.')
parser.add_argument('--data_dir', default='/gpfs/data/ceickhof/GAILA',
                    help='Data directory')
parser.add_argument('--load_model',
                    help='Path to pretrained model')
parser.add_argument('--debug_size', type=int,
                    help='For rapid testing purposes (e.g. debugging), limit training set to a small random sample')
parser.add_argument('--name', dest='experiment_name', default='',
                    help='A string identifier/name for the experiment to be run - it will be appended to the output directory name, before the timestamp')
parser.add_argument('--no_timestamp', action='store_true',
                    help='If set, a timestamp will not be appended to the output directory name')
parser.add_argument('--records_file', default='/gpfs/data/ceickhof/GAILA/records.xls',
                    help='Excel file keeping all records of experiments')
# Training
parser.add_argument('--gpu', type=str, default='0', 
                    help='GPU index, -1 for CPU')
parser.add_argument('--num_workers', type=int, default=4,
                    help='dataloader threads. 0 for single-thread.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of training epochs')
parser.add_argument('--lr', type=float, default=1.25e-4, 
                             help='learning rate for batch size 32.')
parser.add_argument('--lr_step', type=str, default='90,120',
                             help='Comma separated string of epochs when to reduce learning rate by a factor of 10')
parser.add_argument('--batch_size', type=int, default=10,
                    help='Training batch size')
parser.add_argument('--img_size', type=int, default=128,
                    help='Final size of training images (after downsampling)')
parser.add_argument('--seed',
                    help='Seed used for splitting sets. None by default, set to an integer for reproducibility')

args = parser.parse_args()

NUM_CLASSES = 16

if args.task == '3d':
    heads = {'hm': NUM_CLASSES, 'dep': 1, 'rot': 8, 'dim': 3}
    if args.reg_bbox:
        heads.update({'wh': 2})
    if args.reg_offset:
        heads.update({'reg': 2})
elif args.task == 'localization':
    heads = {'hm': NUM_CLASSES, 'wh': 2}


def setup(args):
    """Prepare training session: read configuration from file (takes precedence), create directories.
    Input:
        args: arguments object from argparse
    Returns:
        config: configuration dictionary
    """

    config = args.__dict__  # configuration dictionary

    if args.config_filepath is not None:
        logger.info("Reading configuration ...")
        try:  # dictionary containing the entire configuration settings in a hierarchical fashion
            config.update(utils.load_config(args.config_filepath))
        except:
            logger.critical("Failed to load configuration file. Check JSON syntax and verify that files exist")
            traceback.print_exc()
            sys.exit(1)

    # Create output directory
    initial_timestamp = datetime.now()
    output_dir = config['output_dir']
    if not os.path.isdir(output_dir):
        raise IOError(
            "Root directory '{}', where the directory of the experiment will be created, must exist".format(output_dir))

    output_dir = os.path.join(output_dir, config['experiment_name'])

    formatted_timestamp = initial_timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    config['initial_timestamp'] = formatted_timestamp
    if (not config['no_timestamp']) or (len(config['experiment_name']) == 0):
        output_dir += "_" + formatted_timestamp
    utils.create_dirs([output_dir])
    config['output_dir'] = output_dir

    # Save configuration as a (pretty) json file
    with open(os.path.join(output_dir, 'configuration.json'), 'w') as fp:
        json.dump(config, fp, indent=4, sort_keys=True)

    logger.info("Stored configuration file in '{}'".format(output_dir))

    return config





def main():

    start_time = time.time()

    config = setup(args)

    torch.manual_seed(config['seed'])

    device = torch.device('cuda' if (torch.cuda.is_available() and config['gpu'] != '-1') else 'cpu')

    # Create or load model
    # Fail early if something is amiss
    logger.info("Creating model ...")
    my_model = model.DCN_Detector(opt.heads, opt.head_conv)
    optimizer = torch.optim.Adam(my_model.parameters(), opt.lr)
    start_epoch = 0
    if args.load_model:
        my_model, optimizer, start_epoch = utils.load_model(my_model, args.load_model, optimizer, args.resume, args.lr, args.lr_step)
    my_model.to(device)

    # Initialize data generators
    logger.info("Loading and preprocessing data ...")
    my_dataset = dataset.ObjDetDataset()

    val_loader = torch.utils.data.DataLoader(
        my_dataset(opt, 'val'), 
        batch_size=1, 
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    trainer = training.Trainer(opt, my_model, optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

    if opt.test: # Only evaluate and skip training
        _, preds = trainer.val(0, val_loader)
        val_loader.dataset.run_eval(preds, opt.save_dir)
        return

    train_loader = torch.utils.data.DataLoader(
        my_dataset(opt, 'train'), 
        batch_size=opt.batch_size, 
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )

    logger.info('Starting training...')
    best = 1e10
    metrics = []  # list of lists: for each epoch, stores metrics like accuracy, DICE,
    it = tqdm(range(1, config["epochs"] + 1), desc='Training', ncols=0) 
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        mark = epoch if opt.save_all else 'last'
        log_dict_train, _ = trainer.train(epoch, train_loader)
        logger.info('epoch: {} |'.format(epoch))
        for k, v in log_dict_train.items():
            logger.info('{} {:8f} | '.format(k, v))
        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), 
                        epoch, my_model, optimizer)
            with torch.no_grad():
                log_dict_val, preds = trainer.val(epoch, val_loader)
            for k, v in log_dict_val.items():
                logger.scalar_summary('val_{}'.format(k), v, epoch)
                logger.write('{} {:8f} | '.format(k, v))
            if log_dict_val[opt.metric] < best:
                best = log_dict_val[opt.metric]
                save_model(os.path.join(opt.save_dir, 'model_best.pth'), 
                        epoch, my_model)
        else:
            save_model(os.path.join(opt.save_dir, 'model_last.pth'), 
                        epoch, my_model, optimizer)
        if epoch in opt.lr_step:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), epoch, my_model, optimizer)
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    
        metrics.append([epoch, accuracy, dice, precision, recall])

        it.set_postfix(
            Accuracy='{:.3f}%'.format(accuracy * 100),
            Max_Accuracy='{:.3f}%'.format(max_acc * 100),
            Dice='{:.3f}%'.format(dice * 100),
            Max_Dice='{:.3f}%'.format(max_dice * 100),
            Precision='{:.3f}%'.format(precision * 100),
            Max_Precision='{:.3f}%'.format(max_prec * 100),
            Recall='{:.3f}%'.format(recall * 100),
            Max_Recall='{:.3f}%'.format(max_rec * 100),
            Last_pruning='{:2d}'.format(latest_pruning),
            Pruned='{:.2f}%'.format(pruned_ratio * 100))

    # Export evolution of metrics over epoch
    header = ["Epoch", "Accuracy", "DICE", "Precision", "Recall", "Pruned ratio"]
    metrics_filepath = os.path.join(config["output_dir"], "metrics_" + config["experiment_name"] + ".xls")
    book = utils.export_performance_metrics(metrics_filepath, metrics, header, sheet_name="metrics")

    # Export best metrics per pruning level
    header = ["Pruned ratio", "DICE", "Accuracy", "Precision", "Recall"]
    utils.export_performance_metrics(metrics_filepath, best_at_prune_level, header, book, sheet_name="best_metrics")

    # Export record metrics to a file accumulating records from all experiments
    utils.register_record(config["records_file"], config["initial_timestamp"], config["experiment_name"], metrics)

    logger.info('All Done!')

    total_runtime = time.time() - start_time
    logger.info(
        "Total runtime: {} hours, {} minutes, {} seconds\n".format(total_runtime // 3600, (total_runtime // 60) % 60,
                                                                   total_runtime % 60))


if __name__ == '__main__':

    main()

