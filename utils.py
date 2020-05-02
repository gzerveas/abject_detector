import xlrd
import xlwt
from xlutils.copy import copy
import json
import warnings
import numpy as np
import os
import torch


import logging
logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def save_model(path, epoch, model, optimizer=None):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {'epoch': epoch,
            'state_dict': state_dict}
    if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()
    torch.save(data, path)


def load_model(model, model_path, optimizer=None, resume=False, lr=None, lr_step=None):
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    # resume optimizer parameters
    if optimizer is not None and resume:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_lr = lr
            for step in lr_step:
                if start_epoch >= step:
                    start_lr *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = start_lr
            print('Resumed optimizer with start lr', start_lr)
        else:
            print('No optimizer parameters in checkpoint.')
    if optimizer is not None:
        return model, optimizer, start_epoch
    else:
        return model


def save_mask_images(pred_masks, lbl_masks, out_dir, epoch, test_IDs):

    pred_masks = np.concatenate(pred_masks)
    lbl_masks = np.concatenate(lbl_masks)

    for i, image, label in zip(test_IDs, pred_masks, lbl_masks):
        pred_img = (image * 255).astype(np.uint8)
        lbl_img = (label * 255).astype(np.uint8)

        directory = os.path.join(out_dir, "{}".format(i))
        if not os.path.exists(directory):
            os.makedirs(directory)

        with warnings.catch_warnings():  # stop complaining about low contrast
            warnings.simplefilter("ignore")
            imsave(os.path.join(directory, "pred_{}_epoch_{}.jpg".format(i, epoch)), pred_img)
            imsave(os.path.join(directory, "lbl_{}_epoch_{}.jpg".format(i, epoch)), lbl_img)
            # Save as side-by-side panels
            imsave(os.path.join(directory, "both_{}_epoch_{}.jpg".format(i, epoch)), np.hstack((pred_img, lbl_img)))


def load_config(config_filepath):
    """
    Using a json file with the master configuration (config file for each part of the pipeline),
    return a dictionary containing the entire configuration settings in a hierarchical fashion.
    """

    with open(config_filepath) as cnfg:
        config = json.load(cnfg)

    return config


def create_dirs(dirs):
    """
    Input:
        dirs: a list of directories to create, in case these directories are not found
    Returns:
        exit_code: 0 if success, -1 if failure
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)


def export_performance_metrics(filepath, metrics_table, header, book=None, sheet_name="metrics"):
    """Exports performance metrics on the validation set for all epochs to an excel file"""

    if book is None:
        book = xlwt.Workbook()  # new excel work book

    book = write_table_to_sheet([header] + metrics_table, book, sheet_name=sheet_name)

    book.save(filepath)
    logger.info("Exported per epoch performance metrics in '{}'".format(filepath))

    return book


def write_row(sheet, row_ind, data_list):
    """Write a list to row_ind row of an excel sheet"""

    row = sheet.row(row_ind)
    for col_ind, col_value in enumerate(data_list):
        row.write(col_ind, col_value)
    return


def write_table_to_sheet(table, work_book, sheet_name=None):
    """Writes a table implemented as a list of lists to an excel sheet in the given work book object"""

    sheet = work_book.add_sheet(sheet_name)

    for row_ind, row_list in enumerate(table):
        write_row(sheet, row_ind, row_list)

    return work_book


def export_record(filepath, values):
    """Adds a list of values as a bottom row of a table in a given excel file"""

    read_book = xlrd.open_workbook(filepath, formatting_info=True)
    read_sheet = read_book.sheet_by_index(0)
    last_row = read_sheet.nrows

    work_book = copy(read_book)
    sheet = work_book.get_sheet(0)
    write_row(sheet, last_row, values)
    work_book.save(filepath)


def register_record(filepath, timestamp, experiment_name, metrics):
    """
    Adds the best and final metrics of a given experiment as a record in an excel sheet with other experiment records.
    Creates excel sheet if it doesn't exist.
    """
    metrics = np.array(metrics)
    best_inds = np.argmax(metrics, axis=0)
    row_values = [timestamp, experiment_name,
                  metrics[best_inds[2], 2], metrics[best_inds[2], 0], metrics[best_inds[2], 5], metrics[-1, 2], metrics[-1, 5], metrics[-1, 0],
                  metrics[best_inds[1], 1],  metrics[-1, 1],  metrics[best_inds[3], 3], metrics[-1, 3],
                  metrics[best_inds[4], 4], metrics[-1, 4]]

    if not os.path.exists(filepath):  # Create a records file for the first time
        logger.warning("Records file '{}' does not exist! Creating new file ...")
        directory = os.path.dirname(filepath)
        if not os.path.exists(directory):
            os.makedirs(directory)
        header = ["Timestamp", "Name", "BEST DICE", "Epoch at BEST", "PrunedR at BEST", "Final DICE", "Final Pruned Ratio", "Final Epoch",
                  "Best Accuracy", "Final Accuracy", "Best Precision", "Final Precision", "Best Recall", "Final Recall"]
        book = xlwt.Workbook()  # excel work book
        book = write_table_to_sheet([header, row_values], book, sheet_name="records")
        book.save(filepath)
    else:
        try:
            export_record(filepath, row_values)
        except Exception as x:
            alt_path = os.path.join(os.path.dirname(filepath), "record_" + experiment_name)
            logger.error("Failed saving in: '{}'! Will save here instead: {}".format(filepath, alt_path))
            export_record(alt_path, row_values)

    logger.info("Exported performance record to '{}'".format(filepath))