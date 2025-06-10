import time
import torch
import glob
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import DataParallel

########################################################
### Color plate
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_SCARLET_RED_0 = (255, 0, 0)
COLOR_SKY_BLUE_0 = (0, 0, 255)
COLOR_GREEN_0 = (0, 255, 0)
COLOR_LIGHT_GRAY = (196, 196, 196)
COLOR_PINK = (255,19, 203)
COLOR_BUTTER_0 = (252, 233, 79)
COLOR_ORANGE_0 = (252, 175, 62)
COLOR_CHOCOLATE_0 = (233, 185, 110)
COLOR_CHAMELEON_0 = (138, 226, 52)
COLOR_PLUM_0 = (173, 127, 168)
COLOR_ALUMINIUM_0 = (238, 238, 236)

color_plate = {
    '0': COLOR_SCARLET_RED_0,
    '1': COLOR_GREEN_0,
    '2': COLOR_SKY_BLUE_0,
    '3': COLOR_ALUMINIUM_0,
    '4': COLOR_CHAMELEON_0,
    '5': COLOR_CHOCOLATE_0,
    '6': COLOR_LIGHT_GRAY,
    '7': COLOR_PLUM_0,
    '8': COLOR_ORANGE_0,
    '9': COLOR_BUTTER_0
}

########################################################

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    l.sort(key=alphanum_key)

def experiment_log_path(experiment_path, dataset_name):
    # WARNING if the path exist without checkpoints it breaks
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
    return os.path.join(experiment_path, dataset_name + '_result.csv')

class DataParallelWrapper(DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def __len__(self):
        return len(self.module)
#@timeit
def print_train_info(log_frequency, batch_size, model,
                     time_start, acc_time, loss_data, steer_loss_data, acc_loss_data, brake_loss_data=None):

    epoch = model._current_iteration*batch_size / len(model)

    time_end = time.time()
    batch_time = time_end - time_start
    acc_time += batch_time
    if model._current_iteration % log_frequency == 0:

        if brake_loss_data is not None:
            print ("Training epoch {:.2f}, iteration {}, Loss {:.3f}, Steer Loss {:.3f}, Throttle Loss {:.3f}, , Brake Loss {:.3f}, {:.2f} steps/s".format(
                epoch, model._current_iteration, loss_data, steer_loss_data, acc_loss_data, brake_loss_data, (log_frequency / acc_time)))
        else:
            print ("Training epoch {:.2f}, iteration {}, Loss {:.3f}, Steer Loss {:.3f}, Acc Loss {:.3f}, {:.2f} steps/s".format(
                epoch, model._current_iteration, loss_data, steer_loss_data, acc_loss_data,(log_frequency / acc_time)))
        print_alpha_scale(model)
        print()
        acc_time = 0.0

    return acc_time


#@timeit
def test_stop(number_of_data, iterated_data):

    if number_of_data != 0 and \
            iterated_data >= number_of_data:
        return True
    return False


def generate_specific_rows(filePath, row_indices=[]):
    with open(filePath) as f:

        # using enumerate to track line no.
        for i, line in enumerate(f):

            # if line no. is in the row index list, then return that line
            if i in row_indices:
                yield line

def read_results(result_file, metric=''):
    head = np.genfromtxt(generate_specific_rows(result_file, row_indices=[0]), delimiter=',', dtype='str')
    col = [h.strip() for h in list(head)].index(metric)
    res = np.loadtxt(result_file, delimiter=",", skiprows=1)
    if len(res.shape) == 1:
        res = np.expand_dims(res, axis=0)

    return res[:, col]


def draw_offline_evaluation_results(experiment_path, metrics_list, x_range=[0, 10]):
    for metric in metrics_list:
        print('drawing results graph for ', experiment_path, 'of', metric)
        results_files = glob.glob(os.path.join(experiment_path, '*.csv'))
        for results_file in results_files:
            plt.figure()
            output_path = os.path.join(experiment_path, results_file.split('/')[-1].split('.')[-2]+ '_' + metric+'.jpg')
            results_list = read_results(results_file, metric=metric)
            epochs_list = read_results(results_file, metric='epoch')
            plt.ylabel(metric, fontsize=15)
            plt.plot(epochs_list, results_list)
            for i in range(len(results_list)):
                if results_list[i] == min(results_list):
                    plt.text(epochs_list[i], results_list[i], str(results_list[i]), color='blue',
                             fontweight='bold')
                    plt.plot(epochs_list[i], results_list[i], color='blue', marker='*')
            plt.xlabel('Epoch', fontsize=15)
            plt.xlim(left=x_range[0], right=x_range[-1])
            plt.title(results_file.split('/')[-1].split('.')[-2])
            plt.savefig(output_path)
            plt.close()


import os

def write_model_results(experiment_path: str, model_name: str, results_dict: dict, results_loss_dict: dict, acc_as_action: bool = False):
    for dataset_name, results in results_dict.items():
        results_file_csv = experiment_log_path(experiment_path, dataset_name)
        loss_dict = results_loss_dict

        alpha_keys  = sorted(k for k in loss_dict if k.startswith("Loss_Alpha"))
        scale_keys  = sorted(k for k in loss_dict if k.startswith("Loss_Scale"))
        loss_keys   = alpha_keys + scale_keys

        if not os.path.exists(results_file_csv):
            header = ["iteration", "epoch"]

            if acc_as_action:
                header += ["MAE_steer", "MAE_acceleration", "MAE"]
            else:
                header += ["MAE_steer", "MAE_throttle", "MAE_brake", "MAE"]

            header += loss_keys
            with open(results_file_csv, "a") as f:
                f.write(", ".join(header) + "\n")

        row = [
            str(results["iteration"]),
            f"{results['epoch']:.2f}"
        ]

        if acc_as_action:
            mae_block = results[model_name]
            row += [f"{mae_block['MAE_steer']:.4f}",
                    f"{mae_block['MAE_acceleration']:.4f}",
                    f"{mae_block['MAE']:.4f}"]
        else:
            mae_block = results[model_name]
            row += [f"{mae_block['MAE_steer']:.4f}",
                    f"{mae_block['MAE_throttle']:.4f}",
                    f"{mae_block['MAE_brake']:.4f}",
                    f"{mae_block['MAE']:.4f}"]

        for k in loss_keys:
            val = loss_dict.get(k, "")
            if isinstance(val, float):
                row.append(f"{val:.6f}")
            else:
                row.append(str(val))

        with open(results_file_csv, "a") as f:
            f.write(", ".join(row) + "\n")

        print(f"Results saved to: {results_file_csv}")


def eval_done(experiment_path, dataset_paths, epoch):
    results_files = glob.glob(os.path.join(experiment_path, '*.csv'))
    for dataset_path in dataset_paths:
        if os.path.join(experiment_path, dataset_path.split('/')[-1]+'_result.csv') not in results_files:
            return False
        else:
            epochs_list = read_results(os.path.join(experiment_path, dataset_path.split('/')[-1]+'_result.csv'), metric='epoch')
            if float(epoch) in epochs_list:
                return True
            else:
                return False


def is_result_better(experiment_path, model_name, dataset_name):
    results_list= read_results(os.path.join(experiment_path, dataset_name + '_result.csv'), metric='MAE')
    iter_list= read_results(os.path.join(experiment_path, dataset_name + '_result.csv'), metric='iteration')
    epoch_list= read_results(os.path.join(experiment_path, dataset_name + '_result.csv'), metric='epoch')
    if len(results_list) == 1:  # There is just one result so we save the check sure.
        return True
    if results_list[-1] < min(results_list[:-1]):
        print("Result for {} at iteration {} / epoch {} is better than the previous one. SAVE".format(model_name, iter_list[-1], epoch_list[-1]))
        return True
    return False


def extract_targets(data, targets=[], ignore=[]):

    """
    Method used to get to know which positions from the dataset are the targets
    for this experiments
    Args:

    Returns:
        the float data that is actually targets

    Raises
        value error when the configuration set targets that didn't exist in metadata
    """

    targets_vec = []
    for target_name in targets:
        if target_name in ignore:
            continue
        targets_vec.append(data[target_name])

    return torch.stack(targets_vec, 1).float().squeeze()


def extract_other_inputs(data, other_inputs=[], ignore=[]):
    """
    Method used to get to know which positions from the dataset are the inputs
    for this experiments
    Args:

    Returns:
        the float data that is actually targets

    Raises
        value error when the configuration set targets that didn't exist in metadata
    """

    inputs_vec = []
    for input_name in other_inputs:
        if input_name in ignore:
            continue
        inputs_vec.append(data[input_name])
    return torch.stack(inputs_vec, 1).float()


def extract_commands(data):
    return torch.stack(data, 1).float()

def add_alpha_scale_results(model):
    """
    Collect per-element values produced by model._criterion.{alpha, scale}.

    Returns
    -------
    dict
        Keys look like:
            "Loss_Alpha_0", "Loss_Alpha_1", ...
            "Loss_Scale_0", "Loss_Scale_1", ...
        Each maps to a Python float (or whatever the call returns if it
        isn’t tensor-like).
    """
    results = {}

    for name in ("alpha", "scale"):
        if not hasattr(model._criterion, name):
            continue

        attr = getattr(model._criterion, name)
        value = attr() if callable(attr) else attr

        if torch.is_tensor(value):
            scalars = value.detach().cpu().flatten().tolist()
        elif isinstance(value, (list, tuple)):
            scalars = list(value)
        else:
            scalars = [value]

        base = f"Loss_{name.capitalize()}_"
        for idx, scalar in enumerate(scalars):
            results[f"{base}{idx}"] = scalar
    return results

def print_alpha_scale(model):
    """
    Function to check if we should be printing alpha and scale values.
    This only happens when the loss function is adaptative.
    Args:

    Returns:
    """
    if hasattr(model._criterion, "alpha") and hasattr(model._criterion, "scale"):
        alpha_attr = getattr(model._criterion, "alpha")
        scale_attr = getattr(model._criterion, "scale")

        if callable(alpha_attr) and callable(scale_attr):
            alpha = alpha_attr()
            scale = scale_attr()

            print(f'Alpha: {alpha}, Scale: {scale}')

def tensor_to_dict(tensor, prefix="Loss_Alpha"):
    flat_tensor = tensor.flatten()
    return {f"{prefix}{i}": float(val) for i, val in enumerate(flat_tensor)}

def return_alpha_scale_dict(model):
    """
    Function to check if we should be printing alpha and scale values.
    This only happens when the loss function is adaptative.
    Args:

    Returns: alpha and scale values
    """
    if hasattr(model._criterion, "alpha") and hasattr(model._criterion, "scale"):
        alpha_attr = getattr(model._criterion, "alpha")
        scale_attr = getattr(model._criterion, "scale")

        if callable(alpha_attr) and callable(scale_attr):
            alpha = alpha_attr()
            scale = scale_attr()

            return {**tensor_to_dict(alpha, prefix="Loss_Alpha"), **tensor_to_dict(scale, prefix="Loss_Scale")}
    
    return {}

