import os
import torch
import wandb
import time
import shutil
from configs import g_conf, set_type_of_process, merge_with_yaml
from network.models_console import Models
from _utils.training_utils import seed_everything, DataParallelWrapper, check_saved_checkpoints, update_learning_rate
from _utils.utils import extract_targets, extract_other_inputs, extract_commands, print_train_info, return_alpha_scale_dict, test_stop
from _utils.evaluation import evaluation_saving
from logger import _logger

def train_upstream_task(model, optimizer):
    """
    Upstream task is for training your model

    """
    early_stopping_flags = []
    acc_time = 0.0
    time_start = time.time()

    while True:
        # we get dataloader of the model
        dataloader = model._train_loader
        for data in dataloader:
            early_stopping_flags = evaluation_saving(model, optimizer, early_stopping_flags, save_all_checkpoints=True)
            if early_stopping_flags and all(early_stopping_flags[-int(g_conf.EARLY_STOPPING_PATIENCE):]):
                print(' Apply early stopping, training stopped !')
                break

            if g_conf.LEARNING_RATE_DECAY:
                if model._done_epoch in g_conf.LEARNING_RATE_DECAY_EPOCHES and \
                        ((model._current_iteration-1)*g_conf.BATCH_SIZE <= len(model) * model._done_epoch):
                    update_learning_rate(optimizer, minimumlr=g_conf.LEARNING_RATE_MINIMUM)

            src_images = [[data['current'][i][camera_type].cuda() for camera_type in g_conf.DATA_USED] for i in range(len(data['current']))]
            src_directions = [extract_commands(data['current'][i]['can_bus']['direction']).cuda() for i in
                              range(len(data['current']))]
            src_s = [extract_other_inputs(data['current'][i]['can_bus'], g_conf.OTHER_INPUTS,
                                     ignore=['direction']).cuda() for i in range(len(data['current']))]
            if g_conf.ENCODER_OUTPUT_STEP_DELAY > 0 or g_conf.DECODER_OUTPUT_FRAMES_NUM != g_conf.ENCODER_INPUT_FRAMES_NUM:
                tgt_a = [extract_targets(data['future'][i]['can_bus_future'], g_conf.TARGETS).cuda() for i in range(len(data['future']))]
            else:
                tgt_a = [extract_targets(data['current'][i]['can_bus'], g_conf.TARGETS).cuda() for i in range(len(data['current']))]

            action_outputs = model.forward(src_images, src_directions, src_s)
            loss_params = {
                'action_output': action_outputs,
                'targets_action': tgt_a,
                'variable_weights': g_conf.LOSS_WEIGHT
            }

            if g_conf.ACCELERATION_AS_ACTION:
                loss, steer_loss, acceleration_loss, diff = model.loss(loss_params)
                acc_time = print_train_info(g_conf.TRAIN_PRINT_LOG_FREQUENCY, g_conf.BATCH_SIZE, model, time_start,
                                            acc_time, loss.item(), steer_loss.item(), acceleration_loss.item())
            else:
                loss, steer_loss, throttle_loss, brake_loss, diff = model.loss(loss_params)
                acc_time = print_train_info(g_conf.TRAIN_PRINT_LOG_FREQUENCY, g_conf.BATCH_SIZE, model, time_start,
                                            acc_time, loss.item(), steer_loss.item(), throttle_loss.item(), brake_loss.item)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            time_start = time.time()
            """
            ################################################
                Add Weights and Biases logs
            #################################################
            """

            standard_log = {
                "loss": loss.item(),
                "loss_steer": steer_loss.item(),
                "loss_acceleration": acceleration_loss.item(),
                "steer_distribution": wandb.Histogram(diff[:,0].tolist()),
                "acceleration_distribution": wandb.Histogram(diff[:,1].tolist()),
                "current_iteration": model._current_iteration,
                "model_epoch": model._done_epoch,
            }
            loss_params_log = return_alpha_scale_dict(model)
            total_log = {**standard_log, **loss_params_log}

            wandb.log(total_log, step=model._current_iteration)

            """
            ################################################
                Adding tensorboard logs
            #################################################
            """
            _logger.add_scalar('Loss', loss.item(), model._current_iteration)

            ## Adding loss to tensorboard
            _logger.add_scalar('Loss_steer', steer_loss.item(), model._current_iteration)
            if g_conf.ACCELERATION_AS_ACTION:
                _logger.add_scalar('Loss_acceleration', acceleration_loss.item(), model._current_iteration)
            else:
                _logger.add_scalar('Loss_throttle', throttle_loss.item(), model._current_iteration)
                _logger.add_scalar('Loss_brake', brake_loss.item(), model._current_iteration)

            if test_stop(g_conf.NUMBER_EPOCH*len(model), model._current_iteration * g_conf.BATCH_SIZE):
                print('')
                print('Training finished !!')
                break
            model._current_iteration += 1
            model._done_epoch = (model._current_iteration*g_conf.BATCH_SIZE // len(model))

            del src_images
            del src_directions
            del tgt_a
            del src_s
            del action_outputs

        else:
            continue
        break


# The main function maybe we could call it with a default name
def execute(gpus_list, exp_batch, exp_name):
    """
        The main training function for decoder.
    Args:
        gpus_list: The list of all GPU can be used
        exp_batch: The folder with the experiments
        exp_name: the alias, experiment name

    Returns:
        None

    """
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
    print(torch.cuda.device_count(), 'GPUs to be used: ', gpus_list)
    merge_with_yaml(os.path.join('configs', exp_batch, exp_name + '.yaml'))
    shutil.copyfile(os.path.join('configs', exp_batch, exp_name + '.yaml'),
                    os.path.join(os.environ["TRAINING_RESULTS_ROOT"], '_results',
                                 g_conf.EXPERIMENT_BATCH_NAME, g_conf.EXPERIMENT_NAME, exp_name + '.yaml'))
    set_type_of_process('train_val', root= os.environ["TRAINING_RESULTS_ROOT"])
    seed_everything(seed=g_conf.MAGICAL_SEED)

    model = Models(g_conf.MODEL_TYPE, g_conf.MODEL_CONFIGURATION)
    # print("===================== Model Configuration =====================")
    # print("")
    # print(model)

    # Weights and Biases SetUp
    wandb.login()

    wandb.init(
        project=os.environ["WANDB_PROJECT"],
        config={
            "epochs": g_conf.NUMBER_EPOCH,
            "batch_size": g_conf.BATCH_SIZE,
            "loss_name": g_conf.LOSS,
            "network_learning_rate": g_conf.LEARNING_RATE,
            "loss_learning_rate": g_conf.LOSS_LEARNING_RATE,
            "train_dataset_name": str(g_conf.TRAIN_DATASET_NAME),
            "validation_dataset_name": str(g_conf.VALID_DATASET_NAME),
            "magical_seed": g_conf.MAGICAL_SEED,
        },
    )

    # Set up learning rate for base parameters and loss_parameters
    num_params=0
    for param in model.parameters():
        num_params += param.numel()
    print('model params: ', num_params)

    loss_params = list(model.loss_params)
    base_params   = [p for p in model.parameters()
                    if id(p) not in {id(q) for q in loss_params}]

    assert len(base_params) + len(loss_params) == sum(1 for _ in model.parameters()) # Some parameters are missing or duplicated

    LOSS_LR  = g_conf.LOSS_LEARNING_RATE
    BASE_LR  = g_conf.LEARNING_RATE

    optimizer = torch.optim.AdamW(
        [
            {'params': base_params,  'lr': BASE_LR, "name": "network_params"},
            {'params': loss_params,  'lr': LOSS_LR, "name": "loss_params"},
        ],
    )
    
    # optimizer = torch.optim.AdamW(list(model.parameters()), lr=g_conf.LEARNING_RATE) - previous way to define optimizer (I'm going to keep it here just in case)

    if len(gpus_list) > 1 and g_conf.DATA_PARALLEL:
        print("Using multiple GPUs parallel! ")
        model = DataParallelWrapper(model)

    # To load a specific checkpoint
    if g_conf.LOAD_CHECKPOINT:
        latest_checkpoint = os.path.join(os.environ["TRAINING_RESULTS_ROOT"], '_results', g_conf.EXPERIMENT_BATCH_NAME,
                                                                g_conf.EXPERIMENT_NAME, 'checkpoints', g_conf.LOAD_CHECKPOINT)

    # To train model from scratch, or to resume training on a previous one
    elif g_conf.TRAINING_RESUME:
        latest_checkpoint = check_saved_checkpoints(os.path.join(os.environ["TRAINING_RESULTS_ROOT"], '_results', g_conf.EXPERIMENT_BATCH_NAME,
                                                                g_conf.EXPERIMENT_NAME, 'checkpoints'))
    else:
        latest_checkpoint = None

    if latest_checkpoint is not None:
        checkpoint = torch.load(latest_checkpoint)
        pretrained_dict = checkpoint['model']

        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(pretrained_dict)
        else:
            model.load_state_dict(pretrained_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
        # we manually move optimizer state to GPU memory after loading it from the checkpoint
        for state in optimizer.state.values():
            for k,v in state.items():
                if torch.is_tensor(v):
                    state[k]=v.cuda()
        for param_group in optimizer.param_groups:
            print('')
            print('    Resum training from epoch -> ', checkpoint['epoch'])
            print('    Resum the latest learning rate -> ', param_group['lr'])
            if g_conf.LEARNING_RATE_DECAY:
                print('      - learning rate decay at epoch', g_conf.LEARNING_RATE_DECAY_EPOCHES, ', minimum lr:', g_conf.LEARNING_RATE_MINIMUM)
            print('')
            print('=======================================================================================')
            print('')

        model._current_iteration = checkpoint['iteration'] + 1
        model._done_epoch = checkpoint['epoch']
    else:
        print('')
        print('    Training from epoch 0')
        print('    Initial learning rate -> ', g_conf.LEARNING_RATE)
        if g_conf.LEARNING_RATE_DECAY:
            print('      - learning rate decay at epoch', g_conf.LEARNING_RATE_DECAY_EPOCHES, ', minimum lr:', g_conf.LEARNING_RATE_MINIMUM)
        print('')
        print('=======================================================================================')
        print('')

    model.cuda()
    model.train()                           # PyTorch function to enables gradient modification
    train_upstream_task(model, optimizer)   # Training loop function