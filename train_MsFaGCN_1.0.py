#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
import torch.optim as optim
from thop import profile
import numpy as np
import os
from time import time
import datetime
import shutil
import argparse
from tqdm import tqdm
import configparser
from model.MAMGCNtsp_gcn import make_model
from lib.utils import load_graphdata_channel1, get_adjacency_matrix, compute_val_loss_mstgcn, predict_and_save_results_mstgcn, load_adj, load_data_new, load_adjacent
from torch.utils.tensorboard import SummaryWriter
import ast
parser = argparse.ArgumentParser()
parser.add_argument("--config", default='configurations/NINGDE.conf', type=str,
                    help="configuration file path")
args = parser.parse_args()
config = configparser.ConfigParser()
print('Read configuration file: %s' % (args.config))
config.read(args.config)
data_config = config['Data']
training_config = config['Training']

adj_filename = data_config['adj_filename']
graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
if config.has_option('Data', 'id_filename'):
    id_filename = data_config['id_filename']
else:
    id_filename = None
seed = 2024
np.random.seed(seed)
torch.manual_seed(seed)
num_of_vertices = int(data_config['num_of_vertices'])
points_per_hour = int(data_config['points_per_hour'])
num_for_predict = int(data_config['num_for_predict'])
len_input = int(data_config['len_input'])
dataset_name = data_config['dataset_name']

model_name = training_config['model_name'] + "con"

ctx = training_config['ctx']
# os.environ["CUDA_VISIBLE_DEVICES"] = ctx
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:' + str(ctx))
print("CUDA:", USE_CUDA, DEVICE)

learning_rate = float(training_config['learning_rate'])
epochs = int(training_config['epochs'])
start_epoch = int(training_config['start_epoch'])
batch_size = int(training_config['batch_size'])
num_of_weeks = int(training_config['num_of_weeks'])
num_of_days = int(training_config['num_of_days'])
num_of_hours = int(training_config['num_of_hours'])
num_heads= int(training_config['num_heads'])
nb_chev_filter = int(training_config['nb_chev_filter'])
kernel_size = ast.literal_eval(training_config['kernel_size'])
in_channels = int(training_config['in_channels'])
nb_block = int(training_config['nb_block'])
K = int(training_config['K'])
dropout = float(training_config['dropout'])
folder_dir = '%s_h%dd%dw%d_channel%d_%e' % (model_name, num_of_hours, num_of_days, num_of_weeks, in_channels, learning_rate)
print('folder_dir:', folder_dir)
params_path = os.path.join('experiments', dataset_name, folder_dir)
print('params_path:', params_path)


train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, _mean, _std = load_data_new(
    graph_signal_matrix_filename, num_of_hours,
    num_of_days, num_of_weeks, DEVICE, batch_size)

# adj_mx, distance_mx = get_adjacency_matrix(adj_filename, num_of_vertices, id_filename)
# adj_mx = load_adj(adj_filename)

print(adj_filename)
adj_mx = load_adjacent(adj_filename) # 仅对 yinchuan 使用

net = make_model(DEVICE, nb_block, in_channels, K, nb_chev_filter, num_heads, dropout, adj_mx,
                 num_for_predict, len_input, num_of_vertices, _mean, _std, kernel_size)

total_num = sum([param.nelement() for param in net.parameters()])
print('Total params num: {}'.format(total_num))
for batch_index, batch_data in enumerate(tqdm(train_loader, desc="Training")):
    encoder_inputs, labels = batch_data
    input_tensor = encoder_inputs[0:1].to(DEVICE)
    # 使用 thop.profile 计算 FLOPs 和参数数量
    total_ops, total_params = profile(net, (input_tensor,))

    print(f"模型 FLOPs: {total_ops / 1e9:.9f} GFLOPs")
    print(f"模型参数数量: {total_params / 1e6:.2f} M")
    break

def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()

def train_main(_mean, _std):
    if (start_epoch == 0) and (not os.path.exists(params_path)):
        os.makedirs(params_path)
        print('create params directory %s' % (params_path))
    elif (start_epoch == 0) and (os.path.exists(params_path)):
        shutil.rmtree(params_path)
        os.makedirs(params_path)
        print('delete the old one and create params directory %s' % (params_path))
    elif (start_epoch > 0) and (os.path.exists(params_path)):
        print('train from params directory %s' % (params_path))
    else:
        raise SystemExit('Wrong type of model!')

    print('param list:')
    print('CUDA\t', DEVICE)
    print('in_channels\t', in_channels)
    print('nb_block\t', nb_block)
    print('nb_chev_filter\t', nb_chev_filter)
    print('kernel_size\t', kernel_size)
    print('dropout\t', dropout)
    print('batch_size\t', batch_size)
    print('graph_signal_matrix_filename\t', graph_signal_matrix_filename)
    print('start_epoch\t', start_epoch)
    print('epochs\t', epochs)

    # criterion = nn.MSELoss().to(DEVICE)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    sw = SummaryWriter(params_path)
    print(net)

    print('Net\'s state_dict:')
    total_param = 0
    for param_tensor in net.state_dict():
        print(param_tensor, '\t', net.state_dict()[param_tensor].size())
        total_param += np.prod(net.state_dict()[param_tensor].size())
    print('Net\'s total params:', total_param)

    print('Optimizer\'s state_dict:')
    for var_name in optimizer.state_dict():
        print(var_name, '\t', optimizer.state_dict()[var_name])
    print(_mean, _std)
    global_step = 0
    best_epoch = 0
    best_val_loss = np.inf

    start_time = time()

    if start_epoch > 0:

        params_filename = os.path.join(params_path, 'epoch_%s.params' % start_epoch)

        net.load_state_dict(torch.load(params_filename))

        print('start epoch:', start_epoch)

        print('load weight from: ', params_filename)

    # train model
    start_time = datetime.datetime.now()
    iteration = 0
    for epoch in range(start_epoch, epochs):

        params_filename = os.path.join(params_path, 'NINGDE')

        val_loss = compute_val_loss_mstgcn(net, test_loader, sw, epoch)

        if val_loss < best_val_loss:
            best_epoch = epoch
            torch.save(net.state_dict(), params_filename)
            print('Val loss decrease from {:.3f} to {:.3f}, saving to {}'.format(best_val_loss, val_loss, params_filename))
            # print('save parameters to file: %s' % params_filename)
            best_val_loss = val_loss

        net.train()  # ensure dropout layers are in train mode

        for batch_index, batch_data in enumerate(tqdm(train_loader, desc="Training")):
            iteration+=1
            encoder_inputs, labels = batch_data

            optimizer.zero_grad()
            outputs = net(encoder_inputs)

            loss = masked_mae_loss(outputs, labels)

            loss.backward()

            optimizer.step()

            training_loss = loss.item()

            global_step += 1

            sw.add_scalar('training_loss', training_loss, global_step)
            if iteration == 100:
                end_time = datetime.datetime.now()
                total_time = end_time - start_time
                print("Total running times is : %f" % total_time.total_seconds())

            if global_step % 100 == 0:
                print('global step: %s, training loss: %.3f' % (global_step, training_loss))
        # scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch + 1}, Learning rate: {current_lr}")
    print('best epoch:', best_epoch)

    # apply the best model on the test set
    predict_main(best_epoch, test_loader, test_target_tensor, _mean, _std, 'test')


def predict_main(global_step, data_loader, data_target_tensor, _mean, _std, type):
    '''

    :param global_step: int
    :param data_loader: torch.utils.data.utils.DataLoader
    :param data_target_tensor: tensor
    :param mean: (1, 1, 3, 1)
    :param std: (1, 1, 3, 1)
    :param type: string
    :return:
    '''

    params_filename = os.path.join(params_path, 'NINGDE')
    print('load weight from:', params_filename)

    net.load_state_dict(torch.load(params_filename, map_location = DEVICE))

    predict_and_save_results_mstgcn(net, data_loader, data_target_tensor, global_step, _mean, _std, params_path, type)


if __name__ == "__main__":
    train_main(_mean, _std)
    # predict_main(96, test_loader, test_target_tensor, _mean, _std, 'test')