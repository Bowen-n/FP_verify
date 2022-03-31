# @Time: 2022.3.29 16:39
# @Author: Bolun Wu

from addict import Dict


config = {
    # model
    'times': 0.5,
    'input_dim': 1,
    'output_dim': 1,
    'hidden_units': [200, 200, 200, 100],
    # dataset
    'train_size': 2000,
    'test_size': 1000,
    'x_start': -5,
    'x_end': 5,
    # train
    'epoch': 15000,
    'learning_rate': 1e-2,
    'plot_epoch': 250,
    # save
    'basedir': '.',
    # auxiliray
    'function': 0,
}

config = Dict(config)
config.batch_size = config.train_size
config.astddev = 1 / (config.hidden_units[0] ** config.times)
config.bstddev = 1 / (config.hidden_units[0] ** config.times)
config.full_net = [config.input_dim] + config.hidden_units + [config.output_dim]
