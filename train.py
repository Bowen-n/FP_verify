# @Time: 2022.3.29 16:51
# @Author: Bolun Wu

import json
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter, LinearLocator
from sklearn.metrics import r2_score
from torch import nn

from configs import config
from fft import SelectPeakIndex, my_fft
from model import NetWork, init_weights
from utils import func_dict, save_fig


# get ground truth
def get_y(xs):
    # validate input shape
    if config.function == 1: assert xs.shape[1] == 2
    else: assert xs.shape[1] == 1
    return func_dict[config.function](xs)


class Trainer():
    def __init__(self, model, criterion, optimizer):
        """Model Trainer

        Args:
            model (torch.nn.Module): model to be trained
            criterion (torch.nn.Module): loss function
            optimizer (torch.optim...): optimizer
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        
        # initial forward
        with torch.no_grad():
            # trainset
            train_pred = self.model(train_inputs.to(device))
            train_loss = self.criterion(train_pred, train_y.to(device)).cpu().item()
            train_r2 = r2_score(train_y.numpy(), train_pred.cpu().numpy())
            
            # testset
            test_pred = self.model(test_inputs.to(device))
            test_loss = self.criterion(test_pred, test_y.to(device)).cpu().item()
            test_r2 = r2_score(test_y.numpy(), test_pred.cpu().numpy())
        
        # recordings
        self.train_pred = train_pred.cpu().numpy()
        self.test_pred = test_pred.cpu().numpy()
        self.train_loss = [train_loss]
        self.test_loss = [test_loss]
        self.train_r2 = [train_r2]
        self.test_r2 = [test_r2]
        self.train_pred_all = []
        
        # create saving directory
        subdir = f'{config.function}_{datetime.now().strftime("%y%m%d%H%M%S")}'
        self.save_dir = os.path.join(config.basedir, subdir)
        self.plot_y_dir = os.path.join(self.save_dir, 'y')
        self.plot_fft_dir = os.path.join(self.save_dir, 'fft')
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.plot_y_dir, exist_ok=True)
        os.makedirs(self.plot_fft_dir, exist_ok=True)
    
    def run_onestep(self):
        """run one step
        """
        # inference
        with torch.no_grad():
            # trainset
            train_pred = self.model(train_inputs.to(device))
            train_loss = self.criterion(train_pred, train_y.to(device)).cpu().item()
            train_r2 = r2_score(train_y.numpy(), train_pred.cpu().numpy())
            
            # testset
            test_pred = self.model(test_inputs.to(device))
            test_loss = self.criterion(test_pred, test_y.to(device)).cpu().item()
            test_r2 = r2_score(test_y.numpy(), test_pred.cpu().numpy())
            
        # record
        self.train_pred = train_pred.cpu().numpy()
        self.test_pred = test_pred.cpu().numpy()
        self.train_loss.append(train_loss)
        self.test_loss.append(test_loss)
        self.train_r2.append(train_r2)
        self.test_r2.append(test_r2)
        
        # train
        for i in range(config.train_size // config.batch_size + 1): # bootstrap training
            # choose 100 numbers between [0, 100]
            mask = np.random.choice(config.train_size, config.batch_size, replace=False)
            train_pred = self.model(train_inputs[mask].to(device))
            loss = self.criterion(train_pred, train_y[mask].to(device))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
    def run(self, n_epochs):
        """run the whole training procudure

        Args:
            n_epochs (int): total training epochs
        """
        for epoch in range(n_epochs):
            self.run_onestep() # run one step
            self.train_pred_all.append(self.train_pred) # save train predictions

            if epoch % config.plot_epoch == 0:
                # verbose
                print('time elapse: {:.2f}s, epoch: {}, loss: {:.6f}'.format(
                    time.time()-since, epoch, self.train_loss[-1]))
                # plot
                self.plot_y(name=str(epoch))
                self.plot_fft(name=str(epoch))
            
            # simple and lax early stopping
            if self.train_loss[-1] < 1e-5:
                break
            
        # after training: plot loss, r2 & save the model and config
        self.plot_loss()
        self.plot_r2()
        self.save_file()
        self.save_model()
        
    def plot_y(self, name):
        """plot the function learned vs. the ground truth

        Args:
            name (str): saving figure's name
        """
        if config.input_dim == 1:
            plt.figure()
            y1 = self.test_pred
            y2 = train_y.numpy()
            plt.plot(test_inputs.numpy(), y1, 'r.', linewidth=2.5, label='test')
            plt.plot(train_inputs.numpy(), y2, 'b-', label='True')
            plt.title('g2u', fontsize=12)
            plt.legend(fontsize=12)
            save_fig(os.path.join(self.plot_y_dir, f'y_{name}.pdf'))
            plt.close()
        elif config.input_dim == 2:
            X = np.arange(config.x_start, config.x_end, 0.1)
            Y = np.arange(config.x_start, config.x_end, 0.1)
            X, Y = np.meshgrid(X, Y)
            xy = np.concatenate(
                (np.reshape(X, [-1, 1]), np.reshape(Y, [-1, 1])), axis=1)
            Z = np.reshape(get_y(xy), [len(X), -1])
            
            fp = plt.figure()
            ax = fp.add_subplot(projection='3d')
            surf = ax.plot_surface(X, Y, Z-np.min(Z), cmap=cm.coolwarm, linewidth=0, antialiased=False)
            ax.zaxis.set_major_locator(LinearLocator(5))
            ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
            fp.colorbar(surf, shrink=0.5, aspect=5)
            ax.scatter(train_inputs[:, 0], train_inputs[:, 1],
                       self.train_pred - np.min(self.train_pred))
            save_fig(os.path.join(self.plot_y_dir, f'y_{name}.pdf'))
            plt.close()

    def plot_fft(self, name):
        """plot the function fft learned vs. the ground truth

        Args:
            name (str): saving figure's name
        """
        # ground truth fft
        train_y_fft = my_fft(train_y.numpy()) / config.train_size
        plt.semilogy(train_y_fft+1e-5, label='real')
        
        # prediction fft
        train_pred_fft = my_fft(self.train_pred) / config.train_size
        plt.semilogy(train_pred_fft+1e-5, label='pred')
        
        plt.legend()
        plt.xlabel('freq idx')
        plt.ylabel('freq')
        save_fig(os.path.join(self.plot_fft_dir, f'fft_{name}.pdf'))
        plt.close()        

    def plot_loss(self):
        """plot training and testing loss
        """
        plt.figure()
        ax = plt.gca()
        y1 = np.asarray(self.test_loss)
        y2 = np.asarray(self.train_loss)
        plt.plot(y1, 'r-', label='test')
        plt.plot(y2, 'g-', label='train')
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.legend(fontsize=12)
        plt.title('loss', fontsize=12)
        save_fig(os.path.join(self.save_dir, 'loss.pdf'))
        plt.close()

    def plot_r2(self):
        """plot training and testing R2 score
        """
        plt.figure()
        ax = plt.gca()
        y1 = np.asarray(self.test_r2)
        y2 = np.asarray(self.train_r2)
        plt.plot(y1, 'r-', label='test')
        plt.plot(y2, 'g-', label='train')
        plt.legend(fontsize=12)
        plt.title('r2', fontsize=12)
        save_fig(os.path.join(self.save_dir, 'r2.pdf'))
        plt.close()

    def save_file(self):
        """save training spec
        """
        config.end_loss = self.train_loss[-1]
        with open(os.path.join(self.save_dir, 'output.json'), 'w') as f:
            json.dump(config, f, indent=1)            

    def save_model(self):
        """save trained model
        """
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'model.pt'))


if __name__ == '__main__':
    print(config)
    
    os.makedirs(config.basedir, exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # dataset
    ## one-dimension data
    if config.input_dim == 1:
        # trainset
        train_inputs = np.linspace(config.x_start, config.x_end, num=config.train_size, endpoint=True)
        train_inputs = np.expand_dims(train_inputs, axis=-1)
        train_y = get_y(train_inputs)
        
        # testset
        test_inputs = np.linspace(config.x_start, config.x_end, num=config.test_size, endpoint=True)
        test_inputs = np.expand_dims(test_inputs, axis=-1)
        test_y = get_y(test_inputs)
        
        # to Tensor
        train_inputs = torch.FloatTensor(train_inputs)
        test_inputs = torch.FloatTensor(test_inputs)
        train_y = torch.FloatTensor(train_y)
        test_y = torch.FloatTensor(test_y)
        
    ## multi-dimension data
    else:
        # trainset
        train_inputs = np.random.rand(
            config.train_size, config.input_dim) * (config.x_end - config.x_start) + config.x_start
        train_y = get_y(train_inputs)
        train_y = np.expand_dims(train_y, axis=-1)
        
        # testset
        test_inputs = np.random.rand(
            config.test_size, config.input_dim) * (config.x_end - config.x_start) + config.x_start
        test_y = get_y(test_inputs)
        test_y = np.expand_dims(test_y, axis=-1)
        
        # to Tensor
        train_inputs = torch.FloatTensor(train_inputs)
        test_inputs = torch.FloatTensor(test_inputs)
        train_y = torch.FloatTensor(train_y)
        test_y = torch.FloatTensor(test_y)
        
    model = NetWork()
    model.apply(init_weights)
    model = model.to(device)

    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    trainer = Trainer(model, criterion, optimizer)
    
    since = time.time()
    trainer.run(n_epochs=config.epoch)
    
    # PLOT FFT
    # ground truth fft
    train_y_fft = my_fft(train_y.numpy()) / config.train_size
    plt.semilogy(train_y_fft+1e-5, label='real')
    idx = SelectPeakIndex(train_y_fft, endpoint=False)
    plt.semilogy(idx, train_y_fft[idx]+1e-5, 'o')
    
    # prediction fft
    train_pred = trainer.train_pred
    train_pred_fft = my_fft(train_pred) / config.train_size
    plt.semilogy(train_pred_fft+1e-5, label='pred')
    plt.semilogy(idx, train_pred_fft[idx]+1e-5, 'o')
    
    plt.legend()
    plt.xlabel('freq idx')
    plt.ylabel('freq')
    save_fig(os.path.join(trainer.save_dir, 'fft_final.pdf'))
    plt.close()
    
    # PLOT HOT
    train_pred_all = trainer.train_pred_all
    idx1 = idx[:10]
    abs_err = np.zeros([len(idx1), len(train_pred_all)])
    
    train_y_fft = my_fft(train_y.numpy())
    tmp1 = train_y_fft[idx1]

    for i in range(len(train_pred_all)):
        tmp2 = my_fft(train_pred_all[i])[idx1]
        abs_err[:, i] = np.abs(tmp1-tmp2)/(tmp1+1e-5)
    
    plt.figure()
    plt.pcolor(abs_err[:3, :], cmap='RdBu', vmin=0.1, vmax=1)
    plt.colorbar()
    save_fig(os.path.join(trainer.save_dir, 'hot_3.pdf'))
    plt.close()
    
    plt.figure()
    plt.pcolor(abs_err, cmap='RdBu', vmin=0.1, vmax=1)
    plt.colorbar()
    save_fig(os.path.join(trainer.save_dir, 'hot_10.pdf'))
    plt.close()
    
    