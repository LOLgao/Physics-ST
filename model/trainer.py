import torch
import math
import os
import time
import copy
import numpy as np
from lib.logger import get_logger
from lib.metrics import All_Metrics


class Trainer(object):
    def __init__(self, model, optimizer, train_loader, val_loader, test_loader,
                 scaler, args, lr_scheduler=None):
        super(Trainer, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.args = args
        self.lr_scheduler = lr_scheduler
        self.train_per_epoch = len(train_loader)
        if val_loader is not None:
            self.val_per_epoch = len(val_loader)
        self.best_path = os.path.join(self.args.log_dir, 'best_model.pth')
        self.loss_figure_path = os.path.join(self.args.log_dir, 'loss.png')
        # log
        if os.path.isdir(args.log_dir) is False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.model, debug=args.debug)
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))

    def val_epoch(self, epoch, val_dataloader):
        self.model.eval()
        total_val_loss = 0
        total_sep_loss = np.zeros(2)
        with torch.no_grad():
            for batch_idx, (data, label, date, event) in enumerate(val_dataloader):
                outputs, results = self.model(data, date.squeeze(3), event)
                if self.args.real_value:
                    results = self.scaler.inverse_transform(results)
                    label = self.scaler.inverse_transform(label)
                loss, sep_loss = self.model.loss(outputs, results, label)
                # a whole batch of Metr_LA is filtered
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
                    total_sep_loss += sep_loss
        val_loss = total_val_loss / len(val_dataloader)
        val_loss_sep = total_sep_loss / len(val_dataloader)
        self.logger.info('**********Val Epoch {}: average Loss: {}'.format(epoch, val_loss_sep))

        return val_loss

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_sep_loss = np.zeros(2)
        for batch_idx, (data, label, date, event) in enumerate(self.train_loader):

            self.optimizer.zero_grad()
            outputs, results = self.model(data, date.squeeze(3), event)
            if self.args.real_value:
                results = self.scaler.inverse_transform(results)
                label = self.scaler.inverse_transform(label)
            loss, sep_loss = self.model.loss(outputs, results, label)
            loss.backward()

            self.optimizer.step()
            total_loss += loss.item()
            total_sep_loss += sep_loss
            # log information
            if batch_idx % self.args.log_step == 0:
                self.logger.info('Train Epoch {}: {}/{} Loss: {:.6f}'.format(
                    epoch, batch_idx, self.train_per_epoch, loss.item()))
        train_epoch_loss = total_loss / self.train_per_epoch
        total_sep_loss = total_sep_loss / self.train_per_epoch
        self.logger.info('loss weights: {}'.format(total_sep_loss))
        self.logger.info(
            '**********Train Epoch {}: averaged Loss: {:.6f}'.format(epoch, train_epoch_loss,
                                                                     ))

        return train_epoch_loss

    def train(self):
        best_model = None
        best_loss = float('inf')
        not_improved_count = 0
        train_loss_list = []
        val_loss_list = []
        start_time = time.time()
        self.logger.info("lr: {}, ms: {}, weights: {}".format(
            self.args.lr_init, self.args.ms, self.args.weights))
        for epoch in range(1, self.args.epochs + 1):
            # epoch_time = time.time()
            train_epoch_loss = self.train_epoch(epoch)
            # print(time.time()-epoch_time)
            # exit()
            if self.val_loader == None:
                val_dataloader = self.test_loader
            else:
                val_dataloader = self.val_loader
            val_epoch_loss = self.val_epoch(epoch, val_dataloader)

            # print('LR:', self.optimizer.param_groups[0]['lr'])
            train_loss_list.append(train_epoch_loss)
            val_loss_list.append(val_epoch_loss)
            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break
            # if self.val_loader == None:
            # val_epoch_loss = train_epoch_loss
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
            # early stop
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.args.early_stop_patience))
                    break
            # save the best state
            if best_state == True:
                self.logger.info('*********************************Current best model saved!')
                best_model = copy.deepcopy(self.model.state_dict())
            self.test(self.model, self.args, self.test_loader, self.scaler, self.logger)
        training_time = time.time() - start_time
        self.logger.info("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))

        # save the best model to file
        if not self.args.debug:
            torch.save(best_model, self.best_path)
            self.logger.info("Saving current best model to " + self.best_path)

        # test
        self.model.load_state_dict(best_model)
        # self.val_epoch(self.args.epochs, self.test_loader)
        self.test(self.model, self.args, self.test_loader, self.scaler, self.logger)

    def save_checkpoint(self):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.args
        }
        torch.save(state, self.best_path)
        self.logger.info("Saving current best model to " + self.best_path)

    @staticmethod
    def test(model, args, data_loader, scaler, logger, path=None):
        if path is not None:
            check_point = torch.load(path)
            state_dict = check_point['state_dict']
            args = check_point['config']
            model.load_state_dict(state_dict)
            model.to(args.device)
        model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for batch_idx, (data, label, date, event) in enumerate(data_loader):
                outputs, results = model(data, date.squeeze(3), event)
                y_true.append(label)
                y_pred.append(results)
        if args.real_value:
            y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))
            y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
        else:
            y_pred = torch.cat(y_pred, dim=0)
            y_true = torch.cat(y_true, dim=0)
        if args.save_dir is not None:
            # _energy.append(energy)
            args.log_dir = args.save_dir

        np.save('{}/model_true.npy'.format(args.log_dir), y_true.cpu().numpy())
        np.save('{}/model_pred.npy'.format(args.log_dir), y_pred.cpu().numpy())

        errors = []
        for i in range(args.horizon):
            mae, rmse, mape = All_Metrics(y_pred[:, i, :, :], y_true[:, i, :, :], args.mae_thresh, args.mape_thresh)
            errors.append([mae.cpu().numpy(), rmse.cpu().numpy(), mape.cpu().numpy()])
        mae, rmse, mape = All_Metrics(y_pred, y_true, args.mae_thresh, args.mape_thresh)
        errors.append([mae.cpu().numpy(), rmse.cpu().numpy(), mape.cpu().numpy()])
        errors = np.array(errors)
        np.save(args.log_dir + '/errors.npy', errors)
        logger.info("Average Horizon, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}%".format(
            mae, rmse, mape * 100))

