from matplotlib import pyplot as plt
from torch.optim import lr_scheduler

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, save_to_csv, visual_weights
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.data == 'PEMS':
            criterion = nn.L1Loss()
        else:
            criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                batch_x, batch_x_mark, batch_y, batch_y_mark = self.__multi_scale_process_inputs(batch_x, batch_x_mark,
                                                                                                 batch_y,
                                                                                                 batch_y_mark)
                if 'PEMS' == self.args.data or 'Solar' == self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None

                if self.args.down_sampling_layers == 0:
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                else:
                    dec_inp = None

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y, outputs = self.__process_outputs(batch_y, f_dim, outputs)

                if isinstance(batch_y, list) and isinstance(outputs, list):
                    pred = outputs[0].detach()
                    true = batch_y[0].detach()
                else:
                    pred = outputs.detach()
                    true = batch_y.detach()

                if self.args.data == 'PEMS':
                    B, T, C = pred.shape
                    pred = pred.cpu().numpy()
                    true = true.cpu().numpy()
                    pred = vali_data.inverse_transform(pred.reshape(-1, C)).reshape(B, T, C)
                    true = vali_data.inverse_transform(true.reshape(-1, C)).reshape(B, T, C)
                    mae, mse, rmse, mape, mspe = metric(pred, true)
                    total_loss.append(mae)

                else:
                    loss = criterion(pred, true)
                    total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                batch_x, batch_x_mark, batch_y, batch_y_mark = self.__multi_scale_process_inputs(batch_x, batch_x_mark,
                                                                                                 batch_y,
                                                                                                 batch_y_mark)

                if 'PEMS' == self.args.data or 'Solar' == self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None

                if self.args.down_sampling_layers == 0:
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                else:
                    dec_inp = None

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    batch_y, outputs = self.__process_outputs(batch_y, f_dim, outputs)

                    loss = self.do_criterion(batch_y, criterion, outputs)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(test_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=True)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def do_criterion(self, batch_y, criterion, outputs):
        if isinstance(batch_y, list) and isinstance(outputs, list):
            loss_list = []
            for batch_y, output in zip(batch_y, outputs):
                loss = criterion(output, batch_y)
                loss_list.append(loss)
            loss = sum(loss_list) / len(loss_list)
        else:
            loss = criterion(outputs, batch_y)
        return loss

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        checkpoints_path = './checkpoints/' + setting + '/'
        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                batch_x, batch_x_mark, batch_y, batch_y_mark = self.__multi_scale_process_inputs(batch_x, batch_x_mark,
                                                                                                 batch_y,
                                                                                                 batch_y_mark)
                if 'PEMS' == self.args.data or 'Solar' == self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None

                if self.args.down_sampling_layers == 0:
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                else:
                    dec_inp = None

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y, outputs = self.__process_outputs(batch_y, f_dim, outputs)
                if isinstance(batch_y, list) and isinstance(outputs, list):
                    outputs = outputs[0].detach().cpu().numpy()
                    batch_y = batch_y[0].detach().cpu().numpy()
                else:
                    outputs = outputs.detach().cpu().numpy()
                    batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    if self.args.down_sampling_method and self.args.only_use_down_sampling == False and self.args.pred_down_sampling:
                        input = batch_x[-1].detach().cpu().numpy()
                        gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                        pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                        visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
                        save_to_csv(gt, pd, os.path.join(folder_path, str(i) + '.csv'))
                    elif self.args.down_sampling_method and self.args.only_use_down_sampling == False and self.args.pred_down_sampling == False:
                        input = batch_x[0].detach().cpu().numpy()
                        gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                        pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                        visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
                        save_to_csv(gt, pd, os.path.join(folder_path, str(i) + '.csv'))
                    else:
                        input = batch_x.detach().cpu().numpy()
                        gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                        pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                        visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
                        save_to_csv(gt, pd, os.path.join(folder_path, str(i) + '.csv'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        if self.args.data == 'PEMS':
            B, T, C = preds.shape
            preds = test_data.inverse_transform(preds.reshape(-1, C)).reshape(B, T, C)
            trues = test_data.inverse_transform(trues.reshape(-1, C)).reshape(B, T, C)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        print('rmse:{}, mape:{}, mspe:{}'.format(rmse, mape, mspe))

        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        if self.args.data == 'PEMS':
            f.write('mae:{}, mape:{}, rmse:{}'.format(mae, mape, rmse))
        else:
            f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        return

    def __multi_scale_process_inputs(self, batch_x, batch_x_mark, batch_y, batch_y_mark):
        if self.args.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.args.down_sampling_window, return_indices=False)

        elif self.args.down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(self.args.down_sampling_window)
        else:
            return batch_x, batch_x_mark, batch_y, batch_y_mark
        # B,T,C -> B,C,T
        batch_x = batch_x.permute(0, 2, 1)
        batch_y = batch_y.permute(0, 2, 1)

        batch_x_ori = batch_x
        batch_y_ori = batch_y
        batch_x_mark_ori = batch_x_mark
        batch_y_mark_ori = batch_y_mark

        batch_x_sampling_list = []
        batch_y_sampling_list = []
        batch_x_mark_list = []
        batch_y_mark_list = []
        batch_x_sampling_list.append(batch_x.permute(0, 2, 1))
        batch_y_sampling_list.append(batch_y.permute(0, 2, 1))
        batch_x_mark_list.append(batch_x_mark)
        batch_y_mark_list.append(batch_y_mark)

        for i in range(self.args.down_sampling_layers):
            batch_x_sampling = down_pool(batch_x_ori)
            batch_y_sampling = batch_y_ori

            batch_x_sampling_list.append(batch_x_sampling.permute(0, 2, 1))
            batch_y_sampling_list.append(batch_y_sampling.permute(0, 2, 1))

            batch_x_mark_list.append(batch_x_mark_ori[:, ::self.args.down_sampling_window, :])
            batch_y_mark_list.append(batch_y_mark_ori)

            batch_x_ori = batch_x_sampling
            batch_y_ori = batch_y_sampling

            batch_x_mark_ori = batch_x_mark_ori[:, ::self.args.down_sampling_window, :]
            batch_y_mark_ori = batch_y_mark_ori

        if self.args.only_use_down_sampling and self.args.down_sampling_layers == 1:
            return batch_x_sampling.permute(0, 2, 1), batch_x_mark[:, ::self.args.down_sampling_window, :], \
                batch_y_sampling.permute(0, 2, 1), batch_y_mark[:, ::self.args.down_sampling_window, :]
        # B,C,T -> B,T,C
        if self.args.down_sampling_layers == 1 and self.args.pred_down_sampling:
            batch_x = [batch_x.permute(0, 2, 1), batch_x_sampling.permute(0, 2, 1)]
            batch_y = batch_y_sampling.permute(0, 2, 1)
            batch_x_mark = [batch_x_mark, batch_x_mark[:, ::self.args.down_sampling_window, :]]
            batch_y_mark = [batch_y_mark, batch_y_mark[:, ::self.args.down_sampling_window, :]]
        else:
            batch_x = batch_x_sampling_list
            batch_y = batch_y.permute(0, 2, 1)
            batch_x_mark = batch_x_mark_list
            batch_y_mark = batch_y_mark

        return batch_x, batch_x_mark, batch_y, batch_y_mark

    def __process_outputs(self, batch_y, f_dim, outputs):
        if self.args.down_sampling_method and self.args.pred_down_sampling and self.args.down_sampling_layers == 1:
            batch_y, outputs = self.__do_process_outputs(batch_y, f_dim, outputs)
        else:
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
        return batch_y, outputs

    def __do_process_outputs(self, batch_y, f_dim, outputs):
        outputs = outputs[:, -(self.args.pred_len) // self.args.down_sampling_window:, f_dim:]
        batch_y = batch_y[:, -(self.args.pred_len) // self.args.down_sampling_window:, f_dim:].to(
            self.device)
        return batch_y, outputs
