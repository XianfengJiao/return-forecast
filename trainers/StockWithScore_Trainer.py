import os
from tqdm import tqdm
import torch
import pandas as pd
import torch.nn as nn
from torch.autograd import grad as torch_grad
import numpy as np
import torch.optim as optim
from sklearn.decomposition import PCA
from datetime import datetime, date
import pandas as pd
import pickle as pkl
from sklearn.preprocessing import StandardScaler
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from utils.metric_utils import get_loss_fn, get_metric_fn, get_cosine_schedule_with_warmup

class StockWithScore_Trainer(object):
    def __init__(
        self,
        train_loader,
        valid_loader,
        test_loader,
        batch_size,
        num_epochs,
        log_dir,
        device,
        model,
        save_path,
        theta=0.5,
        lr=5e-4,
        early_stop=100,
        loss='mse',
        metric='corr',
        checkpoint=None,
    ):
        self.train_loader = train_loader
        self.save_path = save_path
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.checkpoint = checkpoint
        self.log_dir = log_dir
        self.theta = theta
        self.lr = lr
        self.best_model_path = None
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = self.configure_optimizer()
        self.loss_fn = get_loss_fn(loss)
        self.metric_fn = get_metric_fn(metric)
        self.metric_rmse = get_metric_fn('rmse')
        self.metric_mae = get_metric_fn('mae')
        total_steps = len(train_loader) * self.num_epochs
        self.scheduler = get_cosine_schedule_with_warmup(optimizer=self.optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        os.makedirs(self.log_dir, exist_ok=True)
        self.tensorwriter = SummaryWriter(log_dir)
        
        os.makedirs(self.save_path, exist_ok=True)
        self.best_loss = 1e9
        self.best_corr = -1e9
        self.best_mse = 1e9
        self.early_stop = early_stop
        self.remain_epoch = early_stop


    def train_epoch(self, epoch):
        self.model.train()
        train_iterator = tqdm(
            self.train_loader, desc="Epoch {}/{}".format(epoch, self.num_epochs), leave=False
        )
        loss_epoch = 0
        corr_epoch = 0
        for stock_name, date, score_feature, pred, label in train_iterator:
            score_feature = score_feature.to(self.device)
            pred = pred.to(self.device)
            label = label.to(self.device)

            outputs = self.model(score_feature)
            outputs = self.theta * outputs + (1 - self.theta) * pred
            loss = self.loss_fn(pred=outputs, label=label)
            loss = torch.mean(loss, dim=0, keepdim=False)
            corr = self.metric_fn(x=outputs, y=label).mean()
            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_epoch += loss.item()
            corr_epoch += corr.item()

        loss_epoch /= len(self.train_loader)
        corr_epoch /= len(self.train_loader)
        # Print epoch stats
        print(f"Epoch {epoch}:")
        print(f"Train Loss: {loss_epoch:.4f} corr: {corr_epoch:.4f}")
        self.tensorwriter.add_scalar("train_loss/epoch", loss_epoch, epoch)
        self.tensorwriter.add_scalar("train_corr/epoch", corr_epoch, epoch)


        mse, corr, mae, rmse = self.evaluate(epoch)
        if mse < self.best_mse:
            self.best_mse = mse
            self.remain_epoch = self.early_stop
            self.best_model_path = os.path.join(self.save_path, 'epoch_'+str(epoch)+'-best_eval_mse'+str(mse.item())+'.pth')
            torch.save(self.model.state_dict(), self.best_model_path)
            print(f"Epoch {epoch} | Eval best mse: {mse:.4f}")
        else:
            self.remain_epoch -= 1
        
        # 仅用于观察 test 集上 loss 曲线的变化
        self.test(self.model, epoch)


    __call__ = train_epoch

    def evaluate(self, epoch):
        self.model.eval()
        valid_iterator = tqdm(
            self.valid_loader, desc="Evaluation", total=len(self.valid_loader)
        )
        all_outputs = []
        all_labels = []
        with torch.no_grad():
            for stock_name, date, score_feature, pred, label in valid_iterator:
                score_feature = score_feature.to(self.device)
                pred = pred.to(self.device)
                label = label.to(self.device)
                outputs = self.model(score_feature)
                outputs = self.theta * outputs + (1 - self.theta) * pred

                all_outputs.append(outputs)
                all_labels.append(label)
        
        all_outputs = torch.cat(all_outputs, dim=0).squeeze()
        all_labels = torch.cat(all_labels, dim=0).squeeze()
        loss = self.loss_fn(pred=all_outputs, label=all_labels).mean()
        corr = self.metric_fn(x=all_outputs, y=all_labels).mean()
        mae = self.metric_mae(pred=all_outputs, label=all_labels).mean()
        rmse = self.metric_rmse(pred=all_outputs, label=all_labels).mean()
        self.tensorwriter.add_scalar("eval_mse/epoch", loss.item(), epoch)
        self.tensorwriter.add_scalar("eval_corr/epoch", corr.item(), epoch)
        self.tensorwriter.add_scalar("eval_mae/epoch", mae.item(), epoch)
        self.tensorwriter.add_scalar("eval_rmse/epoch", rmse.item(), epoch)
        print(f'Eval: mse: {loss:.4f}, corr: {corr:.4f}, mae: {mae:.4f}, rmse: {rmse:.4f}')
        return loss, corr, mae, rmse


    def test_with_check(self, model, pred_path):
        EPS = 1e-6
        df_pred = pd.read_pickle(pred_path)
        model.to(self.device)
        model.eval()
        test_iterator = tqdm(
            self.test_loader, desc="Test", total=len(self.test_loader)
        )
        all_outputs = []
        all_labels = []
        all_dates = []
        with torch.no_grad():
            for stock_name, date, score_feature, pred, label in test_iterator:
                score_feature = score_feature.to(self.device)
                pred = pred.to(self.device)
                label = label.to(self.device)
                outputs = model(score_feature)
                outputs = self.theta * outputs + (1 - self.theta) * pred

                all_outputs.append(outputs)
                all_labels.append(label)
                all_dates += [d[0] for d in date]
        
        all_outputs = torch.cat(all_outputs, dim=0).squeeze()
        all_labels = torch.cat(all_labels, dim=0).squeeze()
        loss = self.loss_fn(pred=all_outputs, label=all_labels).mean()
        corr = self.metric_fn(x=all_outputs, y=all_labels).mean()
        mae = self.metric_mae(pred=all_outputs, label=all_labels).mean()
        rmse = self.metric_rmse(pred=all_outputs, label=all_labels).mean()

        for d, o, l in zip(all_dates, all_outputs, all_labels):
            score_selected = df_pred.loc[(pd.to_datetime(d),slice(None)),['score']].values.squeeze()
            label_selected = df_pred.loc[(pd.to_datetime(d),slice(None)),['label']].values.squeeze()
            assert(score_selected.sum() - o.sum().cpu().item() < EPS)
            assert(label_selected.sum() - l.sum().cpu().item() < EPS)

        return loss, corr, mae, rmse

    def test(self, model, epoch=-1):
        model.to(self.device)
        model.eval()
        test_iterator = tqdm(
            self.test_loader, desc="Test", total=len(self.test_loader)
        )
        all_outputs = []
        all_labels = []
        all_dates = []
        with torch.no_grad():
            for stock_name, date, score_feature, pred, label in test_iterator:
                score_feature = score_feature.to(self.device)
                pred = pred.to(self.device)
                label = label.to(self.device)
                outputs = model(score_feature)
                outputs = self.theta * outputs + (1 - self.theta) * pred

                all_outputs.append(outputs)
                all_labels.append(label)
                all_dates.append(date)
        
        all_outputs = torch.cat(all_outputs, dim=0).squeeze()
        all_labels = torch.cat(all_labels, dim=0).squeeze()
        loss = self.loss_fn(pred=all_outputs, label=all_labels).mean()
        corr = self.metric_fn(x=all_outputs, y=all_labels).mean()
        mae = self.metric_mae(pred=all_outputs, label=all_labels).mean()
        rmse = self.metric_rmse(pred=all_outputs, label=all_labels).mean()
        if epoch > 0:
            self.tensorwriter.add_scalar("test_mse/epoch", loss.item(), epoch)
            self.tensorwriter.add_scalar("test_corr/epoch", corr.item(), epoch)
            self.tensorwriter.add_scalar("test_mae/epoch", mae.item(), epoch)
            self.tensorwriter.add_scalar("test_rmse/epoch", rmse.item(), epoch)
            print(f'Test: mse: {loss:.4f}, corr: {corr:.4f}, mae: {mae:.4f}, rmse: {rmse:.4f}')
        return loss, corr, mae, rmse

    def configure_optimizer(self):
        return optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.1)