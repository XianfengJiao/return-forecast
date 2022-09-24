import os
import torch
import torch.nn as nn
import torch.utils.data as data
import pandas as pd
import pickle as pkl
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from argparse import ArgumentParser
from tensorboardX import SummaryWriter
from datasets import StockWithScoreDataset
from trainers import StockWithScore_Trainer
from shutil import copyfile
from models import MLP, GRU
from utils.setting_utils import setup_seed

def get_model(model_name):
    if model_name == 'GRU':
        return GRU()
    elif model_name == 'MLP':
        return MLP()
    else:
        raise NotImplementedError('model %s is not implemented'%model_name)

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    setup_seed(args.seed)
    # ----------------- Instantiate Dataset ------------------------
    train_dataset = StockWithScoreDataset(
        input_stock_path=os.path.join(args.input_stock_path, 'train', 'stock_data_all.pkl'),
        input_score_path=os.path.join(args.input_score_path, 'train.pkl'),
        input_pred_path=os.path.join(args.input_pred_path, 'train.pkl'),
        type='train',
        save_preprocess_path=args.save_preprocess_path
        )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)

    valid_dataset = StockWithScoreDataset(
        input_stock_path=os.path.join(args.input_stock_path, 'valid', 'stock_data_all.pkl'),
        input_score_path=os.path.join(args.input_score_path, 'valid.pkl'),
        input_pred_path=os.path.join(args.input_pred_path, 'valid.pkl'),
        scaler_price = train_dataset.scaler_price,
        scaler_volume = train_dataset.scaler_volume,
        type='valid',
        save_preprocess_path=args.save_preprocess_path
        )
    
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=valid_dataset.collate_fn)

    test_dataset = StockWithScoreDataset(
        input_stock_path=os.path.join(args.input_stock_path, 'test', 'stock_data_all.pkl'),
        input_score_path=os.path.join(args.input_score_path, 'test.pkl'),
        input_pred_path=os.path.join(args.input_pred_path, 'test.pkl'),
        scaler_price = train_dataset.scaler_price,
        scaler_volume = train_dataset.scaler_volume,
        type='test',
        save_preprocess_path=args.save_preprocess_path
        )
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)

    # ----------------- Instantiate Model --------------------------
    model = get_model(args.model)
    # ----------------- Instantiate Trainer ------------------------
    trainer = StockWithScore_Trainer(
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        batch_size=args.batch_size,
        num_epochs=args.epoch,
        early_stop=args.early_stop,
        log_dir=args.log_dir,
        device=device,
        model=model,
        theta=args.theta,
        save_path=args.save_path,
    )

    print('#'*20,"Starting training...",'#'*20)
    for epoch in range(1, args.epoch + 1):
        trainer.train_epoch(epoch)
        if trainer.remain_epoch == 0:
            break
    print('#'*20,"End training",'#'*20)
    copyfile(trainer.best_model_path, os.path.join(os.path.dirname(trainer.best_model_path), 'best_model.pth'))

    print("Starting testing...")
    model = get_model(args.model)
    ckpt = torch.load(trainer.best_model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(ckpt)
    if args.theta == 0:
        loss, corr, mae, rmse = trainer.test_with_check(model, pred_path=os.path.join(args.input_pred_path, 'test.pkl'))
    else:
        loss, corr, mae, rmse = trainer.test(model)
    print("End testing")
    print(f'mse: {loss:.4f}, corr: {corr:.4f}, mae: {mae:.4f}, rmse: {rmse:.4f}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_stock_path', default='/D_data/v-xfeng/code/Return-Forecast/inputs/SZ_data', type=str)
    parser.add_argument('--input_score_path', default='/D_data/v-xfeng/code/Return-Forecast/inputs/feature/score_ano', type=str)
    parser.add_argument('--input_pred_path', default='/D_data/v-xfeng/code/Return-Forecast/inputs/feature/pred_rnn_v1_0', type=str)
    parser.add_argument("--save_preprocess_path", default='/D_data/v-xfeng/code/Return-Forecast/inputs/preprocessed/SZ_ano_pred_rnn_v1_0', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--model', default='MLP', type=str)
    parser.add_argument('--seed', default=5, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--theta', default=0, type=float)
    parser.add_argument('--epoch', default=1, type=int)
    parser.add_argument('--early_stop', default=200, type=int)
    parser.add_argument('--log_dir', default='/D_data/v-xfeng/code/Return-Forecast/logs/debug', type=str)
    parser.add_argument('--save_path', default='/D_data/v-xfeng/code/Return-Forecast/checkpoints/debug', type=str)
    args = parser.parse_args()
    main(args)