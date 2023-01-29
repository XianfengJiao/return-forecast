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
from datasets import StockWithMultiScoreDataset
from trainers import StockWithScore_Trainer
from shutil import copyfile
import csv
from models import MLP, GRU, MLP_resnet
from utils.setting_utils import setup_seed
from utils.schedule_utils import load_model

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    setup_seed(args.seed)
    # ----------------- Instantiate Dataset ------------------------
    train_dataset = StockWithMultiScoreDataset(
        input_stock_path=os.path.join(args.input_stock_path, 'train', 'stock_data_all.pkl'),
        input_score_path=os.path.join(args.input_score_path, 'train.pkl'),
        input_pred_path=os.path.join(args.input_pred_path, 'train.pkl'),
        score_select=args.score_select,
        type='train',
        save_preprocess_path=args.save_preprocess_path
        )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)

    valid_dataset = StockWithMultiScoreDataset(
        input_stock_path=os.path.join(args.input_stock_path, 'valid', 'stock_data_all.pkl'),
        input_score_path=os.path.join(args.input_score_path, 'valid.pkl'),
        input_pred_path=os.path.join(args.input_pred_path, 'valid.pkl'),
        scaler_price = train_dataset.scaler_price,
        scaler_volume = train_dataset.scaler_volume,
        score_select=args.score_select,
        type='valid',
        save_preprocess_path=args.save_preprocess_path
        )
    
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=valid_dataset.collate_fn)

    test_dataset = StockWithMultiScoreDataset(
        input_stock_path=os.path.join(args.input_stock_path, 'test', 'stock_data_all.pkl'),
        input_score_path=os.path.join(args.input_score_path, 'test.pkl'),
        input_pred_path=os.path.join(args.input_pred_path, 'test.pkl'),
        scaler_price = train_dataset.scaler_price,
        scaler_volume = train_dataset.scaler_volume,
        score_select=args.score_select,
        type='test',
        save_preprocess_path=args.save_preprocess_path
        )
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)

    # ----------------- Instantiate Model --------------------------
    model = load_model(args.model, args)
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

    print('#'*20,"Start training...",'#'*20)
    for epoch in range(1, args.epoch + 1):
        trainer.train_epoch(epoch)
        if trainer.remain_epoch == 0:
            break
    print('#'*20,"End training",'#'*20)
    copyfile(trainer.best_model_path, os.path.join(os.path.dirname(trainer.best_model_path), 'best_model.pth'))

    print("Start testing...")
    model = load_model(args.model, args)
    ckpt = torch.load(trainer.best_model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(ckpt)
    if args.theta == 0:
        mse, corr, mae, rmse = trainer.test_with_check(model, pred_path=os.path.join(args.input_pred_path, 'test.pkl'))
    else:
        mse, corr, mae, rmse = trainer.test(model)
    print("End testing")
    print(f'mse: {mse:.4f}, corr: {corr:.4f}, mae: {mae:.4f}, rmse: {rmse:.4f}')

    print("Start saving metric res...")
    os.makedirs(os.path.dirname(args.metric_csv_path), exist_ok=True)
    if not os.path.exists(args.metric_csv_path):
        with open(args.metric_csv_path,'w') as f:
            csv_write = csv.writer(f)
            csv_head = ["Model","theta","seed","mse","mae","rmse","corr"]
            csv_write.writerow(csv_head)
    with open(args.metric_csv_path,'a+') as f:
        csv_write = csv.writer(f)
        data_row = [args.model,args.theta,args.seed,mse.item(),mae.item(),rmse.item(),corr.item()]
        csv_write.writerow(data_row)
    print("End saving metric res...")




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_stock_path', default='/home/v-xiajiao/code/return_forecast/inputs/SZ_data/10stock_1year', type=str)
    parser.add_argument('--input_score_path', default='/home/v-xiajiao/code/return_forecast/inputs/feature/10stock_1year/score_dyna', type=str)
    parser.add_argument('--input_pred_path', default='/home/v-xiajiao/code/return_forecast/inputs/feature/10stock_1year/pred_rnn_v1_0', type=str)
    parser.add_argument("--save_preprocess_path", default='/home/v-xiajiao/code/return_forecast/inputs/preprocessed/10stock_1year/SZ_dyna_pred_rnn_v1_0', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--model', default='MLP_resnet', type=str)
    parser.add_argument('--score_select', default='m', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--n_input', default=284, type=int)
    parser.add_argument('--n_feature', default=145, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--theta', default=0., type=float)
    parser.add_argument('--epoch', default=2, type=int)
    parser.add_argument('--early_stop', default=200, type=int)
    parser.add_argument('--log_dir', default='/home/v-xiajiao/code/return_forecast/logs/debug', type=str)
    parser.add_argument('--save_path', default='/home/v-xiajiao/code/return_forecast/checkpoints/debug', type=str)
    parser.add_argument('--metric_csv_path', default='/home/v-xiajiao/code/return_forecast/results/debug.csv', type=str)
    args = parser.parse_args()
    
    if args.theta == 0:
        args.epoch = 2

    main(args)