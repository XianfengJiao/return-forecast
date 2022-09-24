import os
import sys
import numpy as np
import pandas as pd
import pickle as pkl
import copy
import torch
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
sys.path.append("..")

class StockWithScoreDataset(Dataset):
    def __init__(self, input_stock_path, input_score_path, input_pred_path, save_preprocess_path, type, scaler_price=None, scaler_volume=None, topn=64):
        os.makedirs(save_preprocess_path, exist_ok=True)
        preprocessed_path = os.path.join(save_preprocess_path, type+'.pkl')
        self.type = type
        if not os.path.isfile(preprocessed_path):
        # if True:
            # -------------------------- Preprocessing DATA --------------------------
            print('#'*20,'Start Processing Data','#'*20)
            input_preprocessed = []
            miss_count = 0
            input_score = pkl.load(open(input_score_path, 'rb'))
            input_pred = pkl.load(open(input_pred_path, 'rb'))
            # 按股票对 norm-price 和 norm-volume 做 z-score
            # 对时间进行 max-min normlization
            input_stock = pd.read_pickle(input_stock_path)
            input_stock.set_index(["stock","date"], inplace=True)
            input_stock.sort_index(level=[0, 1], inplace=True)
            input_stock, self.scaler_price, self.scaler_volume = self._norm_transaction(input_stock, scaler_price, scaler_volume)
            # 将 scaler 存起来
            pkl.dump(scaler_price, open(os.path.join(save_preprocess_path, "scaler_price.pkl"), 'wb'))
            pkl.dump(scaler_volume, open(os.path.join(save_preprocess_path, "scaler_volume.pkl"), 'wb'))
            # 循环读取异常点文件，进行对齐
            all_stock_names = set([key.split('_')[-1] for key in input_score.keys()])
            all_dates = sorted(list(set([key.split('_')[0] for key in input_score.keys()])))
            for date_string in tqdm(all_dates, desc='gen preprocessed data with date'):
                mor_t = pd.date_range(start=date_string+' 09:30:04', end=date_string+' 11:30:00', freq='4s').tolist()
                aft_t = pd.date_range(start=date_string+' 13:00:04', end=date_string+' 14:57:00', freq='4s').tolist()
                complete_t = np.array(mor_t + aft_t)
                date_collect = [[] for _ in range(len(all_stock_names))]
                if not self._check_stock_data(all_stock_names, date_string, input_score):
                    miss_count += 1
                    continue
                date = datetime.strptime(date_string, '%Y%m%d').date()
                try:
                    stock_data_selected = input_stock.loc[(slice(None),date), :]
                    pred_date_selected = input_pred.loc[(pd.to_datetime(date),slice(None)), :]
                except KeyError:
                    miss_count += 1
                    continue
                for s_i, stock_name in enumerate(all_stock_names):
                    # 将异常点时刻对应的 pred 和 label 拼进去
                    date_collect[s_i].append('SZ'+stock_name)
                    date_collect[s_i].append(date)
                    stock_data_tmp = stock_data_selected.loc[('SZ'+stock_name,slice(None)),:]
                    pred_data_tmp = pred_date_selected.loc[(slice(None),'SZ'+stock_name),:]
                    # Warning: just for test
                    # pred_data_tmp = pred_date_selected.loc[(slice(None),test_map['SZ'+stock_name]),:]
                    date_collect[s_i].append(pred_data_tmp['score'].values[0])
                    date_collect[s_i].append(pred_data_tmp['label'].values[0])
                    # 对每一天取 topn
                    score_selected = np.array(input_score[date_string+'_'+stock_name])
                    score_selected = (score_selected - np.min(score_selected)) / (np.max(score_selected) - np.min(score_selected))
                    ind_topn = np.argpartition(score_selected, -topn)[-topn:]
                    score_topn = score_selected[ind_topn]
                    time_topn = complete_t[ind_topn]
                    date_collect_score = [[] for _ in range(len(ind_topn))]
                    for i, (t, s) in enumerate(zip(time_topn, score_topn)):
                        # 将异常点时刻对应的 price 和 volume 拼进去
                        date_collect_score[i].append(s) # score
                        slice_data = stock_data_tmp[stock_data_tmp['datetime'] <= t]
                        slice_data = slice_data[slice_data['datetime'] > t - pd.Timedelta(seconds=4)]
                        while len(slice_data) <= 0 and t >= complete_t[0]:
                            t -= pd.Timedelta(seconds=4)
                            slice_data = stock_data_tmp[stock_data_tmp['datetime'] <= t]
                            slice_data = slice_data[slice_data['datetime'] > t - pd.Timedelta(seconds=4)]
                        if t < complete_t[0]:
                            date_collect_score[i].append(0) # price
                            date_collect_score[i].append(0) # volume
                            date_collect_score[i].append(0) # time
                        else:
                            date_collect_score[i].append(slice_data['scared_norm_price'].values[-1]) # price
                            date_collect_score[i].append(np.sum(slice_data['scared_norm_volume'].values)) # volume
                            date_collect_score[i].append(slice_data['maxmin_scared_time'].values[-1]) # time

                    date_collect[s_i].append(date_collect_score)
                
                input_preprocessed.append(date_collect)
            pkl.dump(input_preprocessed, open(preprocessed_path, 'wb'))
        
        else:
            # -------------------------- Load Preprocessed DATA ----------------------
            print("Found preprocessed transaction. Loading that!")
            input_preprocessed = pkl.load(open(preprocessed_path, 'rb'))
            self.scaler_price = pkl.load(open(os.path.join(save_preprocess_path, "scaler_price.pkl"), 'rb'))
            self.scaler_volume = pkl.load(open(os.path.join(save_preprocess_path, "scaler_volume.pkl"), 'rb'))

        self.data = input_preprocessed # day_num x stock_num x feature_num (stock, date, score_feature, pred, label)

    def _check_stock_data(self, all_stock_names, date, input_score):
        for name in all_stock_names:
            if date+'_'+name not in input_score:
                return False
        return True

    def _norm_transaction(self, df, scaler_price_all, scaler_volume_all):
        stock_names = set([i[0] for i in df.index])
        max_min_scaler = lambda x : (x-np.min(x))/(np.max(x)-np.min(x))
        df['scared_norm_price'] = 0
        df['scared_norm_volume'] = 0
        df['maxmin_scared_time'] = df[['time']].apply(max_min_scaler)
        scaler_price_all = {} if scaler_price_all == None else scaler_price_all
        scaler_volume_all = {} if scaler_volume_all == None else scaler_volume_all

        for stock_name in tqdm(stock_names, desc='preprocess transaction by stock_code', leave=False):
            select_df = df.loc[(stock_name,slice(None)),:]
            price_feature_all = np.expand_dims(select_df['norm_price'].values, axis=1)
            volume_feature_all = np.expand_dims(select_df['norm_volume'].values, axis=1)
            if self.type == 'train':
                scaler_price = StandardScaler().fit(price_feature_all)
                scaler_volume = StandardScaler().fit(volume_feature_all)
                scaler_price_all[stock_name] = copy.deepcopy(scaler_price)
                scaler_volume_all[stock_name] = copy.deepcopy(scaler_volume)
            df.loc[(stock_name,slice(None)),['scared_norm_price']] = scaler_price_all[stock_name].transform(price_feature_all)
            df.loc[(stock_name,slice(None)),['scared_norm_volume']] = scaler_volume_all[stock_name].transform(volume_feature_all)
        
        return df, scaler_price_all, scaler_volume_all


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def collate_fn(dataset):
        stock_name = [np.array(info)[:, 0] for info in dataset]
        date = [np.array(info, dtype=object)[:, 1] for info in dataset]
        pred = [np.array(info, dtype=object)[:, 2] for info in dataset]
        label = [np.array(info, dtype=object)[:, 3] for info in dataset]
        score_feature = [np.array(info, dtype=object)[:, 4] for info in dataset]

        return stock_name, date, torch.FloatTensor(score_feature), torch.FloatTensor(pred), torch.FloatTensor(label)