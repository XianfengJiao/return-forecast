import os
import pickle as pkl
import pandas as pd
from argparse import ArgumentParser
from glob import glob
from datetime import datetime
from tqdm import tqdm

def trans_time(date, time):
    try:
        h = int(time / 10000000)
        m = int(str(time)[len(str(h)): len(str(h))+2])
        s = int(str(time)[len(str(h)) + 2: len(str(h))+4])

        y = int(date[:4])
        mon = int(date[4:6])
        d = int(date[6:8])
        return datetime(y, mon, d, h, m, s)
    except Exception as e:
        print(e)
        print(y, mon, d, h, m, s)

def main(args):
    df = pd.DataFrame()
    for fn in tqdm(glob(os.path.join(args.data_path, args.sub_dir, '*.csv')), desc='process data for '+args.sub_dir):
        stock_date_data = pd.read_csv(fn)
        date_str = os.path.basename(fn).split('_')[0]
        stock_code = os.path.basename(fn).split('_')[1].replace('.csv', '')
        stock_date_data['datetime'] = stock_date_data.apply( lambda x: trans_time( date_str, x['time']), axis=1 )
        stock_date_data['date'] = stock_date_data.apply( lambda x: x['datetime'].date(), axis=1)
        stock_date_data['stock'] = 'SZ' + stock_code
        df = pd.concat([df, stock_date_data], axis=0, ignore_index=True)
    df.to_pickle(os.path.join(args.data_path, args.sub_dir, 'stock_data_all.pkl'))
    



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', default='/D_data/v-xfeng/code/Return-Forecast/inputs/SZ_data', type=str)
    parser.add_argument('--sub_dir', default='test', type=str)
    args = parser.parse_args()
    main(args)