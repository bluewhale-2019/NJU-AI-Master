from src.models.baselines import ZeroForecast, MeanForecast
from src.models.TsfKNN import TsfKNN
from src.models.ARIMA import ARIMA
from src.models.ThetaMethod import ThetaMethod
from src.models.ResidualModel import ResidualModel
from src.utils.transforms import IdentityTransform, Standardization
from src.utils.decomposition import STL_decomposition, X11_decomposition, moving_average, differential_decomposition
from trainer import MLTrainer
from src.dataset.dataset import get_dataset
import argparse
import random
import numpy as np

import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'


def get_args():
    parser = argparse.ArgumentParser()

    # dataset config
    parser.add_argument('--data_path', type=str, default='./src/dataset/ETT-small/ETTh1.csv')
    parser.add_argument('--train_data_path', type=str, default='./dataset/m4/Daily-train.csv')
    parser.add_argument('--test_data_path', type=str, default='./dataset/m4/Daily-test.csv')
    parser.add_argument('--dataset', type=str, default='ETT', help='dataset type, options: [M4, ETT, Custom]')
    parser.add_argument('--target', type=str, default='OT', help='target feature')
    parser.add_argument('--frequency', type=str, default='h', help='frequency of time series data, options: [h, m]')

    parser.add_argument('--ratio_train', type=int, default=0.7, help='train dataset length')
    parser.add_argument('--ratio_val', type=int, default=0, help='validate dataset length')
    parser.add_argument('--ratio_test', type=int, default=0.3, help='input sequence length')

    # forcast task config
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length in [96, 192, 336, 720]')

    # model define
    parser.add_argument('--model', type=str, required=False, default='TsfKNN', help='model name')
    parser.add_argument('--n_neighbors', type=int, default=1, help='number of neighbors used in TsfKNN')
    parser.add_argument('--distance', type=str, default='euclidean', help='distance used in TsfKNN')
    parser.add_argument('--msas', type=str, default='MIMO', help='multi-step ahead strategy used in TsfKNN, options: '
                                                                 '[MIMO, recursive]')

    # transform define
    parser.add_argument('--transform', type=str, default='Standardization', help='transform name')

    # DLinear
    parser.add_argument('--individual', action='store_true', default=False,
                        help='DLinear: a linear layer for each variate(channel) individually')
    parser.add_argument('--decomposition', type=str, default='STL_decomposition', help='decomposition method')
    parser.add_argument('--enc_in', type=int, default=7, help='number of channels in encoder convolution(ETT-small)')
    # parser.add_argument('--embedding', type=str, default='fourier_embedding', help='embedding method used in DLinear')

    # ARIMA
    parser.add_argument('--seasonal_period', type=int, default=24, help='seasonal time step in ARIMA')

    # ThetaMethod
    parser.add_argument('--theta', type=float, default=1.5, help='theta in ThetaMethod')

    args = parser.parse_args()
    return args


def get_model(args):
    model_dict = {
        'TsfKNN': TsfKNN,
        'ThetaMethod': ThetaMethod,
        'ResidualModel': ResidualModel,
        'ARIMA': ARIMA
    }
    return model_dict[args.model](args)


def get_transform(args):
    transform_dict = {
        # 'IdentityTransform': IdentityTransform,
        'Standardization': Standardization
    }
    return transform_dict[args.transform](args)

def get_decomposition(args):
    decomposition_dict = {
        'moving_average': moving_average,
        'differential_decomposition': differential_decomposition,
        'STL_decomposition': STL_decomposition,
        'X11_decomposition': X11_decomposition
    }
    return decomposition_dict[args.decomposition]


if __name__ == '__main__':
    fix_seed = 2023
    random.seed(fix_seed)
    np.random.seed(fix_seed)

    args = get_args()
    if str(args.model) == 'TsfKNN':
        title = str(args.data_path) + " " + str(args.model) + " " + str(args.decomposition)
        print(title)
    else:
        title = str(args.data_path) + " " + str(args.model)
        print(title)
    # load dataset
    dataset = get_dataset(args)
    # create model
    model = get_model(args)
    # data transform
    transform = get_transform(args)
    # create trainer
    trainer = MLTrainer(model=model, transform=transform, dataset=dataset)
    # train model
    trainer.train()
    # evaluate model
    trainer.evaluate(dataset, seq_len=args.seq_len, pred_len=args.pred_len)
