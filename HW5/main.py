import argparse
import random

import numpy as np
import torch
from src.dataset.dataset import get_dataset
from src.models.PatchTST import PatchTST
from src.models.Transformer import Transformer
from src.models.TsfKNN import TsfKNN
from src.models.baselines import ZeroForecast, MeanForecast
from src.utils.transforms import IdentityTransform, StandardizationTransform
from trainer import MLTrainer

from src.utils.decomposition import STL_decomposition, X11_decomposition, moving_average, differential_decomposition
from src.models.DLinear import DLinear
from src.models.SPIRIT import SPIRIT
from src.models.ResidualModel import ResidualModel
from src.models.ThetaMethod import ThetaMethod
from src.models.ARIMA import ARIMA

def get_args():
    parser = argparse.ArgumentParser()

    # dataset config
    parser.add_argument('--data_path', type=str, default='./src/dataset/ETT-small/ETTh1.csv')
    parser.add_argument('--train_data_path', type=str, default='./dataset/m4/Daily-train.csv')
    parser.add_argument('--test_data_path', type=str, default='./dataset/m4/Daily-test.csv')
    parser.add_argument('--dataset', type=str, default='ETT', help='dataset type, options: [M4, ETT, Custom]')
    parser.add_argument('--target', type=str, default='OT', help='target feature')
    parser.add_argument('--frequency', type=str, default='h', help='frequency of time series data, options: [h, m]')
    parser.add_argument('--ratio_train', type=float, default=0.7, help='ratio of training set')
    parser.add_argument('--ratio_val', type=float, default=0.1, help='ratio of validation set')
    parser.add_argument('--ratio_test', type=float, default=0.2, help='ratio of test set')

    # forcast task config
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # model define
    parser.add_argument('--model', type=str, default='PatchTST', help='model name')
    parser.add_argument('--n_neighbors', type=int, default=1, help='number of neighbors used in TsfKNN')
    parser.add_argument('--distance', type=str, default='euclidean', help='distance used in TsfKNN')
    parser.add_argument('--msas', type=str, default='MIMO', help='multi-step ahead strategy used in TsfKNN, options: '
                                                                 '[MIMO, recursive]')

    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, \
                        b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='hidden size')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--n_heads', type=int, default=8, help='number of heads')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate')
    parser.add_argument('--activation', type=str, default='gelu', help='activation function')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--output_attention', type=bool, default=False, help='output attention')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--patch_len', type=int, default=16, help='patch_len')

    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='epochs')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')

    # gpu define
    parser.add_argument('--device', type=str, default='1', help='gpu id or cpu')

    # transform define
    parser.add_argument('--transform', type=str, default='Standardization')

    # DLinear
    parser.add_argument('--individual', action='store_true', default=False,
                        help='DLinear: a linear layer for each variate(channel) individually')
    parser.add_argument('--decomposition', type=str, default='None', help='decomposition method')
    # parser.add_argument('--enc_in', type=int, default=7, help='number of channels in encoder convolution(ETT-small)')
    # parser.add_argument('--embedding', type=str, default='fourier_embedding', help='embedding method used in DLinear')

    # ARIMA
    parser.add_argument('--seasonal_period', type=int, default=24, help='seasonal time step in ARIMA')

    # ThetaMethod
    parser.add_argument('--theta', type=float, default=1.5, help='theta in ThetaMethod')

    # SPIRIT
    parser.add_argument('--n_principal', type=int, default=7, help='number of principal components')

    parser.add_argument('--pattern', type=str, default='global')
    
    # Ditillation
    parser.add_argument('--alpha', type=float, default=0.1, help='interpolation')

    args = parser.parse_args()
    return args


def get_model(args):
    model_dict = {
        'ZeroForecast': ZeroForecast,
        'MeanForecast': MeanForecast,
        'TsfKNN': TsfKNN,
        'PatchTST': PatchTST,
        'Distillation': PatchTST,
        'Transformer': Transformer,
        'ARIMA': ARIMA,
        'DLinear': DLinear,
        'SPIRIT': SPIRIT,
        'ResidualModel': ResidualModel,
        'ThetaMethod': ThetaMethod
    }
    return model_dict[args.model](args)


def get_transform(args):
    transform_dict = {
        'IdentityTransform': IdentityTransform,
        'Standardization': StandardizationTransform,
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
    fix_seed = 2023 # origin_seed 2023
    random.seed(fix_seed)
    np.random.seed(fix_seed)
    torch.manual_seed(fix_seed)

    args = get_args()
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
