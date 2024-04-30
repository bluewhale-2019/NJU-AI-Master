import os

data_dirs = [
    './src/dataset/electricity/electricity.csv',
    './src/dataset/exchange_rate/exchange_rate.csv',
    './src/dataset/illness/national_illness.csv',
    './src/dataset/traffic/traffic.csv',
    './src/dataset/weather/weather.csv'
]

model_dirs = [
    'AutoRegressiveLinearForecast',
    'ExponentialSmoothingForecast'
]

transform_dirs = [
    'IdentityTransform',
    'Normalization',
    'Standardization',
    'MeanNormalization',
    'BoxCox'
]

if __name__ == '__main__':
    for data_dir in data_dirs:
        for model_dir in model_dirs:
            for transform_dir in transform_dirs:
                run = 'python main.py' + ' --data_path ' + data_dir + ' --dataset ' + 'Custom' + ' --model ' + model_dir + ' --transform ' + transform_dir
                res = data_dir + ' ' + model_dir + ' ' + transform_dir
                print(res)
                os.system(run)
