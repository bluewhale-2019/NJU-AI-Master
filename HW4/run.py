import os

model_names = [
    'TsfKNN',
    'DLinear',
    'SPIRIT',
    'ThetaMethod',
]
data_paths = [
    './src/dataset/ETT-small/ETTh1.csv',
    './src/dataset/ETT-small/ETTh2.csv',
]

patterns = [
    'global',
    'local'
]

if __name__ == '__main__':
    for data_path in data_paths:
        for model_name in model_names:
            for pattern in patterns:
                run = ('python main.py' + ' --data_path ' + data_path + ' --model ' + model_name + ' --pattern ' + pattern +
                       ' >> ./res/' + model_name + '.txt 2>&1')
                # print(run)

                os.system(run)