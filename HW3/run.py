import os

decompositions = [
    'moving_average',
    'differential_decomposition',
    'STL_decomposition',
    'X11_decomposition'
]

model_names = [
    'TsfKNN',
    'ThetaMethod',
    'ResidualModel',
    'ARIMA'

]
data_paths = [
    './src/dataset/ETT-small/ETTh1.csv',
    './src/dataset/ETT-small/ETTh2.csv'
]

if __name__ == '__main__':
    for data_path in data_paths:
        for model_name in model_names:
            if model_name == 'TsfKNN':
                for decomposition in decompositions:
                    run = ('python main.py' + ' --data_path ' + data_path + ' --model ' + model_name + ' --decomposition ' + decomposition +
                           ' >> ./res/' + model_name + '2.txt 2>&1')
                    # print(run)
                    os.system(run)
            else:
                run = ('python main.py' + ' --data_path ' + data_path + ' --model ' + model_name +
                       ' >> ./res/' + model_name + '2.txt 2>&1')
                # print(run)
                os.system(run)
