import os

embedding_methods = [
    'lag_m_embedding',
    'fourier_embedding'
]

distance_methods = [
    'euclidean',
    'chebyshev'
]

decompositions = [
    'Moving_Average',
    'Differential_Decomposition'
]

model_names = [
    'TsfKNN',
    'DLinear'
]

if __name__ == '__main__':

    print('Evaluation 1')
    print('tau :32, m :3')
    for embedding_method in embedding_methods:
        for distance_method in distance_methods:
            run = 'python main.py' + ' --embedding ' + embedding_method + ' --distance ' + distance_method
            res = embedding_method + ' ' + distance_method
            print(res)
            os.system(run)



    print('Evaluation 2')
    for model_name in model_names:
        for decomposition in decompositions:
            run = 'python main.py' + ' --model ' + model_name + ' --decomposition ' + decomposition
            res = model_name + ' ' + decomposition
            print(res)
            os.system(run)
