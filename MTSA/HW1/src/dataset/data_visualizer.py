import matplotlib.pyplot as plt

def data_visualize(dataset, t):
    """
    Choose t continous time points in data and visualize the chosen points. Note that some datasets have more than one
    channel.
    param:
        dataset: dataset to visualize
        t: the number of timestamps to visualize
    """

    train_X = dataset.train_data
    channel_size = train_X.shape[-1]
    data_cols = dataset.data_cols
    for channel in range(channel_size):
        sample = train_X[0, :, channel]
        plt.plot(range(1, t + 1), sample[:t])
    plt.xlabel('timestamps')
    plt.ylabel('channel ')
    plt.legend(data_cols)
    plt.title('Custom weather')
    plt.show()
