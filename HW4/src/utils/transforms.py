import numpy as np
class Transform:
    """
    Preprocess time series
    """

    def transform(self, data):
        """
        :param data: raw timeseries
        :return: transformed timeseries
        """
        raise NotImplementedError

    def inverse_transform(self, data):
        """
        :param data: raw timeseries
        :return: inverse_transformed timeseries
        """
        raise NotImplementedError


class IdentityTransform(Transform):
    def __init__(self, args):
        pass

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data

# TODO: add other transforms
class Standardization(Transform):
    def __init__(self, args):
        self.mu = []
        self.sigma = []
        self.aug = []

    def transform(self, data):
        self.mu = []
        self.sigma = []
        self.aug = []
        if len(data.shape) == 3:
            for i in range(data.shape[2]):
                self.mu.append(np.mean(data[0, :, i]))
                self.sigma.append(np.var(data[0, :, i]))
                self.aug.append(np.ones(data[0, :, i].shape) * self.mu[-1])
                data[0, :, i] = (data[0, :, i] - self.aug[-1]) / (self.sigma[-1])
        else:
            for sample in range(data.shape[0]):
                self.mu.append(np.mean(data[sample]))
                self.sigma.append(np.var(data[sample]))
                self.aug.append(np.ones(data[sample].shape) * self.mu[-1])
                data[sample] = (data[sample] - self.aug[-1]) / (self.sigma[-1])

        return data
