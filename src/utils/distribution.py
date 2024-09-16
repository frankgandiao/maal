import torch as th

REGISTRY = {}
REGISTRY["DiagGaussian"] = DiagGaussianPd

class Pd(object):
    """
    A particular probability distribution
    """
    def flatparam(self):
        raise NotImplementedError
    def mode(self):
        raise NotImplementedError
    def logp(self, x):
        raise NotImplementedError
    def kl(self, other):
        raise NotImplementedError
    def entropy(self):
        raise NotImplementedError
    def sample(self):
        raise NotImplementedError


class DiagGaussianPd(Pd):
    def __init__(self, flat):
        self.flat = flat
        mean, logstd = th.split(flat, split_size_or_sections=2, axis=1)
        self.mean = mean
        self.logstd = logstd
        self.std = th.exp(logstd)
    def flatparam(self):
        return self.flat        
    def mode(self):
        return self.mean
    # def logp(self, x):
    #     return - 0.5 * U.sum(tf.square((x - self.mean) / self.std), axis=1) \
    #            - 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(x)[1]) \
    #            - U.sum(self.logstd, axis=1)
    # def kl(self, other):
    #     assert isinstance(other, DiagGaussianPd)
    #     return U.sum(other.logstd - self.logstd + (tf.square(self.std) + tf.square(self.mean - other.mean)) / (2.0 * tf.square(other.std)) - 0.5, axis=1)
    def entropy(self):
        return th.sum(self.logstd + .5 * np.log(2.0 * np.pi * np.e), 1)
    def sample(self):
        return self.mean + self.std * th.normal(th.zeros_like(self.mean), th.ones_like(self.mean))
    @classmethod
    def fromflat(cls, flat):
        return cls(flat)