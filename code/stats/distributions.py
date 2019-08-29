import abc

import numpy as np
import scipy.special

from code import stats


class ProbabilityDistribution(abc.ABC):

    def __init__(self, seed=None):
        self._discrete = False
        self.random_state = np.random.RandomState(seed)
        self.generator = self._generator()

    def is_continuous(self):
        return not self.is_discrete()

    def is_discrete(self):
        return self._discrete

    @abc.abstractmethod
    def theoretical_distribution(self, x):
        return np.array([1.] * len(x))

    def cumulative_distribution(self, x):
        if np.ndim(x) == 0:
            return self._cdf(x)
        else:
            r = np.zeros(len(x))
            for i in range(len(x)):
                r[i] = self._cdf(x[i])
            return r

    def percent_point(self, p):
        if np.ndim(p) == 0:
            return self._ppf(p)
        else:
            r = np.zeros(len(p))
            for i in range(len(p)):
                r[i] = self._ppf(p[i])
            return r

    @abc.abstractmethod
    def mean(self):
        return .5

    @abc.abstractmethod
    def variance(self):
        return 1/12.

    def standard_deviation(self):
        return np.sqrt(self.variance())

    @abc.abstractmethod
    def value_range(self):
        return 0., 1.

    def x_array(self, lower_limit=None, upper_limit=None):
        first, last = self.value_range()
        if lower_limit is not None:
            first = lower_limit
        if upper_limit is not None:
            last = upper_limit
        if self.is_continuous():
            return np.linspace(first, last, num=1000)
        else:
            return np.arange(first, last + 1)

    def random_sample(self, size=None):
        if size is not None:
            return np.array([next(self.generator) for _ in range(size)])
        return next(self.generator)

    def plot_theoretical(self, axes, label=None):
        x = self.x_array()
        if self.is_continuous():
            axes.plot(x, self.theoretical_distribution(x), label=label)
        else:
            axes.plot(x, self.theoretical_distribution(x), 'o-', label=label)

    @abc.abstractmethod
    def _cdf(self, x):
        # implementación básica por defecto
        v = self.x_array(upper_limit=x)
        y = self.theoretical_distribution(v)
        if self.is_continuous():
            return np.trapz(y, v)
        else:
            return y.sum()

    @abc.abstractmethod
    def _ppf(self, p):
        # implementación básica por defecto
        x = self.x_array()
        for i in range(len(x)):
            if self._cdf(x[i]) >= p:
                return x[i]
        return np.nan

    def _generator(self):
        while True:
            for r in self._distribution_values():
                yield r

    @abc.abstractmethod
    def _distribution_values(self):
        return [self.random_state.random_sample()]


class UniformDistribution(ProbabilityDistribution):

    def __init__(self, a, b, seed=None):
        super().__init__(seed=seed)
        self.a = a
        self.b = b

    def theoretical_distribution(self, x):
        dist = np.zeros(len(x))
        for i in range(len(x)):
            if self.a <= x[i] <= self.b:
                dist[i] = 1. / (self.b - self.a)
        return dist

    def mean(self):
        return (self.a + self.b)/2.

    def variance(self):
        return (self.b - self.a)**2/12.

    def value_range(self):
        return self.a, self.b

    def _cdf(self, x):
        if x < self.a:
            return 0.
        elif x > self.b:
            return 1.
        return (x - self.a)/(self.b - self.a)

    def _ppf(self, p):
        if not 0 <= p <= 1:
            return np.nan
        return self.a + (self.b - self.a) * p

    def _distribution_values(self):
        u = self.random_state.random_sample()
        r = self.a + (self.b - self.a) * u
        return [r]


class ExponentialDistribution(ProbabilityDistribution):

    def __init__(self, alpha, seed=None):
        super().__init__(seed=seed)
        self.alpha = alpha

    def theoretical_distribution(self, x):
        dist = np.zeros(len(x))
        for i in range(len(x)):
            if x[i] >= 0:
                dist[i] = self.alpha * np.exp(-self.alpha * x[i])
        return dist

    def mean(self):
        return 1./self.alpha

    def variance(self):
        return self.mean()**2

    def value_range(self):
        # cubrir el 99.9% de los valores
        return 0, 7. / self.alpha

    def _cdf(self, x):
        return super()._cdf(x)

    def _ppf(self, p):
        return super()._ppf(p)

    def _distribution_values(self):
        u = self.random_state.random_sample()
        r = -1. / self.alpha * np.log(u)
        return [r]


class GammaDistribution(ProbabilityDistribution):

    def __init__(self, k, alpha, seed=None):
        super().__init__(seed=seed)
        self.alpha = alpha
        self.k = k

    def theoretical_distribution(self, x):
        return self.alpha ** self.k * x ** (self.k - 1) * np.exp(-self.alpha * x) / scipy.special.gamma(self.k)

    def mean(self):
        return self.k/self.alpha

    def variance(self):
        return self.k/self.alpha**2

    def value_range(self):
        first = self.percent_point(0.0001)
        last = self.percent_point(0.9999)
        return first, last

    def _cdf(self, x):
        return scipy.special.gammaincc(self.k, 0) - scipy.special.gammaincc(self.k, self.alpha * x)

    def _ppf(self, p):
        return scipy.special.gammainccinv(self.k, scipy.special.gammaincc(self.k, 0) - p)/self.alpha

    def _distribution_values(self):
        k = int(np.floor(self.k))
        if k != self.k:
            q = self.k - k
            if self.random_state.random_sample() < q:
                k += 1
        u = self.random_state.random_sample(k)
        r = -1. / self.alpha * np.log(u.prod())
        return [r]


class NormalDistribution(ProbabilityDistribution):

    def __init__(self, mu, sigma, seed=None):
        super().__init__(seed=seed)
        self.mu = mu
        self.sigma = sigma

    def theoretical_distribution(self, x):
        return 1 / np.sqrt(2 * np.pi * self.sigma ** 2) * np.exp(-(x - self.mu) ** 2 / (2 * self.sigma ** 2))

    def mean(self):
        return self.mu

    def variance(self):
        return self.sigma**2

    def value_range(self):
        # cubrir el 99.99% de los valores
        delta = 4 * self.sigma
        return self.mu - delta, self.mu + delta

    def _cdf(self, x):
        return super()._cdf(x)

    def _ppf(self, p):
        return super()._ppf(p)

    def _distribution_values(self):
        u1 = self.random_state.random_sample()
        u2 = self.random_state.random_sample()
        r = np.sqrt(-2 * np.log(u1))
        z0, z1 = r * np.cos(2 * np.pi * u2), r * np.sin(2 * np.pi * u2)
        x0 = z0 * self.sigma + self.mu
        x1 = z1 * self.sigma + self.mu
        return [x0, x1]


class BinomialDistribution(ProbabilityDistribution):

    def __init__(self, n, p, seed=None):
        super().__init__(seed=seed)
        self._discrete = True
        self.n = n
        self.p = p

    def theoretical_distribution(self, x):
        return scipy.special.binom(self.n, x) * self.p**x * (1 - self.p)**(self.n - x)

    def mean(self):
        return self.n * self.p

    def variance(self):
        return self.n * self.p * (1 - self.p)

    def value_range(self):
        delta = self.standard_deviation() * 4
        return max(np.round(self.mean() - delta), 0.), np.round(self.mean() + delta)

    def _cdf(self, x):
        return super()._cdf(x)

    def _ppf(self, p):
        return super()._ppf(p)

    def _distribution_values(self):
        r = 0
        for i in range(self.n):
            u = self.random_state.random_sample()
            if u < self.p:
                r += 1
        return [r]


class PoissonDistribution(ProbabilityDistribution):

    def __init__(self, lmbd, seed=None):
        super().__init__(seed=seed)
        self._discrete = True
        self.lmbd = lmbd

    def theoretical_distribution(self, x):
        return np.exp(x * np.log(self.lmbd) - self.lmbd - scipy.special.loggamma(x + 1))

    def mean(self):
        return self.lmbd

    def variance(self):
        return self.lmbd

    def value_range(self):
        delta = self.standard_deviation() * 4
        return max(np.round(self.mean() - delta), 0.), np.round(self.mean() + delta)

    def _cdf(self, x):
        return super()._cdf(x)

    def _ppf(self, p):
        return super()._ppf(p)

    def _distribution_values(self):
        r = 0
        b = np.exp(-self.lmbd)
        tr = 1.
        while tr >= b:
            u = self.random_state.random_sample()
            tr *= u
            if tr >= b:
                r += 1
        return [r]


class EmpiricalDistribution(ProbabilityDistribution):

    def __init__(self, distribution, seed=None):
        super().__init__(seed=seed)
        self.freq_distribution = distribution
        self._discrete = self.freq_distribution.is_discrete()
        cumulative = self.freq_distribution.cumulative_distribution()
        # agrego frecuencia ficticia (siempre menor que r)
        # para facilitar la generación cuando cae en la
        # primera clase/valor
        self.cumulative = np.insert(cumulative, 0, -1.)

    def theoretical_distribution(self, x):
        arr = np.zeros(len(x))
        for i in range(len(x)):
            arr[i] = self.freq_distribution.get_freq(x[i])
        return arr

    def mean(self):
        return self.freq_distribution.mean()

    def variance(self):
        return self.freq_distribution.variance()

    def value_range(self):
        first = self.freq_distribution[0]
        last = self.freq_distribution[-1]
        if self.is_continuous():
            return first.val[0], last.val[1]
        return first.val, last.val

    def _cdf(self, x):
        return super()._cdf(x)

    def _ppf(self, p):
        return super()._ppf(p)

    def _distribution_values(self):
        r = self.random_state.random_sample()
        for i in range(len(self.cumulative) - 1):
            if self.cumulative[i] < r <= self.cumulative[i + 1]:  # encontrada la clase/valor
                v = self.freq_distribution[i].val
                if self.is_continuous():
                    r = self.random_state.random_sample()  # a ver donde cae dentro del rango
                    return [v[0] + (v[1] - v[0]) * r]
                else:
                    return [v]


class RandomDistributionPopulation(stats.RandomPopulation):

    def __init__(self, distribution, size=1000):
        self.distribution = distribution
        super().__init__(self.distribution.random_sample(size))

    def plot_theoretical(self, axes, label=None):
        self.distribution.plot_theoretical(axes, label=label)


# clases envoltorio de conveniencia

class UniformRandomPopulation(RandomDistributionPopulation):

    def __init__(self, a, b, size=1000, seed=None):
        super().__init__(UniformDistribution(a, b, seed=seed),
                         size=size)


class ExponentialRandomPopulation(RandomDistributionPopulation):

    def __init__(self, alpha, size=1000, seed=None):
        super().__init__(ExponentialDistribution(alpha, seed=seed),
                         size=size)


class GammaRandomPopulation(RandomDistributionPopulation):

    def __init__(self, k, alpha, size=1000, seed=None):
        super().__init__(GammaDistribution(k, alpha, seed=seed),
                         size=size)


class NormalRandomPopulation(RandomDistributionPopulation):

    def __init__(self, mu, sigma, size=1000, seed=None):
        super().__init__(NormalDistribution(mu, sigma, seed=seed),
                         size=size)


class BinomialRandomPopulation(RandomDistributionPopulation):

    def __init__(self, n, p, size=1000, seed=None):
        super().__init__(BinomialDistribution(n, p, seed=seed),
                         size=size)


class PoissonRandomPopulation(RandomDistributionPopulation):

    def __init__(self, lmbd, size=1000, seed=None):
        super().__init__(PoissonDistribution(lmbd, seed=seed),
                         size=size)


class EmpiricalRandomPopulation(RandomDistributionPopulation):

    def __init__(self, distribution, size=1000, seed=None):
        super().__init__(EmpiricalDistribution(distribution, seed=seed),
                         size=size)
