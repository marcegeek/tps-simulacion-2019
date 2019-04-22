import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

data = np.recfromcsv('Datos 03.csv')
data = data.astype(float)
bins = int(np.ceil(np.sqrt(len(data))))

# stats.chisquare()

hist, bin_edges = np.histogram(data, bins, density=True)

plt.hist(data, bins='sqrt', density=True, edgecolor='black', linewidth=1)

plt.grid()



#plt.bar(bin_edges[:-1], hist, width=.1)

vx = np.var(data)
ex = np.average(data)

print('La varianza es', vx)
print('La esperanza es', ex)


n = ex ** 2 / (ex - vx)
p = (ex - vx) / ex

ex = n * p
vx = n * p * (1 - p)

-(ex-vx)/(ex-vx)**2/ex**2 = n


print('Binomial con n={}, p={}'.format(n, p))


x = np.arange(stats.binom.ppf(0.01, n, p),
              stats.binom.ppf(0.99, n, p),
              step=1)
#plt.plot(x, stats.binom.pmf(x, n, p), 'bo', ms=8, label='binom pmf')
#plt.vlines(x, 0, stats.binom.pmf(x, n, p), colors='b', lw=5, alpha=0.5)

rv = stats.binom(n, p)
plt.plot(x, rv.pmf(x), 'b', label='binom pmf')
plt.vlines(x, 0, rv.pmf(x), colors='k', linestyles='-', lw=1,
           label='frozen pmf')
#plt.legend(loc='best', frameon=False)

xn = np.arange(stats.norm.ppf(0.01, scale=np.sqrt(vx), loc=ex),
               stats.norm.ppf(0.99, scale=np.sqrt(vx), loc=ex),
               step=.1)
plt.plot(xn, stats.norm.pdf(xn, scale=np.sqrt(vx), loc=ex))

plt.show()

