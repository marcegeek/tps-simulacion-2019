from code.random import MiddleSquareGenerator, LinearCongruentialGenerator


def middle_square_method(z0, n):
    print('Método del cuadrado de los medios (z_0={}):'.format(z0))
    print('z_0: {}, z^2: {}'.format(z0, z0 ** 2))
    rng = MiddleSquareGenerator(seed=z0)
    sample = rng.randint_raw(n)
    for i in range(n):
        u = sample[i] / 10000.
        print('z_{i}: {z}, u_{i} = {u}, z_{i}^2: {z2}'.format(i=i + 1, z=sample[i], u=u, z2=sample[i] ** 2))


def linear_congruential_method(mod, mult, incr, seed, n):
    print('Método congruencial lineal con m={}, a={}, c={} y z_0={}'.format(mod, mult, incr, seed))
    rng = LinearCongruentialGenerator(mod, mult, incr, seed=seed)
    sample = rng.random_sample(n)
    for i in range(n):
        print('u_{}: {}'.format(i + 1, sample[i]))


def main():
    middle_square_method(1009, 15)
    print()
    linear_congruential_method(16, 5, 3, 7, 32)
    print()
    linear_congruential_method(16, 4, 3, 7, 32)
    print()
    linear_congruential_method(16, 5, 4, 7, 32)
    print()
    linear_congruential_method(16, 4, 4, 7, 32)


if __name__ == '__main__':
    main()
