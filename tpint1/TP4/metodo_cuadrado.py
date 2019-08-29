#!/usr/bin/env python3

def nuevo_z(z):
    z_squared = z ** 2
    last_digits = z_squared // 100
    divided = last_digits/10000.
    int_part = int(divided)
    return int((divided - int_part) * 10000)


def get_u(z):
    return z/10000


def genz(z, n):
    for i in range(n):
        z = nuevo_z(z)
        yield z/10000.



def generador_cuadrado_medios(z, n=11):
    print('Método del cuadrado de los medios (z_0={}):'.format(z))
    print('z_0: {}, z^2: {}'.format(z, z ** 2))
    for i in range(n):
        z = nuevo_z(z)
        u = z/10000
        print('z_{i}: {}, u_{i} = {}, z_{i}^2: {}'.format(z, u, z ** 2, i=i + 1))


def main():
    generador_cuadrado_medios(1009)
    for u in genz(1009, 10):
        print('u: {}'.format(u))


if __name__ == '__main__':
    main()
