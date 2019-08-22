# OrthoBasis
A python module for orthonormalizing an arbitrary set of one-dimensional functions (nor arrays!) within an arbitrary interval using the Gram-Schmidt process and numerical integration.

The used scalar product is defined by

<img src="https://github.com/ahmetcik/OrthoBasis/blob/master/docs/Scalar_product_1.png" width="40%">

The method 'get_scalar_product' could be overwritten in order to
change the scalar product definition or adjust numerical parameters
as numerical stability is an issue. 
The stability should be always checked, e.g. by confirming that the scalar products 
between two different orthonormalized functions is zero.

Note that the modified Gram-Schmidt implementation is used as default which 
is numeracially more stable, however, computationally more demanding 
than the straightforward implementaion (modified_gs=False).




# Examples

## Polynomials
A simple example orthonormalizing the polynomials 1, x, x^2, x^3 for the interval [-1, 1].

```py
from ortho_basis import OrthoBasis
import matplotlib.pyplot as plt
import numpy as np

# create list of polynomial functions
v_list = []
exponents = np.arange(4)
for d in exponents: 
    def f(x, d=d):
        return x**d
    v_list.append(f)

interval = (-1, 1)
ob = OrthoBasis(interval=interval, modified_gs=True) 
ob.fit(v_list)

colors = ['g', 'r', 'b', 'k']
x = np.linspace(*interval, 100)

# get outputs in orthonormal basis set B either through linear mapping
# of outputs of reference basis set
F = x[..., np.newaxis]**exponents
B = ob.transform(F)

# or calculate input directly in orthonormal basis set
B = np.transpose([b(x) for b in ob.b_list])

# Analytical solution for orthonormal basis set (for comparison)
B_analytical = [np.ones(x.size) * 1/np.sqrt(2),
                np.sqrt(3. / 2.)                   * x,
                np.sqrt(45. / 8.)                  * (x**2 - 1. / 3.),
                np.sqrt(1. / (2. / 7. - 6. / 25.)) * (x**3 - 3. / 5. * x)]

for i in exponents:
    plt.plot(x, B_analytical[i], '%s-'  %colors[i])
    plt.plot(x, B[:, i], '%s:' %colors[i], linewidth=3)
plt.show()
```

The solid lines represent the analytical solutions while the dashed ones represent the outcomes of the code.
![alt text](https://github.com/ahmetcik/OrthoBasis/blob/master/docs/Polynomials.png)

Check numerical stability by looking at the correlation matrix (scalar products between all 'orthonormal' functions).
```py
print(ob.get_corr_matrix())
print(abs(ob.get_corr_matrix() - np.eye(exponents.size)).max())
```
With increasing number of Gram-Schmidt iterations (adding new orthonormal functions) the errors should increase. 

## Define own scalar product
In order to use a different scalar product, for example

<img src="https://github.com/ahmetcik/OrthoBasis/blob/master/docs/Scalar_product_2.png" width="40%">

or different integration parameters replace the method get_scalar_product by a custom function
```py
import scipy.integrate as integrate
def get_scalar_product(v1, v2):
    return integrate.quad(lambda x: v1(x) * v2(x) * np.exp(-x), *ob.interval, epsabs=1.49e-18)[0]

interval = (0, np.inf)
ob = OrthoBasis(interval=interval, modified_gs=True)
ob.get_scalar_product = get_scalar_product
ob.fit(v_list)
```










