# OrthoBasis
A python module for orthonormalizing an arbitrary set of one-dimensional functions (nor arrays!) within an arbitrary interval using the Gram-Schmidt process and numerical integration.

The used scalar product is defined by

<img src="https://github.com/ahmetcik/OrthoBasis/blob/master/docs/Scalar_product.png" width="40%">

The method 'get_scalar_product' could be overwritten in order to
change the scalar product definition or adjust numerical parameters
as numerical stability is an issue. 
The stability should be always checked, e.g. by confirming that the scalar products 
between two different orthonormalized functions is zero.

# Examples

# Polynomials
A simple example orthonormalizing the polynomials 1, x, x^2 for the interval [-1, 1].

```py
from ortho_basis import OrthoBasis
import matplotlib.pyplot as plt

v_list = []
exponents = np.arange(3)
for d in exponents: 
    def f(x, d=d):
        return x**d
    v_list.append(f)

ob = OrthoBasis(interval=(-1, 1), modified_gs=True) 
ob.fit(v_list)

colors = ['g', 'r', 'b', 'k']
x = np.linspace(-1, 1, 100)

# get outputs in orthonormal basis set B either through linear mapping
# of outputs of reference basis set
F = x[..., np.newaxis]**exponents
B = ob.transform(F)

# or calculate input directly in orthonormal basis set
B = np.transpose([b(x) for b in ob.b_list])

# Analytical solution for orthonormal basis set (for comparison)
B_analytical = [np.ones(x.size) * 1/np.sqrt(2),
                np.sqrt(3/2)*x,
                np.sqrt(45/8)*(x**2-1./3.)]

for i in exponents:
    plt.plot(x, B_analytical[i], '%s-'  %colors[i])
    plt.plot(x, B[:, i], '%s:' %colors[i], linewidth=3)
plt.show()

```

![alt text](https://github.com/ahmetcik/OrthoBasis/blob/master/docs/Polynomials.png)










