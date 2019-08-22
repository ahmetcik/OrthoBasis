from ortho_basis import OrthoBasis
import matplotlib.pyplot as plt 
import numpy as np
import scipy.integrate as integrate



print(integrate.quad(lambda x: np.exp(-100*x), 0, 100, epsabs=1.49e-48, epsrel=1.49e-48, limit=500,))

exi()
# create list of polynomial functions
v_list = []
exponents = np.arange(4)
for d in exponents: 
    def f(x, d=d):
        return x**d
    v_list.append(f)

def get_scalar_product(v1, v2):
    return integrate.quad(lambda x: v1(x) * v2(x) * np.exp(-100*x), *ob.interval, epsabs=1.49e-48, epsrel=1.49e-48, limit=500,)[0]

interval = (0, np.inf)
ob = OrthoBasis(interval=interval, modified_gs=True) 
#ob.get_scalar_product = get_scalar_product
ob.fit(v_list)

colors = ['g', 'r', 'b', 'k']
x = np.linspace(*interval, 100)

# get outputs in orthonormal basis set B either through linear mapping
# of outputs of reference basis set 
F = x[..., np.newaxis]**exponents
B = ob.transform(F)

# or calculate input directly in orthonormal basis set 
B = np.transpose([b(x) for b in ob.b_list])
"""
# Analytical solution for orthonormal basis set (for comparison)
B_analytical = [np.ones(x.size) * 1/np.sqrt(2),
                np.sqrt(3. / 2.)                   * x,
                np.sqrt(45. / 8.)                  * (x**2 - 1. / 3.),
                np.sqrt(1. / (2. / 7. - 6. / 25.)) * (x**3 - 3. / 5. * x)] 

for i in exponents:
    plt.plot(x, B_analytical[i], '%s-'  %colors[i])
    plt.plot(x, B[:, i], '%s:' %colors[i], linewidth=3)
"""
print(ob.get_corr_matrix())
print(abs(ob.get_corr_matrix() - np.eye(4)).max())
