# Differences to the Old Algorithm

## Liouvillian
The Liouvillian matrix is now analytically constructed and since it avoids looping over $n^2$ basis vectors it is faster. This analytical appraoches allows for clear separation in the dynamics.

The old algorithm uses the following mathematical equation:
```math
L = i[\rho, H] +\sum_j \Big[c_j\rho c_j^\dagger - \frac{1}{2} (c_j^\dagger c_j \rho + \rho c_j^\dagger c_j)\Big]
```
To program this equation, we have to use a vectorized $\rho$ and then reshape it to the dimensions we need. After that, we “probe” it column by column to build the superoperator.
whereas the new algorithm separates the terms to be consitent with the publications and does not need to reshape or vectorize any `tensor` on sight:

```math
L = i(H\otimes I - I \otimes H) + \sum_k \gamma_k \Big[(d^*_k \otimes d_k) - \frac{1}{2}(I \otimes d_k^\dagger d_k + d_k^Td_k^* \otimes I)\Big]
+ \sum_k \beta_k^2\Big[(c^*_k \otimes c_k)- \frac{1}{2}(I \otimes c_k^\dagger c_k + c_k^Tc_k^* \otimes I)\Big]
```

Since we now separate the strength from the operator itself, we don't need to replace $\gamma$ with $\sqrt{\gamma}$ like in the old algorithm anymore.

## Measurement superoperators
The measurements superoperators are now defined consistent with our publications and is calculated via:

```math
\mathcal{A}_k = \left(\frac{I \otimes c_k + c^*_k \otimes I}{2}\right)
```

## Modified measurement superoperator
The definition of the modified measurement superoperator is:
```math
\mathcal{A}^\prime _k = \left(\frac{I \otimes c_k + c^*_k \otimes I}{2}\right)
```
