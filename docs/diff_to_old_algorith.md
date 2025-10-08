# Differences to the Old Algorithm

## Liouvillian
The Liouvillian matrix is now analytically constructed and since it avoids looping over $n^2$ basis vectors it is faster. This analytical appraoches allows for clear separation in the dynamics.

The old algorithm uses the following mathematical equation:
```math
L = i[\rho, H] +\sum_j \Big[c_j\rho c_j^\dagger - \frac{1}{2} (c_j^\dagger c_j \rho + \rho c_j^\dagger c_j)\Big]
```

where the new algorithm separates the terms to be consitent with the publications:

```math
L = i(H\otimes I - I \otimes H) + \sum_k \gamma_k \Big[(d^*_k \otimes d_k) - \frac{1}{2}(I \otimes d_k^\dagger d_k + d_k^Td_k^* \otimes I)\Big]
+ \sum_k \beta_k^2\Big[(c^*_k \otimes c_k)- \frac{1}{2}(I \otimes c_k^\dagger c_k + c_k^Tc_k^* \otimes I)\Big]
```