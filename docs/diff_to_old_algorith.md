# Differences to the Old Algorithm

## Basic Superoperators
### Liouvillian
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

### Measurement superoperators
The measurements superoperators are now defined consistent with our publications and is calculated via:

```math
\mathcal{A}_k = \left(\frac{I \otimes c_k + c^*_k \otimes I}{2}\right)
```
This also fixes the old code where the superoperator was mirrored because of a wrong asumption in the algorithm. 

### Modified measurement superoperator
The definition of the modified measurement superoperator is:
```math
\mathcal{A}^\prime _k x = \mathcal{A}_k x - \text{Tr}(\mathcal{A_k}\rho_\text{steady})x
```
It stays actually as before and is part of the documentation for the sake of completeness. 

## Batching
To make the code much faster, we go with batching and vectorizing the algorithm. One of the helper function with batching algorith is

### $G(\nu)$
Is a propagator defined as:

```math
G(\nu) = \Lambda D_G(\nu)\Lambda^{-1}
```
with $D_\nu$ being a diagonal matrix with elements $[D]_{jj} = 1/(-\lambda_j - i\nu)$ *while one element corresponding to the steady-state stays zero*.

In the batched version the propagator in frequency domain is calculatated for all the frequencies at once and does not need the slow and old algorithm that used loop.

The speed up is as follows:
| N points  |   Loop time (s) |  Batch time (s) |    Speedup   |
|-----------|-----------------|-----------------|--------------|
|       128 |        0.018338 |        0.000131 |     139.72|
|       256 |        0.005807 |        0.000143 |      40.72|
|      512  |        0.011198 |        0.000126 |      89.11|
|      1024 |        0.021214 |        0.000305 |      69.59|
|      2048 |        0.044244 |        0.000360 |     122.80|
|      4096 |        0.088666 |        0.000358 |     247.84|
|      8192 |        0.174481 |        0.000608 |     287.09|
|     16384 |        0.350798 |        0.000965 |     363.38|
|     32768 |        0.693620 |        0.001852 |     374.49|
|     65536 |        1.395504 |        0.002978 |     468.57|
|    131072 |        2.731546 |        0.005693 |     479.77|
|    262144 |        5.615536 |        0.011920 |     471.12|
|    524288 |       11.740128 |        0.020826 |     563.73|

