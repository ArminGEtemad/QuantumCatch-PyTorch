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

The speed up is as follows on ARM64 MacBook:
| N points  |   Loop time (s) |  Batch time (s) |    Speedup   |
|-----------|-----------------|-----------------|--------------|
|       128 |        0.018338 |        0.000131 |     139.72   |
|       256 |        0.005807 |        0.000143 |      40.72   |
|       512 |        0.011198 |        0.000126 |      89.11   |
|      1024 |        0.021214 |        0.000305 |      69.59   |
|      2048 |        0.044244 |        0.000360 |     122.80   |
|      4096 |        0.088666 |        0.000358 |     247.84   |
|      8192 |        0.174481 |        0.000608 |     287.09   |
|     16384 |        0.350798 |        0.000965 |     363.38   |
|     32768 |        0.693620 |        0.001852 |     374.49   |
|     65536 |        1.395504 |        0.002978 |     468.57   |
|    131072 |        2.731546 |        0.005693 |     479.77   |
|    262144 |        5.615536 |        0.011920 |     471.12   |
|    524288 |       11.740128 |        0.020826 |     563.73   |

The Speedup on 13th Gen Intel Core i9-13900KS:
|  N_points |   Loop_time (s) |  Batch_time (s) |    Speedup   |
|-----------|-----------------|-----------------|--------------|
|       128 |        0.006599 |        0.000196 |      33.69|
|      256 |        0.011598 |        0.000154 |      75.41|
|       512 |        0.023642 |        0.000298 |      79.34|
|     1024 |        0.047164 |        0.002253 |      20.93|
|      2048 |        0.090894 |        0.001509 |      60.22|
|     4096 |        0.180787 |        0.001178 |     153.53|
|     8192 |        0.355613 |        0.002070 |     171.80|
|    16384 |        0.730620 |        0.002693 |     271.30|
|     32768 |        1.417991 |        0.003015 |     470.28|
|    65536 |        2.917571 |        0.004534 |     643.43|
|   131072 |        5.790812 |        0.005603 |    1033.48|
|   262144 |       11.581944 |        0.008771 |    1320.48|
|   524288 |       23.806667 |        0.014420 |    1651.01|

The Speedup on NVIDIA GeForce RTX 4090:
  N_points |   Loop_time (s) |  Batch_time (s) |    Speedup
|-----------|-----------------|-----------------|--------------|
|       128 |        0.131187 |        0.002863 |      45.82   |
|       256 |        0.075124 |        0.000342 |     219.40   |
|       512 |        0.139944 |        0.000341 |     410.76   |
 |     1024 |        0.280051 |        0.000349 |     801.75   |
 |     2048 |        0.548806 |        0.000377 |    1455.33   |
 |     4096 |        1.072313 |        0.000378 |    2835.31   |
 |     8192 |        2.160001 |        0.000531 |    4068.56   |
 |    16384 |        4.674778 |        0.000555 |    8415.44   |
 |    32768 |       10.769831 |        0.000857 |   12571.30   |
|     65536 |       22.135232 |        0.000993 |   22280.05   |
 |   131072 |       45.085042 |        0.001193 |   37794.49   |
 |   262144 |       89.326903 |        0.001710 |   52241.01   |
 |   524288 |      177.806920 |        0.005040 |   35281.95   |

