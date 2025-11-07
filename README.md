# QuantumCatch-PyTorch

## Overview

**GPU-accelerated Python library for analytical higher-order spectra**

QuantumCatch-PyTorch is a Python library for calculating higher-order spectra (polyspectra) of continuous quantum measurements.
These spectra — the first-, second-, third-, and fourth-order correlations (commonly known as the mean signal, power spectrum, bispectrum, and trispectrum) provide detailed insights into non-Gaussian statistics and system dynamics.
However, for reasons discussed later, we avoid using those traditional names throughout this documentation.

Multi-Channel QuantumCatch is the successor to [QuantumCatch](https://github.com/MarkusSifft/QuantumCatch).
This new implementation is built on PyTorch rather than ArrayFire, providing a more familiar API, robust GPU acceleration and well-written documentations.
The switch simplifies installation across all OS (Windows, Linux, MacOS) and maintenance while enabling broader collaboration within the open-source community. Moreover, PyTorch is experimenting on MPS backend allowing Mac users to utilize the integrated graphic cards on their Apple Silicon processors in the future. 

In addition to improved performance (see [comparison here](docs/diff_to_old_algorith.md)), this version introduces multi-detector support, allowing users to simulate and analyze measurement spectra across multiple channels with flexible combinations of cross-spectra.

This library enables users to to simulate their quantum systems in a very general way by including:
- system Hamiltonian
- environmental damping (multiple operators)
- collapse strength (multiple strengths)
- measurement backaction (multiple channels)
- detector operators (multiple operators)

How long the program need to calculate the spectra depends on the hardware, backend, resolution of the spectrum. To give user a feeling of the calculation time the program was tested on Apple M2 Pro Chip, 13th Gen Intel Core i9-13900KS as well as NVIDIA GeForce RTX 4090 (CUDA backend). Results are summarized [here](docs/unique_vs_conventional_batch.md). There you can find two algorithmic variants were benchmarked against each other too. These tests guided the decision to retain the faster and more stable algorithm in the current release.

## Installation
Right now the package must be cloned and installed. The package will be installable via PiPy in the future. 
```bash
git clone https://github.com/ArminGEtemad/QuantumCatch-PyTorch.git
cd QuantumCatch-PyTorch
pip install -e .
```

## Examples and Architecture
You can find examples in [here](examples). 
You need to first define your quantum system using `SystemConfig` class. This class needs the 
- `hamiltonian`, 
- `collapse_ops`: which is a list of torch tensors
- `measurement_ops`: which is a dictionary with keys being the key for the detector and the value being the torch tensor
- `collapse_strength`: is a tensor of floats. The number of elements must be the same as the number of collapse operators since they describe the strength of these operators
- `measurement_strength`: is a dictionary. the keys correspond to the keys from `measurement_ops` dictionary. 

The next class is `SpectrumConfig`. In this class, you can configure everything about the calculation of the spectra. 
- `resolution`: must be a positive integer
- `f_min`: minimum frequency of the spectrum
- `f_max`: maximum frequency of the spectrum
- `order_combi`: This is a dictionary with three possible keys, `'s2'`, `'s3'` and `'s4'`. Each are representing second-, third- and fourth- order spectrum. Each of these keys can have a list as value. In these list, you write the order of the detectors in a tuple. For example: a measurement has two detectors:
    ```python
    measurement_ops = {
    '0': torch.tensor([[0., 0.], [0., 1.]]),
    '1': torch.tensor([[1., 0.], [0., 0.]])
    }

    beta_dict = {'0': 5., '1': 3.}
    ```
    Now for the second order spectrum, you have $2^2 = 4$ different combinations of detectors:
     - Auto-correlation spectrum $S^{(2)}_{00}$
     - Auto-correlation spectrum $S^{(2)}_{11}$
     - Cross-correlation spectrum $S^{(2)}_{10}$
     - Cross-correlation spectrum $S^{(2)}_{01}$
    
    You can specify which one of these combinations are to be calculated with `order_combi`. Let's say, you want $S^{(2)}_{11}$ and $S^{(2)}_{01}$. Then you should use:
    ```python
    order_combi = {
        's2': [('1', '1'), 
               ('0', '1')]
    }
    ```
    **It is very important to note that even if you are calculating $S^{(2)}_{11}$ the measurement was done using two measurement operators `'0'` and `'1'`. $S^{(2)}_{11}$ here is just the autocorrelation spectrum in a setup with two detectors.**
- `s3_block` and `s4_block`: Can be either `None` or an integer number. These are safety mechanism for the situations where the system and/or the resolution is too much for the computer to push into RAM/VRAM. With these block the calculation occurs in blocks step by step without overwhelming the RAM/VRAM. 
- `backend`: could be as of now `'cuda'` or `'cpu'`.

When the system and spectrum is configured, we are ready to calculate the spectra. To do that we feed both objects to `SpectrumCalculator` class. 

It is possible to plot the results too. In order to do that, you need to configure the plot with `PlotConfig` class. The configuration only need the `f_min` (minimum frequency), `f_max` (maximum frequency), the orders you want to plot with `display_orders` as a list with integer values `2`, `3` and `4` that correspond to second, third and fourth order, respectively. You can also decide if you are only interested in the real part, imaginary part or both part of the spectrum by using `plot_format`. After the configuration, we make the object `SpectrumPlotter` with the configuration and the results from `SpectrumCalculator`. The function `display()` can the plot the results. 