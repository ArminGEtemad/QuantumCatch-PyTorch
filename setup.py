from setuptools import setup, find_packages

setup(
    name="quantumcatch-pytorch",
    version="0.1.0",
    author="Armin Ghorbanietemad",
    author_email="armin.ghorbanietemad@rub.de",
    description="GPU-accelerated Python library for analytical higher-order quantum spectra",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ArminGEtemad/QuantumCatch-PyTorch.git",
    license="Apache-2.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.3",
        "numpy>=1.24",
        "matplotlib>=3.7",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
