# PINE VDAF Reference Implementation

This directory contains SageMath implementations of PINE VDAF.

## Installation

This code is compatible with SageMath version 9.6. To install Sage,
follow [Sage's installation guide](https://doc.sagemath.org/html/en/installation/index.html).

In order to run the code you will need to install
[PyCryptodome](https://pycryptodome.readthedocs.io/en/latest/index.html).

```
sage --pip install pycryptodomex
```

Version 3.20.0 or later is required.

## Run unit tests

```
sage -python -m unittest
```

## Generating test vectors

```
sage -python gen_test_vec.py
```
