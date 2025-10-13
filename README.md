# ParODE

Parallel fixed-point solver for ODEs using GPU-accelerated polynomial collocation.

## Overview
This method solves `dy/dt = f(y,t)` by fitting a separable basis `F(y,t)` that approximates the flow field directly, avoiding sequential integration.

## Features
- Fully GPU-parallel formulation  
- Monomial basis (y^m t^n)  
- Initial condition enforced analytically  
- Newton solver for implicit fixed-point equation  
- Built-in RK4 comparison  

## Example

```bash
python parode_example.py
