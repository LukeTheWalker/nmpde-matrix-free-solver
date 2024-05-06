# Matrix-free solver for the Advection-Diffusion-Reaction problem
This project implements a matrix-free solver for advection-diffusion-reaction (ADR) problems. It then compare the performance of the matrix-free solver with that of a matrix-based solver, in terms of computational efficiency, parallel scalability and complexity of the implementation.

This solver is based on the [deal.II library](https://github.com/dealii/dealii), specifically on its [step-37](https://www.dealii.org/current/doxygen/deal.II/step_37.html) tutorial program.

The report for this project can be found here (TODO).

## Required software
- dealii - version 9.5.1 (or newer)
- CMake
## Compile and run
- Clone the repository
- Create a build directory and move into here and create an output directory
  ```
  mkdir build
  cd build
  mkdir output
  ```
- Run cmake and make
  ```
  cmake ..
  make
  ```
- Execute the matrixfree and matrixbased versions (mpi is also supported)
  ```
  ./matrixfree [solve | convergence]
  ./matrixbased
  ```
- Output .vtu files can be found in `/build/output` folder for the matrixfree algorithm and in `/build` folder for the matrixbased one.
