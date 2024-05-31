# Matrix-free solver for the Advection-Diffusion-Reaction problem
This project implements a matrix-free solver for advection-diffusion-reaction (ADR) problems. It then compare the performance of the matrix-free solver with that of a matrix-based solver, in terms of computational efficiency, parallel scalability and complexity of the implementation.
The matrixfree solver implements a Geometric MultiGrid preconditioner with Chebyshev iteration being a good fit for a matrixfree implementation since it only needs the computation of the diagonal.

This solver is based on the [deal.II library](https://github.com/dealii/dealii), mainly but not only on its [step-37](https://www.dealii.org/current/doxygen/deal.II/step_37.html) tutorial for the matrixfree solver and on other tutorials for the implementation of the matrixbased comparison.

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
- Execute the matrixfree and matrixbased versions, more details on the available tests that can be run is provided below
  ```
  [mpirun -n XX] ./<solver_type> <test> [parameters]
  ```
  The integer value `XX` is the required number of MPI processes to be spawned.
- Output .vtu files can be found in `/build/output_ZZ` folder where `ZZ = [mf | mb | mg]` respectively for the matrixfree with GMG, basic matrixbased w/o multigrid, and matrixbased with GMG.
- Available `<test> = [solve | convergence | dimension | polynomial] [optional parameters]` are described below.
- An optional parameter is available only for the `solve` test where the user can specify the problem size (see below).
## Available tests
- `solve [n]`:
  Solve the problem printing informations about dofs, overall CPU time, and others at each multigrid cycle. At the end it provides the L2 and H1 errors to check the correctness of the solver. The optional parameter `n` is related to the number of initial refinements to perform on the mesh before starting the first cycle (default is `n=5`). The initial refinements that are actually performed on the mesh are computed as `n-dim` where `dim` is the dimension of the space in which the problem is defined.
- `convergence`: Execute a convergence test solving with our multigrid matrix-free solver few times the same problem halving the cell size at each iteration. It prints only few summary informations for each step, and at the end it shows the convergence table and saves into the `build/output/convergence_mf.csv` the same data.
- `dimension`: TODO
- `polynomial`: TODO

  ## Code documentation
  The complete documentation of this project code can be found [here](https://lukethewalker.github.io/matrix-free-solver/docs/html/namespaces.html).
