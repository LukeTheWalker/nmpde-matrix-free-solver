# Matrix-free solver for the Advection-Diffusion-Reaction problem
This project implements a matrix-free solver for advection-diffusion-reaction (ADR) problems. It then compare the performance of the matrix-free solver with that of a matrix-based solver, in terms of computational efficiency, parallel scalability and complexity of the implementation.
The matrixfree solver implements a Geometric MultiGrid preconditioner with Chebyshev iteration being a good fit for a matrixfree implementation since it only needs the computation of the diagonal.

This solver is based on the [deal.II library](https://github.com/dealii/dealii), mainly but not only on its [step-37](https://www.dealii.org/current/doxygen/deal.II/step_37.html) tutorial for the matrixfree solver and on other tutorials for the implementation of the matrixbased comparison.

The report for this project can be found [here](https://github.com/LukeTheWalker/nmpde-matrix-free-solver/blob/main/report/pdf/matrixfree_report.pdf).

## Required software
- dealii - version 9.5.1 (or newer)
- CMake

## Compile and run
- Clone the repository
- Create a build directory
  ```
  mkdir build && cd build
  ```
- Run cmake and make
  ```
  cmake ..
  make
  ```
- Execute the matrixfree and matrixbased versions, more details on the available tests that can be run is provided below
  ```
  [mpirun -n N] ./<solver_type> <test> [parameters]
  ```
  The integer value `N` is the requested number of MPI processes to be spawned. \
  Output .vtu files can be found in `/build/output_##` folder where `## = [mf | mb | mg]` respectively for the matrixfree with GMG, basic matrixbased w/o multigrid, and matrixbased with GMG. \
  Available `<test> = [solve | convergence | dimension | polynomial]` are described below. \
  An optional parameter is available only for the `solve` test where the user can specify the problem size (see below).
- Optional: run the tests on the MOX cluster (at Politecnico di Milano) using the submission script. \
  We assume to be in the hpc machine and to have loaded the last toolchain and `dealii` module in the `~/.bashrc` file.
  ```
  mkdir build && cd build
  cmake ..
  make
  qsub run_simulation.sub
  ```
  The `run_simulation_multi.sub` is also available to run on 2 nodes, but due to toolchain issues it does not run succesfully on the MOX cluster.

  
## Available tests
- `solve [n]`: Solve the problem printing informations about dofs, overall CPU time, and others at each multigrid cycle. At the end it provides the L2 and H1 errors to check the correctness of the solver. The optional parameter `n` is related to the number of initial refinements to perform on the mesh before starting the first cycle (default is `n=5`). The initial refinements that are actually performed on the mesh are computed as `n-dim` where `dim` is the dimension of the space in which the problem is defined.
- `convergence`: Execute a convergence test solving with our multigrid matrix-free solver few times the same problem halving the cell size at each iteration. It prints only few summary informations for each step, and at the end it shows the convergence table and saves into the `convergence_##.csv` the same data.
- `dimension`: Runs the solver with increasing problem sizes and fixed FE polynomial degree. Starting from a predefined size, at each iteration the mesh is refined and the problem solved. In a 2D space at each iteration the number of dofs is 4 times more. The timing data is then stored in the `dimension_##.csv` file.
- `polynomial`: As the above one but changes the polynomial degree instead of the problem size.

  ## Code documentation
  The complete documentation of this project code can be found [here](https://lukethewalker.github.io/matrix-free-solver/docs/html/namespaces.html).
