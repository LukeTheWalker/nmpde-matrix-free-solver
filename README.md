## Required software
- dealii version 9.5.1 (no more recent versions)
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
  ./matrixfree
  ./matrixbased
  ```
- Output .vtu files can be found in `/build/output` folder for the matrixfree algorithm and in `/build` folder for the matrixbased one.
