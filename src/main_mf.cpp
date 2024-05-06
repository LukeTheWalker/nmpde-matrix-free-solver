#include "DTR_mf.hpp"

int main(int argc, char *argv[])
{

  try
  {
    using namespace DTR_mf;

    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

    DTRProblem<dimension> laplace_problem;
    laplace_problem.run();

    const double error_L2 = laplace_problem.compute_error(VectorTools::L2_norm);
    const double error_H1 = laplace_problem.compute_error(VectorTools::H1_norm);

    std::cout << "L2 error: " << error_L2 << std::endl;
    std::cout << "H1 error: " << error_H1 << std::endl;

  }
  catch (std::exception &exc)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }
  catch (...)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }

  return 0;
}