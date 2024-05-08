#include "DTR_mf.hpp"
#include <deal.II/base/convergence_table.h>

using namespace dealii;
using namespace DTR_mf;

/**
 * @brief Solve the ADR problem.
 * It prints all the verbose information to the standard output, including timings, solver information, and errors.
 * @param initial_refinements Number of initial refinements to provide to the run method. Default is 5.
 */
void solve_problem(unsigned int initial_refinements = 5);
/**
 * @brief Execute a convergence study for the ADR problem, extracting the L2 and H1 errors and the convergence rates.
 * It writes the convergence table both to the /output/convergence_mf.csv file and to the standard output.
 */
void convergence_study();

int main(int argc, char *argv[])
{
  try
  {

    // Initialize the MPI environment also with multithreading
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

    if (argc == 2 && std::string(argv[1]) == "solve")
      solve_problem();
    else if (argc == 3 && std::string(argv[1]) == "solve")
      solve_problem(atoi(argv[2]));
    else if (argc == 2 && std::string(argv[1]) == "convergence")
      convergence_study();
    else
    {
      if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cerr << "Usage: " << argv[0] << " [ solve | convergence ] [optional arguments]" << std::endl;
      return 1;
    }

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

void solve_problem(unsigned int initial_refinements)
{
  DTRProblem<dimension> problem;
  problem.run(initial_refinements);

  const double error_L2 = problem.compute_error(VectorTools::L2_norm);
  const double error_H1 = problem.compute_error(VectorTools::H1_norm);

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    std::cout << "L2 error: " << error_L2 << std::endl;
    std::cout << "H1 error: " << error_H1 << std::endl;
  }
}

void convergence_study()
{
  ConvergenceTable table;
  std::ofstream convergence_file;

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    convergence_file.open("./output/convergence_mf.csv");
    convergence_file << "cells,eL2,eH1" << std::endl;
  }

  for (unsigned int refinements = 3; refinements < 7; ++refinements)
  {
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "Starting with " << refinements - dimension << " initial refinements...\n";

    DTRProblem<dimension> problem(false);
    problem.run(refinements);

    const double error_L2 = problem.compute_error(VectorTools::L2_norm);
    const double error_H1 = problem.compute_error(VectorTools::H1_norm);

    table.add_value("cells", problem.get_cells());
    table.add_value("L2", error_L2);
    table.add_value("H1", error_H1);

    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      convergence_file << problem.get_cells() << "," << error_L2 << "," << error_H1 << std::endl;
      std::cout << "\tFE degree:       " << problem.get_fe_degree() << std::endl;
      std::cout << "\tNumber of cells: " << problem.get_cells() << std::endl;
      std::cout << "\tNumber of dofs:  " << problem.get_dofs() << std::endl;
      std::cout << "\tL2 error:        " << error_L2 << std::endl;
      std::cout << "\tH1 error:        " << error_H1 << std::endl;
    }
  }

  table.evaluate_all_convergence_rates(ConvergenceTable::reduction_rate_log2);

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    table.set_scientific("L2", true);
    table.set_scientific("H1", true);
    table.set_precision("h", 6);
    table.set_precision("L2", 6);
    table.set_precision("H1", 6);
    table.write_text(std::cout);
    convergence_file.close();
  }
}