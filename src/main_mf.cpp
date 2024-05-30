#include "DTR_mf.hpp"
#include <deal.II/base/convergence_table.h>
#include <filesystem>

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

/**
 * @brief Evaluate the solver performances for different number of dofs.
 * Metrics are stored in the dimension_time_mf.csv file in the usual output directory, such as the number of dofs and the solver time.
 */
void dimension_time_study();

/**
 * @brief Evaluate the solver performances for different polynomial degrees.
 * Many metrics are stored in the polynomial_degree_mf.csv file in the usual output directory, such as the number of dofs,
 * the number of iterations, and the solver time. For a predefined problem size.
 */
void polynomial_degree_study();

int main(int argc, char *argv[])
{
  try
  {
    // Create the output directory if it does not exist
    std::filesystem::create_directory(output_dir);

    // Initialize the MPI environment also with multithreading
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

    if (argc == 2 && std::string(argv[1]) == "solve")
      solve_problem();
    else if (argc == 3 && std::string(argv[1]) == "solve")
      solve_problem(atoi(argv[2]));
    else if (argc == 2 && std::string(argv[1]) == "convergence")
      convergence_study();
    else if (std::string(argv[1]) == "dimension")
      dimension_time_study();
    else if (std::string(argv[1]) == "polynomial")
      polynomial_degree_study();
    else
    {
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
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
  DTRProblem<dim> problem;
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
    convergence_file.open(output_dir + "convergence_mf.csv");
    convergence_file << "cells,eL2,eH1" << std::endl;
  }

  for (unsigned int refinements = 6; refinements < 11; ++refinements)
  {
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "Starting with " << refinements - dim << " initial refinements...\n";

    DTRProblem<dim> problem(false);
    problem.run(refinements, dim + 1);

    const double error_L2 = problem.compute_error(VectorTools::L2_norm);
    const double error_H1 = problem.compute_error(VectorTools::H1_norm);

    unsigned int cells = problem.get_cells();
    Utilities::MPI::sum<int>(cells, MPI_COMM_WORLD);

    table.add_value("cells", cells);
    table.add_value("L2", error_L2);
    table.add_value("H1", error_H1);

    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      convergence_file << cells << "," << error_L2 << "," << error_H1 << std::endl;
      std::cout << "\tFE degree:       " << problem.get_fe_degree() << std::endl;
      std::cout << "\tNumber of cells: " << cells << std::endl;
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
    table.set_precision("L2", 6);
    table.set_precision("H1", 6);
    table.write_text(std::cout);
    convergence_file.close();
  }
}

void dimension_time_study()
{
  std::ofstream file_out;
  unsigned int refinements = 8;

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    const unsigned int n_vect_doubles = VectorizedArray<double>::size();
    const unsigned int n_vect_bits = 8 * sizeof(double) * n_vect_doubles;
    const unsigned int n_ranks = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

    file_out.open(output_dir + "dimension_time_mf_" + std::to_string(n_ranks) + ".csv");

    file_out << "# Vectorization: " << n_vect_doubles
          << " doubles = " << n_vect_bits << " bits ("
          << Utilities::System::get_current_vectorization_level() << ')'
          << std::endl;
    file_out << "# Processes:     " << n_ranks << std::endl;
    file_out << "# Threads:       " << MultithreadInfo::n_threads() << std::endl;
    file_out << "n_dofs,setup+assemble,solve,iterations" << std::endl;

    std::cout << "Starting test with " << refinements << " initial refinements...\n";
  }

  DTRProblem<dim> problem(file_out, false);
  problem.run(refinements, 8);

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    file_out.close();
  }
}

void polynomial_degree_study()
{
  std::ofstream file_out;

  constexpr int degree[] = {1, 2, 3, 4, 5, 7, 8, 10}; // fill always with 8 integers
  const int initial_refinements = 11;

  // Open file and add comments about processes and threads
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    const unsigned int n_vect_doubles = VectorizedArray<double>::size();
    const unsigned int n_vect_bits = 8 * sizeof(double) * n_vect_doubles;
    const unsigned int n_ranks = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

    file_out.open(output_dir + "polynomial_degree_mf_" + std::to_string(n_ranks) + ".csv");

    file_out << "# Vectorization: " << n_vect_doubles
          << " doubles = " << n_vect_bits << " bits ("
          << Utilities::System::get_current_vectorization_level() << ')'
          << std::endl;
    file_out << "# Processes:     " << n_ranks << std::endl;
    file_out << "# Threads:       " << MultithreadInfo::n_threads() << std::endl;

    file_out << "degree,dofs,setup+assemble,solve,iterations" << std::endl;
  }

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    std::cout << "Starting with degree " << degree[0] << std::endl;
    file_out << degree[0] << ",";
  }
  {
    DTRProblem<2, degree[0]> problem(file_out, false);
    problem.run(initial_refinements, dim + 1);
  }

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    std::cout << "Starting with degree " << degree[1] << std::endl;
    file_out << degree[1] << ",";
  }
  {
    DTRProblem<2, degree[1]> problem(file_out, false);
    problem.run(initial_refinements, dim + 1);
  }

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    std::cout << "Starting with degree " << degree[2] << std::endl;
    file_out << degree[2] << ",";
  }
  {
    DTRProblem<2, degree[2]> problem(file_out, false);
    problem.run(initial_refinements, dim + 1);
  }
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    std::cout << "Starting with degree " << degree[3] << std::endl;
    file_out << degree[3] << ",";
  }
  {
    DTRProblem<2, degree[3]> problem(file_out, false);
    problem.run(initial_refinements, dim + 1);
  }
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    std::cout << "Starting with degree " << degree[4] << std::endl;
    file_out << degree[4] << ",";
  }
  {
    DTRProblem<2, degree[4]> problem(file_out, false);
    problem.run(initial_refinements, dim + 1);
  }
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    std::cout << "Starting with degree " << degree[5] << std::endl;
    file_out << degree[5] << ",";
  }
  {
    DTRProblem<2, degree[5]> problem(file_out, false);
    problem.run(initial_refinements, dim + 1);
  }
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    std::cout << "Starting with degree " << degree[6] << std::endl;
    file_out << degree[6] << ",";
  }
  {
    DTRProblem<2, degree[6]> problem(file_out, false);
    problem.run(initial_refinements, dim + 1);
  }
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    std::cout << "Starting with degree " << degree[7] << std::endl;
    file_out << degree[7] << ",";
  }
  {
    DTRProblem<2, degree[7]> problem(file_out, false);
    problem.run(initial_refinements, dim + 1);
  }

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    file_out.close();
  }
}