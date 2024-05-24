#include <deal.II/base/convergence_table.h>
#include "DTR.hpp"

using namespace dealii;

void solve_problem();
void convergence_study();
void dimension_time_study();
void polynomial_degree_study();

// Main function with convergence table.
int main(int argc, char * argv[])
{
  try
  {
    // Create the output directory if it does not exist
    std::filesystem::create_directory(output_dir);

    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

    if (argc < 2)
    {
      std::cerr << "Usage: " << argv[0] << " [ solve | convergence | dimension ]" << std::endl;
      return 1;
    }
    if (std::string(argv[1]) == "solve")
      solve_problem();
    else if (std::string(argv[1]) == "convergence")
      convergence_study();
    else if (std::string(argv[1]) == "dimension")
      dimension_time_study();
    else if (std::string(argv[1]) == "polynomial")
      		polynomial_degree_study();

    else
    {
      std::cerr << "Usage: " << argv[0] << " [ solve | convergence | dimension ]" << std::endl;
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

/**
 * @brief Solve the ADR problem.
 * It prints all the verbose information to the standard output, including timings, solver information, and errors.
 */
void solve_problem()
{
  //const std::string  mesh_filename = "../mesh/mesh-square-h0.012500.msh";
  const unsigned int degree        = 1;

  DTR problem(degree);

  problem.setup();
  problem.assemble();
  problem.solve();
  problem.output();

  const double error_L2 = problem.compute_error(VectorTools::L2_norm);
  const double error_H1 = problem.compute_error(VectorTools::H1_norm);

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    std::cout << "L2 error: " << error_L2 << std::endl;
    std::cout << "H1 error: " << error_H1 << std::endl;
  }
}

/**
 * @brief Execute a convergence study for the ADR problem, extracting the L2 and H1 errors and the convergence rates.
 * It writes the convergence table both to the ./output_mb/convergence_mb.csv file and to the standard output.
 */
void convergence_study()
{
  ConvergenceTable table;
  std::ofstream convergence_file;

  const unsigned int             degree = 2;
  const unsigned int n_initial_refinements = 4;

   if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    convergence_file.open(output_dir + "convergence_mb.csv");
    convergence_file << "h,eL2,eH1" << std::endl;
  }

for (unsigned int cycle = 0; cycle < 5; ++cycle)
  {
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::cout << "Cycle " << cycle << std::endl;
    }

    DTR problem(degree);

    problem.setup(n_initial_refinements + cycle);
    problem.assemble();
    problem.solve();

    const double error_L2 = problem.compute_error(VectorTools::L2_norm);
    const double error_H1 = problem.compute_error(VectorTools::H1_norm);

    table.add_value("cycle", cycle);
    table.add_value("L2", error_L2);
    table.add_value("H1", error_H1);

    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {

      convergence_file << cycle << "," << error_L2 << "," << error_H1 << std::endl;
      std::cout << "\tFE degree:       " << degree << std::endl;
      std::cout << "\tcycle:               " << cycle << std::endl;
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

void dimension_time_study()
{
  std::ofstream file_out;
  const unsigned int degree = 2;
  const unsigned int n_initial_refinements = 4;


  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {

    const unsigned int n_vect_doubles = VectorizedArray<double>::size();
    const unsigned int n_vect_bits = 8 * sizeof(double) * n_vect_doubles;
    const unsigned int n_ranks = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

    file_out.open(output_dir + "dimension_time_mb_" + std::to_string(n_ranks) + ".csv");

    file_out << "# Vectorization: " << n_vect_doubles
          << " doubles = " << n_vect_bits << " bits ("
          << Utilities::System::get_current_vectorization_level() << ')'
          << std::endl;
    file_out << "# Processes:     " << n_ranks << std::endl;
    file_out << "# Threads:       " << 1 << std::endl;

    file_out << "n_dofs,steup+assemble,solve,iterations" << std::endl;
  }

  for (unsigned int cycle = 0; cycle < 7; ++cycle)
  {
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::cout << "Cycle " << cycle << std::endl;
    }

    DTR problem(degree, file_out);

    problem.setup(n_initial_refinements + cycle);
    problem.assemble();
    problem.solve();
    //problem.output();

  }

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    file_out.close();
  }
}

void polynomial_degree_study()
{
  std::ofstream file_out;

  constexpr int degree[] = {1, 2/*, 3, 4, 5, 7, 8, 10*/}; // fill always with 8 integers
  const int initial_refinements = 9;

  // Open file and add comments about processes and threads
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    const unsigned int n_vect_doubles = VectorizedArray<double>::size();
    const unsigned int n_vect_bits = 8 * sizeof(double) * n_vect_doubles;
    const unsigned int n_ranks = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

    file_out.open(output_dir + "polynomial_degree_mb_" + std::to_string(n_ranks) + ".csv");

    file_out << "# Vectorization: " << n_vect_doubles
          << " doubles = " << n_vect_bits << " bits ("
          << Utilities::System::get_current_vectorization_level() << ')'
          << std::endl;
    file_out << "# Processes:     " << n_ranks << std::endl;
    file_out << "# Threads:       " << 1 << std::endl;

    file_out << "degree,dofs,setup+assemble,solve,iterations" << std::endl;
  }

  for(int d : degree)
  {
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::cout << "Starting with degree " << d << std::endl;
      file_out << d << ",";
    }
    DTR problem(d, file_out);

    problem.setup(initial_refinements);
    problem.assemble();
    problem.solve();
  }
}