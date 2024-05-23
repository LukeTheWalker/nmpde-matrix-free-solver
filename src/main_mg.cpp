#include "DTR_mg.hpp"

using namespace DTR_mg;
using namespace dealii;

/**
 * @brief Evaluate the solver performances for different number of dofs.
 * Metrics are stored in the dimension_time_mf.csv file in the usual output directory, such as the number of dofs and the solver time.
 */
void dimension_time_study();

/**
 * @brief Solve the ADR problem.
 * It prints all the verbose information to the standard output, including timings, solver information, and errors.
 * @param initial_refinements Number of initial refinements to provide to the run method. Default is 5.
 */
void solve_problem(unsigned int initial_refinements = 5);

int main(int argc, char *argv[])
{
	// Create the output directory if it does not exist
    std::filesystem::create_directory(output_dir);
	Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

	try
	{
		if (argc == 2 && std::string(argv[1]) == "solve")
			solve_problem();
		else if (argc == 3 && std::string(argv[1]) == "solve")
			solve_problem(atoi(argv[2]));
		else if (std::string(argv[1]) == "dimension")
			dimension_time_study();
		else
		{
			if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
				std::cerr << "Usage: " << argv[0] << " [ solve | dimension]" << std::endl;
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
		MPI_Abort(MPI_COMM_WORLD, 1);
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
		MPI_Abort(MPI_COMM_WORLD, 2);
		return 1;
	}

	return 0;
}

void solve_problem(unsigned int initial_refinements)
{
  const unsigned int degree = 2;

  DTRProblem<2> problem(degree);
  problem.run(initial_refinements);
}

void dimension_time_study()
{
  std::ofstream dimension_time_file;
  const unsigned int degree = 2;
  unsigned int refinements = 3;

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    dimension_time_file.open(output_dir + "dimension_time_mg.csv");
    dimension_time_file << "n_dofs,setup+assemble,solve,iterations" << std::endl;
  }

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << "Starting with " << refinements << " initial refinements...\n";

  DTRProblem<2> problem(degree, dimension_time_file);
  problem.run(refinements, 10);

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    dimension_time_file.close();
  }
}