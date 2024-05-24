#include "DTR_mg.hpp"

using namespace DTR_mg;
using namespace dealii;

/**
 * @brief Evaluate the solver performances for different number of dofs.
 * Metrics are stored in the dimension_time_mg.csv file in the usual output directory, such as the number of dofs and the solver time.
 */
void dimension_time_study();

/**
 * @brief Solve the ADR problem.
 * It prints all the verbose information to the standard output, including timings, solver information, and errors.
 * @param initial_refinements Number of initial refinements to provide to the run method. Default is 5.
 */
void solve_problem(unsigned int initial_refinements = 5);

/**
 * @brief Evaluate the solver performances for different polynomial degrees.
 * Many metrics are stored in the polynomial_degree_mf.csv file in the usual output directory, such as the number of dofs,
 * the number of iterations, and the solver time. For a predefined problem size.
 */
void polynomial_degree_study();


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
		else if (std::string(argv[1]) == "polynomial")
      		polynomial_degree_study();
		else
		{
			if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
				std::cerr << "Usage: " << argv[0] << " [ solve | dimension | polynomial]" << std::endl;
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
  std::ofstream file_out;
  const unsigned int degree = 2;
  unsigned int refinements = 3;

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
	const unsigned int n_vect_doubles = VectorizedArray<double>::size();
    const unsigned int n_vect_bits = 8 * sizeof(double) * n_vect_doubles;
    const unsigned int n_ranks = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

	file_out.open(output_dir + "dimension_time_mg_" + std::to_string(n_ranks) + ".csv");

    file_out << "# Vectorization: " << n_vect_doubles
          << " doubles = " << n_vect_bits << " bits ("
          << Utilities::System::get_current_vectorization_level() << ')'
          << std::endl;
    file_out << "# Processes:     " << n_ranks << std::endl;
    file_out << "# Threads:       " << MultithreadInfo::n_threads() << std::endl;

	file_out << "n_dofs,setup+assemble,solve,iterations" << std::endl;
  }

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << "Starting with " << refinements << " initial refinements...\n";

  DTRProblem<2> problem(degree, file_out);
  problem.run(refinements, 10);

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

    file_out.open(output_dir + "polynomial_degree_mg_" + std::to_string(n_ranks) + ".csv");

    file_out << "# Vectorization: " << n_vect_doubles
          << " doubles = " << n_vect_bits << " bits ("
          << Utilities::System::get_current_vectorization_level() << ')'
          << std::endl;
    file_out << "# Processes:     " << n_ranks << std::endl;
    file_out << "# Threads:       " << MultithreadInfo::n_threads() << std::endl;

    file_out << "degree,dofs,setup+assemble,solve,iterations" << std::endl;
  }

  for(int d : degree)
  {
	if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
	{
	  std::cout << "Starting with degree " << d << std::endl;
	  file_out << d << ",";
	}
	DTRProblem<2> problem(d, file_out);
	problem.run(initial_refinements, 2 + 1);
  }
}