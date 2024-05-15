#include "DTR_mg.hpp"

using namespace DTR_mg;
using namespace dealii;

int main(int argc, char *argv[])
{
	// Create the output directory if it does not exist
    std::filesystem::create_directory(output_dir);

	Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  const unsigned int degree = 2;
	try
	{
		DTRProblem<2> test(degree);
		test.run();
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