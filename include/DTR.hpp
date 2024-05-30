#ifndef DTR_HPP
#define DTR_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/timer.h>


#include <filesystem>
#include <fstream>
#include <iostream>

#include "problem_data.hpp"

static const std::string output_dir = "./output_mb/";

using namespace dealii;

/**
 * @class DTR
 * @brief Class managing the differential problem.
 */
class DTR
{
public:
  /// @brief Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 2;

  /**
   * @brief Constructor.
   * 
   * @param r_ Polynomial degree.
   */
  DTR(const unsigned int &r_)
    : r(r_)
    , mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , mesh(MPI_COMM_WORLD)
    , pcout(std::cout, true && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    , time_details(std::cout, false && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    , setup_time(0.)
  {}

  /**
   * @brief Constructor used to print the time details to a file.
   * 
   * @param r_ Polynomial degree.
   * @param dimension_time_file Output file stream for time details.
   */  
  DTR(const unsigned int &r_, std::ofstream& dimension_time_file)
    : r(r_)
    , mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , mesh(MPI_COMM_WORLD)
    , pcout(std::cout, true && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    , time_details(dimension_time_file, true && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    , setup_time(0.)
  {}

  /**
   * @brief Initialization.
   * 
   * Sets up the problem by initializing the mesh, finite element space, DoF handler,
   * and linear system.
   * 
   * @param n_initial_refinements Number of initial refinements.
   */
  void setup(unsigned int n_initial_refinements = 8);

  /// @brief System assembly.
  void assemble();

  /// @brief System solution.
  void solve();

  /// @brief Output.
  void output() const;

  /**
   * @brief Compute the error.
   * 
   * @param norm_type Norm type.
   * @return double Computed error.
   */
  double compute_error(const VectorTools::NormType &norm_type) const;

protected:
  /// @brief Path to the mesh file.
  const std::string mesh_file_name;

  /// @brief Polynomial degree.
  const unsigned int r;

  /// @brief Number of MPI processes.
  const unsigned int mpi_size;

  /// @brief This MPI process.
  const unsigned int mpi_rank;

  /// @brief Diffusion coefficient.
  problem_data::DiffusionCoefficient<dim> diffusion_coefficient;

  /// @brief Reaction coefficient.
  problem_data::ReactionCoefficient<dim> reaction_coefficient;

  /// @brief Transport coefficient.
  problem_data::TransportCoefficient<dim> transport_coefficient;

  /// @brief Forcing term.
  problem_data::ForcingTerm<dim> forcing_term;

  /// @brief boundary conditions.
  problem_data::DirichletBC1<dim> dirichletBC1;
  problem_data::DirichletBC2<dim> dirichletBC2;
  problem_data::NeumannBC1<dim> neumannBC1;
  problem_data::NeumannBC2<dim> neumannBC2;

  /**
   * @brief Triangulation
   * 
   * The parallel::fullydistributed::Triangulation class manages
   * a triangulation that is completely distributed (i.e. each process only
   * knows about the elements it owns and its ghost elements).
   */
  parallel::fullydistributed::Triangulation<dim> mesh;

  /// @brief Finite element space.
  // We use a unique_ptr here so that we can choose the type and degree of the
  // finite elements at runtime (the degree is a constructor parameter). The
  // class FiniteElement<dim> is an abstract class from which all types of
  // finite elements implemented by deal.ii inherit.
  std::unique_ptr<FiniteElement<dim>> fe;

  /// @brief Quadrature formula.
  // We use a unique_ptr here so that we can choose the type and order of the
  // quadrature formula at runtime (the order is a constructor parameter).
  std::unique_ptr<Quadrature<dim>> quadrature;

  /// @brief Quadrature formula for the boundary.
  std::unique_ptr<Quadrature<dim - 1>> quadrature_boundary;

  /// @brief DoF handler.
  DoFHandler<dim> dof_handler;

  /// @brief System matrix.
  TrilinosWrappers::SparseMatrix system_matrix;

  /// @brief System right-hand side.
  TrilinosWrappers::MPI::Vector system_rhs;

  /// @brief System solution.
  TrilinosWrappers::MPI::Vector solution;

  /// @brief Parallel output stream.
  ConditionalOStream pcout;

  /// @brief Time details parallel output stream.
  ConditionalOStream time_details;


  /// @brief DoFs owned by current process.
  IndexSet locally_owned_dofs;

  /// @brief duration of the setup phase
  double setup_time;
};

#endif