#pragma once

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/data_out_base.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>

#include <deal.II/lac/generic_linear_algebra.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

// The following files are used to assemble the error estimator like in step-12:
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/meshworker/mesh_loop.h>

#include <filesystem>

const std::string output_dir = "./output_mg/";

#include "problem_data.hpp"

using namespace dealii;

namespace DTR_mg
{
  /**
   * @brief The main class of the program.
   *
   * This class manages the setup, assembly, solution, and output for the multigrid solver 
   * applied to a differential problem using the deal.II library.
   *
   * @tparam dim The spatial dimension of the problem.
   */
  template <int dim>
  class DTRProblem
  {
  public:
    /**
     * @brief Constructor.
     *
     * @param degree The polynomial degree of the finite element basis functions.
     */
    DTRProblem(unsigned int degree);

    /**
     * @brief Constructor with a file stream for timing information.
     *
     * @param degree The polynomial degree of the finite element basis functions.
     * @param dimension_time_file Output file stream for logging timing information.
     */
    DTRProblem(unsigned int degree, std::ofstream& dimension_time_file);

    /**
     * @brief Run the problem.
     *
     * This function executes the entire process from setup to output.
     *
     * @param n_initial_refinements Number of initial mesh refinements.
     * @param n_cycles Number of cycles for the multigrid method.
     */
    void run(unsigned int n_initial_refinements = 3, unsigned int n_cycles = 9);

  private:
    using MatrixType = LinearAlgebraTrilinos::MPI::SparseMatrix;
    using VectorType = LinearAlgebraTrilinos::MPI::Vector;

    /**
     * @brief Setup the system.
     *
     * This function initializes the triangulation, DoF handler, and system matrices.
     */
    void setup_system();

    /**
     * @brief Setup the multigrid method.
     *
     * This function initializes the multigrid hierarchy and associated matrices.
     */
    void setup_multigrid();

    /**
     * @brief Assemble the system matrix and right-hand side vector.
     */
    void assemble_system();

    /**
     * @brief Assemble the multigrid levels.
     */
    void assemble_multigrid();

    /**
     * @brief Solve the linear system.
     */
    void solve();

    /**
     * @brief Output the results to a file.
     *
     * @param cycle The current cycle number in the multigrid method.
     */
    void output_results(const unsigned int cycle);
    /**
     * @brief Communicator for parallel execution. 
     */
    MPI_Comm mpi_communicator;

    /**
     * @brief A conditional output stream for parallel execution.
     */
    ConditionalOStream pcout;

    /**
     * @brief A conditional output stream for logging timing details.
     */
    ConditionalOStream time_details;

    /**
     * @brief Time taken for setup.
     */
    double setup_time;

    /**
     * @brief Parallel distributed triangulation.
     */
    parallel::distributed::Triangulation<dim> triangulation;

    /**
     * @brief Mapping for the finite element space.
     */
    const MappingQ1<dim> mapping;

    /**
     * @brief Finite element space.
     */
    const FE_Q<dim> fe;

    /**
     * @brief DoF handler for managing degrees of freedom.
     */
    DoFHandler<dim> dof_handler;

    /**
     * @brief Index set for locally owned DoFs.
     */
    IndexSet locally_owned_dofs;

    /**
     * @brief Index set for locally relevant DoFs.
     */
    IndexSet locally_relevant_dofs;

    /**
     * @brief Affine constraints for the linear system.
     */
    AffineConstraints<double> constraints;

    /**
     * @brief System matrix.
     */
    MatrixType system_matrix;

    /**
     * @brief Solution vector.
     */
    VectorType solution;

    /**
     * @brief Right-hand side vector.
     */
    VectorType right_hand_side;

    /**
     * @brief Vector storing estimated error per cell.
     */
    Vector<double> estimated_error_square_per_cell;

    /**
     * @brief Multigrid level matrices.
     */
    MGLevelObject<MatrixType> mg_matrix;

    /**
     * @brief Multigrid interface matrices.
     */
    MGLevelObject<MatrixType> mg_interface_in;

    /**
     * @brief Multigrid constrained DoFs.
     */
    MGConstrainedDoFs mg_constrained_dofs;

    /**
     * @brief boolean controlling whether to output result
     */
    bool verbose;
    
    /**
     * @brief Diffusion coefficient.
     */
    problem_data::DiffusionCoefficient<dim> diffusion_coefficient;

    /**
     * @brief Transport coefficient.
     */
    problem_data::TransportCoefficient<dim> transport_coefficient;

    /**
     * @brief Reaction coefficient.
     */
    problem_data::ReactionCoefficient<dim> reaction_coefficient;

    /**
     * @brief Forcing term.
     */
    problem_data::ForcingTerm<dim> forcing_term;

    /**
     * @brief The object representing the Dirichlet boundary condition at the left boundary.
     */
    problem_data::DirichletBC1<dim> dirichletBC1;

    /**
     * @brief The object representing the Dirichlet boundary condition at the bottom boundary.
     */
    problem_data::DirichletBC2<dim> dirichletBC2;

    /**
     * @brief The object representing the Neumann boundary condition at the right boundary.
     */
    problem_data::NeumannBC1<dim> neumannBC1;

    /**
     * @brief The object representing the Neumann boundary condition at the top boundary.
     */
    problem_data::NeumannBC2<dim> neumannBC2;
  };
}

#include "DTR_mg.cpp"
