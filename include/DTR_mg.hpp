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
   * @class DTRProblem
   * @brief Main class of the program managing the differential problem using multigrid.
   * 
   * This class handles the setup, assembly, solution, and output of the differential problem
   * using multigrid.
   * 
   * @tparam dim Dimension of the problem.
   */
  template <int dim>
  class DTRProblem
  {
  public:
    /**
     * @brief Constructor.
     * 
     * @param degree Polynomial degree of the finite elements.
     */
    DTRProblem(unsigned int degree, bool verbose = true);
    
    /**
     * @brief Constructor with output file stream for time details.
     * 
     * @param degree Polynomial degree of the finite elements.
     * @param dimension_time_file Output file stream for time details.
     */
    DTRProblem(unsigned int degree, bool verbose, std::ofstream& dimension_time_file);
    
    /**
     * @brief Run the simulation starting from the initial refinements and for 
     * a number of cycles equal to n_cycles-dim.
     * 
     * @param n_initial_refinements Number of initial refinements.
     * @param n_cycles Number of cycles.
     */
    void run(unsigned int n_initial_refinements = 3, unsigned int n_cycles = 9);

  private:
    using MatrixType = LinearAlgebraTrilinos::MPI::SparseMatrix;
    using VectorType = LinearAlgebraTrilinos::MPI::Vector;

    /// @brief Setup the system.
    void setup_system();

    /// @brief Setup the multigrid hierarchy.
    void setup_multigrid();

    /// @brief Assemble the system matrix and right-hand side.
    void assemble_system();

    /// @brief Assemble the multigrid hierarchy.
    void assemble_multigrid();

    /// @brief Solve the system.
    void solve();

    /// @brief Output the results.
    void output_results(const unsigned int cycle);

    /// @brief MPI communicator.
    MPI_Comm mpi_communicator;

    /// @brief Parallel output stream.
    ConditionalOStream pcout;
    
    /// @brief Time details parallel output stream.
    ConditionalOStream time_details;

    /// @brief duration of the setup phase
    double setup_time;


    // p4est triangulation
    parallel::distributed::Triangulation<dim> triangulation;

    /// @brief Mapping.
    const MappingQ1<dim> mapping;

    /// @brief Finite element space.
    const FE_Q<dim> fe;

    /// @brief DoF handler.
    DoFHandler<dim> dof_handler;

    /// @brief DoFs owned by current process.
    IndexSet locally_owned_dofs;

    /// @brief DoFs relevant to current process.
    IndexSet locally_relevant_dofs;

    /// @brief Affine constraints.
    AffineConstraints<double> constraints;

    /// @brief System matrix.
    MatrixType system_matrix;

    /// @brief System solution.
    VectorType solution;

    /// @brief System right-hand side.
    VectorType right_hand_side;

    /// @brief Vectors to store error square estimator per cell.
    Vector<double> estimated_error_square_per_cell;

    /// @brief Matrix for different levels of the multigrid hierarchy.
    MGLevelObject<MatrixType> mg_matrix;

    /// @brief interface matrix coupling different levels.
    MGLevelObject<MatrixType> mg_interface_in;

    /// @brief constrained dofs at a level of multigrid solver.
    MGConstrainedDoFs mg_constrained_dofs;

    /// @brief boolean controlling whether to output result.
    bool verbose;

    /// @brief Diffusion coefficient.
    problem_data::DiffusionCoefficient<dim> diffusion_coefficient;

    /// @brief Transport coefficient.
    problem_data::TransportCoefficient<dim> transport_coefficient;

    /// @brief Reaction coefficient.
    problem_data::ReactionCoefficient<dim> reaction_coefficient;

    /// @brief Forcing term.
    problem_data::ForcingTerm<dim> forcing_term;

    /// @brief boundary conditions.
    problem_data::DirichletBC1<dim> dirichletBC1;
    problem_data::DirichletBC2<dim> dirichletBC2;
    problem_data::NeumannBC1<dim> neumannBC1;
    problem_data::NeumannBC2<dim> neumannBC2;
  };

}

#include "DTR_mg.cpp"
