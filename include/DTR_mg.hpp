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

  // This is the main class of the program.
  template <int dim>
  class DTRProblem
  {
  public:
    DTRProblem(unsigned int degree, bool verbose = true);
    DTRProblem(unsigned int degree, bool verbose, std::ofstream& dimension_time_file);
    void run(unsigned int n_initial_refinements = 3, unsigned int n_cycles = 9);

  private:
    using MatrixType = LinearAlgebraTrilinos::MPI::SparseMatrix;
    using VectorType = LinearAlgebraTrilinos::MPI::Vector;

    void setup_system();
    void setup_multigrid();
    void assemble_system();
    void assemble_multigrid();
    void solve();
    void output_results(const unsigned int cycle);

    MPI_Comm mpi_communicator;
    ConditionalOStream pcout;
    ConditionalOStream time_details;
    double setup_time;


    // p4est triangulation
    parallel::distributed::Triangulation<dim> triangulation;
    const MappingQ1<dim> mapping;
    const FE_Q<dim> fe;

    DoFHandler<dim> dof_handler;

    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;
    AffineConstraints<double> constraints;

    MatrixType system_matrix;
    VectorType solution;
    VectorType right_hand_side;
    Vector<double> estimated_error_square_per_cell;

    MGLevelObject<MatrixType> mg_matrix;
    MGLevelObject<MatrixType> mg_interface_in;
    MGConstrainedDoFs mg_constrained_dofs;

    bool verbose;

    // Coefficients and forcing term
    problem_data::DiffusionCoefficient<dim> diffusion_coefficient;
    problem_data::TransportCoefficient<dim> transport_coefficient;
    problem_data::ReactionCoefficient<dim> reaction_coefficient;
    problem_data::ForcingTerm<dim> forcing_term;

    problem_data::DirichletBC1<dim> dirichletBC1;
    problem_data::DirichletBC2<dim> dirichletBC2;
    problem_data::NeumannBC1<dim> neumannBC1;
    problem_data::NeumannBC2<dim> neumannBC2;
  };

}

#include "DTR_mg.cpp"
