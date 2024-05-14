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

using namespace dealii;

namespace DTR_mg
{

  const char bcs[4] = {'Z', 'N', 'Z', 'N'};

  // This is the main class of the program.
  template <int dim>
  class DTRProblem
  {
  public:
    class DiffusionCoefficient : public Function<dim>
    {
    public:
      virtual double
      value(const Point<dim> & /*p*/,
            const unsigned int /*component*/ = 0) const override
      {
        // EXAM: DIFFUSION COEFFICIENT
        return 1.;
      }
    };

    // Transport coefficient.
    class TransportCoefficient : public Function<dim>
    {
    public:
      virtual void
      vector_value(const Point<dim> & /*p*/,
                   Vector<double> &values) const override
      {
        values[0] = 1.;
        values[1] = 1.;
      }

      virtual double
      value(const Point<dim> & /*p*/,
            const unsigned int component = 0) const override
      {
        if (component == 0)
          return 1.;
        else
          return 1.;
      }
    };

    // Reaction coefficient.
    class ReactionCoefficient : public Function<dim>
    {
    public:
      virtual double
      value(const Point<dim> & /*p*/,
            const unsigned int /*component*/ = 0) const override
      {
        return 1.;
      }
    };

    // Forcing term.
    class ForcingTerm : public Function<dim>
    {
    public:
      virtual double
      value(const Point<dim> &p,
            const unsigned int /*component*/ = 0) const override
      {
        return (std::exp(p[0]) - 1.) * (std::exp(p[1]) - 1.);
      }
    };

    // Dirichlet boundary conditions.
    class DirichletBC : public Function<dim>
    {
    public:
      virtual double
      value(const Point<dim> & /*p*/,
            const unsigned int /*component*/ = 0) const override
      {
        return 0.;
      }
    };

    class NeumannBC1 : public Function<dim>
    {
    public:
      virtual double
      value(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
      {
        return std::exp(1.) * (std::exp(p[1]) - 1.);
      }
    };

    class NeumannBC2 : public Function<dim>
    {
    public:
      virtual double
      value(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
      {
        return std::exp(1.) * (std::exp(p[0]) - 1.);
      }
    };

    DTRProblem(unsigned int degree);
    void run();

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

    // Coefficients and forcing term
    DiffusionCoefficient diffusion_coefficient;
    ReactionCoefficient reaction_coefficient;
    TransportCoefficient transport_coefficient;
    ForcingTerm forcing_term;

    // Dirichlet boundary conditions
    DirichletBC dirichletBC;

    // Neumann boundary conditions
    NeumannBC1 neumannBC1;
    NeumannBC2 neumannBC2;

    TimerOutput computing_timer;
  };

}

#include "DTR_mg.cpp"
