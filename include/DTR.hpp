#ifndef DTR_HPP
#define DTR_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

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

#include <filesystem>
#include <fstream>
#include <iostream>

static const std::string output_dir = "./output_mb/";

using namespace dealii;

/**
 * Class managing the differential problem.
 */
class DTR
{
public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 2;

  // Diffusion coefficient.
  // In deal.ii, functions are implemented by deriving the dealii::Function
  // class, which provides an interface for the computation of function values
  // and their derivatives.
  class DiffusionCoefficient : public Function<dim>
  {
  public:
    // Constructor.
    DiffusionCoefficient()
    {}

    // Evaluation.
    virtual double
    value(const Point<dim> &/*p*/,
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
    // Constructor.
    TransportCoefficient()
    {}

   virtual void
    vector_value(const Point<dim> & /*p*/,
                 Vector<double> &values) const override
    {
      values[0] = 1.;
      values[1] = 0.;
    }

    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int component = 0) const override
    {
        if (component == 0)
            return 1.;
        else
            return 0.;
    }
  };

  // Reaction coefficient.
  class ReactionCoefficient : public Function<dim>
  {
  public:
    // Constructor.
    ReactionCoefficient()
    {}

    // Evaluation.
    virtual double
    value(const Point<dim> &/*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      // EXAM: REACTION COEFFICIENT
      return 1.;
    }
  };

  // Forcing term.
  class ForcingTerm : public Function<dim>
  {
  public:
    // Constructor.
    ForcingTerm()
    {}

    // Evaluation.
    virtual double
    value(const Point<dim> & p,
          const unsigned int /*component*/ = 0) const override
    {
        return 1. - 2. * exp(p[0]);
    }
  };

  class DirichletBC1 : public Function<dim>
  {
  public:
    virtual double value(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
    {
      return 2.*exp(p[1]) - 1.;
    }
  };

  class DirichletBC2 : public Function<dim>
  {
  public:
    virtual double value(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
    {
      return 2.*exp(p[0]) - 1.;
    }
  };

  class NeumannBC1 : public Function<dim>
  {
  public:
    virtual double value(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
    {
      return 2.*exp(p[0]) * (2.*exp(p[1]) - 1.);
    }
  };

  class NeumannBC2 : public Function<dim>
  {
  public:
    virtual double value(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
    {
      return 2.*exp(p[1]) * (2.*exp(p[0]) - 1.);
    }
  };

  // Exact solution.
  class ExactSolution : public Function<dim>
  {
  public:
    virtual double value(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
    {
      return (2.*exp(p[0]) - 1.)*(2.*exp(p[1]) - 1.);
    }

    virtual Tensor<1, dim> gradient(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
    {
      Tensor<1, dim> result;

      result[0] = 2.*exp(p[0]) * (2.*exp(p[1]) - 1.);
      result[1] = 2.*exp(p[1]) * (2.*exp(p[0]) - 1.);

      return result;
    }
  };

  // Constructor.
  DTR(const unsigned int &r_, std::ofstream& dimension_time_file)
    : r(r_)
    , mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , mesh(MPI_COMM_WORLD)
    , pcout(std::cout, true && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    , time_details(dimension_time_file, true && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    , setup_time(0.)
  {}

  DTR(const unsigned int &r_)
    : r(r_)
    , mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , mesh(MPI_COMM_WORLD)
    , pcout(std::cout, true && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    , time_details(std::cout, false && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    , setup_time(0.)
  {}

  // Initialization.
  void
  setup(unsigned int n_initial_refinements = 3);

  // System assembly.
  void
  assemble();

  // System solution.
  void
  solve();

  // Output.
  //void
  //output() const;

  // Compute the error.
  double
  compute_error(const VectorTools::NormType &norm_type) const;

protected:
  // Path to the mesh file.
  const std::string mesh_file_name;

  // Polynomial degree.
  const unsigned int r;

  // Number of MPI processes.
  const unsigned int mpi_size;

  // This MPI process.
  const unsigned int mpi_rank;

  // Diffusion coefficient.
  DiffusionCoefficient diffusion_coefficient;

  // Reaction coefficient.
  ReactionCoefficient reaction_coefficient;

  // Transport coefficient.
  TransportCoefficient transport_coefficient;

  // Forcing term.
  ForcingTerm forcing_term;

  // Dirichlet boundary conditions.
  DirichletBC1 dirichletBC1;
  DirichletBC2 dirichletBC2;
  NeumannBC1 neumannBC1;
  NeumannBC2 neumannBC2;

  // Triangulation. The parallel::fullydistributed::Triangulation class manages
  // a triangulation that is completely distributed (i.e. each process only
  // knows about the elements it owns and its ghost elements).
  parallel::fullydistributed::Triangulation<dim> mesh;

  // Finite element space.
  // We use a unique_ptr here so that we can choose the type and degree of the
  // finite elements at runtime (the degree is a constructor parameter). The
  // class FiniteElement<dim> is an abstract class from which all types of
  // finite elements implemented by deal.ii inherit.
  std::unique_ptr<FiniteElement<dim>> fe;

  // Quadrature formula.
  // We use a unique_ptr here so that we can choose the type and order of the
  // quadrature formula at runtime (the order is a constructor parameter).
  std::unique_ptr<Quadrature<dim>> quadrature;

  std::unique_ptr<Quadrature<dim - 1>> quadrature_boundary;

  // DoF handler.
  DoFHandler<dim> dof_handler;

  // System matrix.
  TrilinosWrappers::SparseMatrix system_matrix;

  // System right-hand side.
  TrilinosWrappers::MPI::Vector system_rhs;

  // System solution.
  TrilinosWrappers::MPI::Vector solution;

  // Parallel output stream.
  ConditionalOStream pcout;
  ConditionalOStream time_details;

  // DoFs owned by current process.
  IndexSet locally_owned_dofs;

  double setup_time;

};

#endif