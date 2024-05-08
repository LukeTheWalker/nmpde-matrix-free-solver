#pragma once

#include <deal.II/base/multithread_info.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

// Include the matrix-free headers
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include <filesystem>
#include <iostream>
#include <fstream>

namespace DTR_mf
{
  using namespace dealii;

  // To be efficient matrix-free implementation require knowledge of loop lengths at compile time
  const unsigned int degree_finite_element = 2;
  const unsigned int dim = 2;
  const char bcs[4] = {'Z', 'N', 'Z', 'N'}; // left, right, bottom, top

  template <int dim>
  class DiffusionCoefficient : public Function<dim>
  {
  public:
    virtual double value(const Point<dim> &p, const unsigned int component = 0) const override
    {
      return value<double>(p, component);
    }

    template <typename number>
    number value(const Point<dim, number> & /*p*/, const unsigned int /*component*/ = 0) const
    {
      return 1.;
    }
  };

  template <int dim>
  class TransportCoefficient : public Function<dim>
  {
  public:
    virtual void vector_value(const Point<dim> &p, Vector<double> &values) const override
    {
      return vector_value<double>(p, values);
    }

    template <typename number>
    void vector_value(const Point<dim, number> & /*p*/, Vector<number> &values) const
    {
      values[0] = 1.;
      values[1] = 1.;
    }

    virtual double value(const Point<dim> &p, const unsigned int component = 0) const override
    {
      return value<double>(p, component);
    }

    template <typename number>
    number value(const Point<dim, number> & /*p*/, const unsigned int component = 0) const
    {
      if (component == 0)
        return 1.;
      else
        return 1.;
    }

    template <typename number>
    void tensor_value(const Point<dim, number> &/*p*/, Tensor<1, dim, number> &values) const
    {
      values[0] = 1.;
      values[1] = 1.;
    }
  };

  template <int dim>
  class ReactionCoefficient : public Function<dim>
  {
  public:
    virtual double value(const Point<dim> &p, const unsigned int component = 0) const override
    {
      return value<double>(p, component);
    }

    template <typename number>
    number value(const Point<dim, number> & /*p*/, const unsigned int /*component*/ = 0) const
    {
      return 1.;
    }
  };

  template <int dim>
  class ForcingTerm : public Function<dim>
  {
  public:
    virtual double value(const Point<dim> &p, const unsigned int component = 0) const override
    {
      return value<double>(p, component);
    }

    template <typename number>
    number value(const Point<dim, number> &p, const unsigned int /*component*/ = 0) const
    {
      return (exp(p[0]) - number(1.)) * (exp(p[1]) - number(1.));
    }
  };

  // Dirichlet boundary conditions.
  class DirichletBC : public Function<dim>
  {
  public:
    virtual double value(const Point<dim> &p, const unsigned int component = 0) const override
    {
      return value<double>(p, component);
    }
    template <typename number>
    double value(const Point<dim, number> & /*p*/, const unsigned int /*component*/ = 0) const
    {
      return 0.;
    }
  };

  class NeumannBC1 : public Function<dim>
  {
  public:
    virtual double value(const Point<dim> &p, const unsigned int component = 0) const override
    {
      return value<double>(p, component);
    }

    template <typename number>
    number value(const Point<dim, number> &p, const unsigned int /*component*/ = 0) const
    {
      AssertThrow(p[0] == number(1.), ExcInternalError());
      return number(exp(1.)) * (exp(p[1]) - number(1.));
    }
  };

  class NeumannBC2 : public Function<dim>
  {
  public:
    virtual double value(const Point<dim> &p, const unsigned int component = 0) const override
    {
      return value<double>(p, component);
    }

    template <typename number>
    number value(const Point<dim, number> &p, const unsigned int /*component*/ = 0) const
    {
      AssertThrow(p[1] == number(1.), ExcInternalError());
      return number(exp(1.)) * (exp(p[0]) - number(1.));
    }
  };

  // Exact solution.
  class ExactSolution : public Function<dim>
  {
  public:
    // Evaluation.
    virtual double value(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
    {
      return (std::exp(p[0]) - 1.) * (std::exp(p[1]) - 1.);
    }

    // Gradient evaluation.
    virtual Tensor<1, dim> gradient(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
    {
      Tensor<1, dim> result;

      result[0] = std::exp(p[0]) * (std::exp(p[1]) - 1.);
      result[1] = std::exp(p[1]) * (std::exp(p[0]) - 1.);

      return result;
    }
  };

  // The following class implements the DTR linear operation that is needed at each
  // iteration of the linear solver. The fe_degree template argument is provided to the
  // FEEvaluation class that needs it for efficiency
  template <int dim, int fe_degree, typename number>
  class DTROperation
      : public MatrixFreeOperators::
            Base<dim, LinearAlgebra::distributed::Vector<number>>
  {
  public:
    using value_type = number;

    DTROperation();

    void clear() override;

    void evaluate_coefficients(const DiffusionCoefficient<dim> &diffusion_function,
                               const TransportCoefficient<dim> &transport_function,
                               const ReactionCoefficient<dim> &reaction_function,
                               const ForcingTerm<dim> &forcing_term_function);

    virtual void compute_diagonal() override;

    const Table<2, VectorizedArray<number>> &get_diffusion_coefficient() const
    {
      return diffusion_coefficient;
    }

    const Table<2, Tensor<1, dim, VectorizedArray<number>>> &get_transport_coefficient() const
    {
      return transport_coefficient;
    }

    const Table<2, VectorizedArray<number>> &get_reaction_coefficient() const
    {
      return reaction_coefficient;
    }

    const Table<2, VectorizedArray<number>> &get_forcing_term_coefficient() const
    {
      return forcing_term_coefficient;
    }

  private:
    virtual void apply_add(
        LinearAlgebra::distributed::Vector<number> &dst,
        const LinearAlgebra::distributed::Vector<number> &src) const override;

    void
    local_apply(const MatrixFree<dim, number> &data,
                LinearAlgebra::distributed::Vector<number> &dst,
                const LinearAlgebra::distributed::Vector<number> &src,
                const std::pair<unsigned int, unsigned int> &cell_range) const;

    void local_compute_diagonal(
        const MatrixFree<dim, number> &data,
        LinearAlgebra::distributed::Vector<number> &dst,
        const unsigned int &dummy,
        const std::pair<unsigned int, unsigned int> &cell_range) const;

    Table<2, VectorizedArray<number>> diffusion_coefficient;
    Table<2, Tensor<1, dim, VectorizedArray<number>>> transport_coefficient;
    Table<2, VectorizedArray<number>> reaction_coefficient;
    Table<2, VectorizedArray<number>> forcing_term_coefficient;
  };


  template <int dim>
  class DTRProblem
  {
  public:
    DTRProblem(bool verbose = true);

    /**
     * @brief Compute the solution of the ADR problem.
     * It executes the setup, rhs assembly, solve, and output_results steps for the solution of the problem.
     * The given number of initial refinements determines the number of cells in the mesh at the first step of the multigrid.
     * The initial number of cells is dim^(initial_refinements-dim).
     * @param n_initial_refinements the number of initial refinements to perform on the mesh.
     */
    void run(unsigned int n_initial_refinements = 3);
    double compute_error(const VectorTools::NormType &norm_type) const;

    unsigned int get_cells() const { return triangulation.n_active_cells(); }
    unsigned int get_dofs() const { return dof_handler.n_dofs(); }
    unsigned int get_fe_degree() const { return degree_finite_element; }

  private:
    void setup_system();
    void assemble_rhs();
    void solve();
    void output_results(const unsigned int cycle) const;

#ifdef DEAL_II_WITH_P4EST
    parallel::distributed::Triangulation<dim> triangulation;
#else
    Triangulation<dim> triangulation;
#endif

    FE_Q<dim> fe;
    DoFHandler<dim> dof_handler;

    // Mapping with polynomial degree=1
    MappingQ1<dim> mapping;

    AffineConstraints<double> constraints;
    using SystemMatrixType =
        DTROperation<dim, degree_finite_element, double>;
    SystemMatrixType system_matrix;

    MGConstrainedDoFs mg_constrained_dofs;
    using LevelMatrixType = DTROperation<dim, degree_finite_element, float>;
    MGLevelObject<LevelMatrixType> mg_matrices;

    LinearAlgebra::distributed::Vector<double> solution;
    LinearAlgebra::distributed::Vector<double> lifting;
    LinearAlgebra::distributed::Vector<double> system_rhs;

    double setup_time;
    ConditionalOStream pcout;
    ConditionalOStream time_details;

    DirichletBC dirichletBC;
    NeumannBC1 neumannBC1;
    NeumannBC2 neumannBC2;
  };
}

#include "DTR_mf.cpp"