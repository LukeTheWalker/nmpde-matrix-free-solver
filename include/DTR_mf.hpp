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

#include "problem_data.hpp"

const std::string output_dir = "./output_mf/";

namespace DTR_mf
{
  using namespace dealii;

  // To be efficient matrix-free implementation require knowledge of loop lengths at compile time
  const unsigned int dim = 2;
  
  /**
   * @brief The DTROperation class implements the DTR linear operation needed at each iteration of the linear solver.
   *
   * This class is derived from the MatrixFreeOperators::Base class provided by deal.II.
   * It is responsible for evaluating the coefficients, computing the diagonal, and applying the linear operation
   * required by the Conjugate Gradient solver for the DTR problem.
   *
   * @tparam dim The dimension of the problem.
   * @tparam fe_degree The degree of the finite element used.
   * @tparam number The number type used for the computations.
   */
  template <int dim, int fe_degree, typename number>
  class DTROperation
      : public MatrixFreeOperators::
            Base<dim, LinearAlgebra::distributed::Vector<number>>
  {
  public:
    using value_type = number;

    /**
     * @brief Default constructor.
     */
    DTROperation();

    /**
     * @brief Clear the operation.
     */
    void clear() override;

    /**
     * @brief Evaluate the coefficients of the DTR problem.
     *
     * @param diffusion_function The diffusion coefficient function.
     * @param transport_function The transport coefficient function.
     * @param reaction_function The reaction coefficient function.
     * @param forcing_term_function The forcing term function.
     */
    void evaluate_coefficients(const problem_data::DiffusionCoefficient<dim> &diffusion_function,
                               const problem_data::TransportCoefficient<dim> &transport_function,
                               const problem_data::ReactionCoefficient<dim> &reaction_function,
                               const problem_data::ForcingTerm<dim> &forcing_term_function);

    /**
     * @brief Compute the diagonal entries of the matrix.
     */
    virtual void compute_diagonal() override;

    /**
     * @brief Get the diffusion coefficient table.
     *
     * @return The diffusion coefficient table.
     */
    const Table<2, VectorizedArray<number>> &get_diffusion_coefficient() const
    {
      return diffusion_coefficient;
    }

    /**
     * @brief Get the transport coefficient table.
     *
     * @return The transport coefficient table.
     */
    const Table<2, Tensor<1, dim, VectorizedArray<number>>> &get_transport_coefficient() const
    {
      return transport_coefficient;
    }

    /**
     * @brief Get the reaction coefficient table.
     *
     * @return The reaction coefficient table.
     */
    const Table<2, VectorizedArray<number>> &get_reaction_coefficient() const
    {
      return reaction_coefficient;
    }

    /**
     * @brief Get the forcing term coefficient table.
     *
     * @return The forcing term coefficient table.
     */
    const Table<2, VectorizedArray<number>> &get_forcing_term_coefficient() const
    {
      return forcing_term_coefficient;
    }

  private:
    /**
     * @brief Apply the linear operation and add the result to the destination vector.
     *
     * @param dst The destination vector.
     * @param src The source vector.
     */
    virtual void apply_add(
        LinearAlgebra::distributed::Vector<number> &dst,
        const LinearAlgebra::distributed::Vector<number> &src) const override;

    /**
     * @brief Apply the local linear operation on a range of cells.
     *
     * @param data The MatrixFree object containing the data.
     * @param dst The destination vector.
     * @param src The source vector.
     * @param cell_range The range of cells to operate on.
     */
    void
    local_apply(const MatrixFree<dim, number> &data,
                LinearAlgebra::distributed::Vector<number> &dst,
                const LinearAlgebra::distributed::Vector<number> &src,
                const std::pair<unsigned int, unsigned int> &cell_range) const;

    /**
     * @brief Compute the local diagonal entries of the matrix on a range of cells.
     *
     * @param data The MatrixFree object containing the data.
     * @param dst The destination vector for the diagonal entries.
     * @param dummy A dummy argument (not used).
     * @param cell_range The range of cells to operate on.
     */
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

  /**
   * @brief The DTRProblem class represents the DTR problem and provides methods to solve it.
   *
   * This class encapsulates the setup, assembly, solution, and output of the DTR problem.
   *
   * @tparam dim The dimension of the problem.
   * @tparam degree_finite_element The degree of the finite element used (default is 2).
   */
  template <int dim, int degree_finite_element = 2>
  class DTRProblem
  {
  public:
    /**
     * @brief Constructor for the DTRProblem class.
     *
     * @param verbose A flag indicating whether to print verbose output (default is true).
     */
    DTRProblem(bool verbose = true);

    /**
     * @brief Constructor for the DTRProblem class with a dimension-time file output stream.
     *
     * @param dimension_time_file An output stream for writing dimension and time information.
     * @param verbose A flag indicating whether to print verbose output (default is true).
     */
    DTRProblem(std::ofstream& dimension_time_file, bool verbose = true);

    /**
     * @brief Compute the solution of the DTR problem.
     *
     * This method executes the setup, right-hand side assembly, solution, and output steps
     * for the DTR problem for a given number of times. The number of initial refinements
     * determines the number of cells in the mesh at the first solution, and the number of
     * executed cycles determines the number of additional refinements performed.
     *
     * @param n_initial_refinements The number of initial refinements to perform on the mesh.
     * @param n_cycles The number of solutions to compute by adding a refinement at each iteration.
     */
    void run(unsigned int n_initial_refinements = 3, unsigned int n_cycles = 9);

    /**
     * @brief Compute the error of the solution using the given norm type.
     *
     * @param norm_type The norm type to use for computing the error.
     * @return The computed error.
     */
    double compute_error(const VectorTools::NormType &norm_type) const;

    /**
     * @brief Get the number of global active cells in the triangulation.
     *
     * @return The number of global active cells.
     */
    unsigned int get_cells() const { return triangulation.n_global_active_cells(); }

    /**
     * @brief Get the number of degrees of freedom.
     *
     * @return The number of degrees of freedom.
     */
    unsigned int get_dofs() const { return dof_handler.n_dofs(); }

    /**
     * @brief Get the degree of the finite element.
     *
     * @return The degree of the finite element.
     */
    unsigned int get_fe_degree() const { return degree_finite_element; }

  private:
    /**
     * @brief Set up the system for solving the DTR problem.
     *
     * This method initializes the triangulation, degree of freedom handler, constraints,
     * system matrix, multigrid constraints, and multigrid matrices.
     */
    void setup_system();

    /**
     * @brief Assemble the right-hand side of the DTR problem.
     *
     * This method computes the right-hand side vector for the linear system.
     */
    void assemble_rhs();

    /**
     * @brief Solve the DTR problem using the Conjugate Gradient method.
     *
     * This method solves the linear system using the Conjugate Gradient solver.
     */
    void solve();

    /**
     * @brief Output the results of the DTR problem for the given cycle.
     *
     * @param cycle The cycle number for which to output the results.
     */
    void output_results(const unsigned int cycle) const;

#ifdef DEAL_II_WITH_P4EST
    parallel::distributed::Triangulation<dim> triangulation;
#else
    Triangulation<dim> triangulation;
#endif

    /**
     * @brief The finite element used for the DTR problem.
     */
    FE_Q<dim> fe;

    /**
     * @brief The degree of freedom handler for the DTR problem.
     */
    DoFHandler<dim> dof_handler;

    /**
     * @brief The mapping used for the DTR problem (polynomial degree 1).
     */
    MappingQ1<dim> mapping;

    /**
     * @brief The affine constraints for the DTR problem.
     */
    AffineConstraints<double> constraints;

    /**
     * @brief The system matrix type for the DTR problem.
     */
    using SystemMatrixType =
        DTROperation<dim, degree_finite_element, double>;

    /**
     * @brief The system matrix for the DTR problem.
     */
    SystemMatrixType system_matrix;

    /**
     * @brief The multigrid constrained degrees of freedom for the DTR problem.
     */
    MGConstrainedDoFs mg_constrained_dofs;

    /**
     * @brief The level matrix type for the DTR problem.
     */
    using LevelMatrixType = DTROperation<dim, degree_finite_element, float>;

    /**
     * @brief The multigrid level matrices for the DTR problem.
     */
    MGLevelObject<LevelMatrixType> mg_matrices;

    /**
     * @brief The solution vector for the DTR problem.
     */
    LinearAlgebra::distributed::Vector<double> solution;

    /**
     * @brief The lifting vector for the DTR problem.
     */
    LinearAlgebra::distributed::Vector<double> lifting;

    /**
     * @brief The right-hand side vector for the DTR problem.
     */
    LinearAlgebra::distributed::Vector<double> system_rhs;

    /**
     * @brief The time taken for system setup.
     */
    double setup_time;

    /**
     * @brief A conditional output stream for printing information.
     */
    ConditionalOStream pcout;

    /**
     * @brief A conditional output stream for printing time details.
     */
    ConditionalOStream time_details;

    /**
     * @brief The object representing the Dirichlet boundary condition 1.
     */
    problem_data::DirichletBC1<dim> dirichletBC1;

    /**
     * @brief The object representing the Dirichlet boundary condition 2.
     */
    problem_data::DirichletBC2<dim> dirichletBC2;

    /**
     * @brief The object representing the Neumann boundary condition 1.
     */
    problem_data::NeumannBC1<dim> neumannBC1;

    /**
     * @brief The object representing the Neumann boundary condition 2.
     */
    problem_data::NeumannBC2<dim> neumannBC2;
  };
}

#include "DTR_mf.cpp"