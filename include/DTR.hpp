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
 * @brief Class managing the differential problem.
 *
 * This class handles the setup, assembly, solution, and output of a differential problem 
 * using the deal.II library. It supports parallel computation using MPI.
 */
class DTR
{
public:
  /// Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 2;

  /**
   * @brief Constructor with a file stream for timing information.
   *
   * @param r_ Polynomial degree of the finite element basis functions.
   * @param dimension_time_file Output file stream for logging timing information.
   */
  DTR(const unsigned int &r_, std::ofstream& dimension_time_file);

  /**
   * @brief Constructor without a file stream for timing information.
   *
   * @param r_ Polynomial degree of the finite element basis functions.
   */
  DTR(const unsigned int &r_);

  /**
   * @brief Initialization of the system.
   *
   * This function sets up the triangulation, DoF handler, and system matrices.
   *
   * @param n_initial_refinements Number of initial mesh refinements.
   */
  void setup(unsigned int n_initial_refinements = 8);

  /**
   * @brief Assembly of the system matrix and right-hand side vector.
   */
  void assemble();

  /**
   * @brief Solution of the linear system.
   */
  void solve();

  /**
   * @brief Output of the solution to a file.
   *
   * This function writes the solution to a VTK file for visualization.
   */
  void output() const;

  /**
   * @brief Compute the error of the solution.
   *
   * This function computes the error of the numerical solution with respect to an exact solution.
   *
   * @param norm_type The type of norm to use for error computation.
   * @return The computed error.
   */
  double compute_error(const VectorTools::NormType &norm_type) const;

protected:
  /** 
   * @brief Path to the mesh file.
   */
  const std::string mesh_file_name;

  /**
   * @brief Polynomial degree.
   */
  const unsigned int r;

  /**
   * @brief Number of MPI processes.
   */
  const unsigned int mpi_size;

  /**
   * @brief Rank of this MPI process.
   */
  const unsigned int mpi_rank;

  /**
   * @brief Diffusion coefficient.
   */
  problem_data::DiffusionCoefficient<dim> diffusion_coefficient;

  /**
   * @brief Reaction coefficient.
   */
  problem_data::ReactionCoefficient<dim> reaction_coefficient;

  /**
   * @brief Transport coefficient.
   */
  problem_data::TransportCoefficient<dim> transport_coefficient;

  /**
   * @brief Forcing term.
   */
  problem_data::ForcingTerm<dim> forcing_term;

  /**
   * @brief Dirichlet boundary conditions for the left and bottom boundaries.
   */
  problem_data::DirichletBC1<dim> dirichletBC1;

  /**
   * @brief Dirichlet boundary conditions for the right and top boundaries.
   */
  problem_data::DirichletBC2<dim> dirichletBC2;

  /**
   * @brief Neumann boundary conditions for the right and top boundaries.
   */
  problem_data::NeumannBC1<dim> neumannBC1;

  /**
   * @brief Neumann boundary conditions for the left and bottom boundaries.
   */
  problem_data::NeumannBC2<dim> neumannBC2;

  /**
   * @brief Triangulation object for managing the mesh.
   */
  parallel::fullydistributed::Triangulation<dim> mesh;

  /**
   * @brief Finite element space.
   */
  std::unique_ptr<FiniteElement<dim>> fe;

  /**
   * @brief Quadrature formula for integration.
   */
  std::unique_ptr<Quadrature<dim>> quadrature;

  /**
   * @brief Quadrature formula for boundary integration.
   */
  std::unique_ptr<Quadrature<dim - 1>> quadrature_boundary;

  /**
   * @brief DoF handler for managing degrees of freedom.
   */
  DoFHandler<dim> dof_handler;

  /**
   * @brief System matrix.
   */
  TrilinosWrappers::SparseMatrix system_matrix;

  /**
   * @brief System right-hand side vector.
   */
  TrilinosWrappers::MPI::Vector system_rhs;

  /**
   * @brief Solution vector.
   */
  TrilinosWrappers::MPI::Vector solution;

  /**
   * @brief Parallel output stream.
   */
  ConditionalOStream pcout;

  /**
   * @brief Output stream for timing details.
   */
  ConditionalOStream time_details;

  /**
   * @brief Index set for locally owned degrees of freedom.
   */
  IndexSet locally_owned_dofs;

  /**
   * @brief Time taken for setup.
   */
  double setup_time;
};

#endif
