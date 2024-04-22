#include "DTR_mf.hpp"

namespace Step37
{
  // This is the constructor of the @p DTROperation class.All it does is
  // to call the default constructor of the base class
  // MatrixFreeOperators::Base, which in turn is based on the Subscriptor
  // class that asserts that this class is not accessed after going out of scope
  // e.g. in a preconditioner.
  template <int dim, int fe_degree, typename number>
  DTROperation<dim, fe_degree, number>::DTROperation()
      : MatrixFreeOperators::Base<dim,
                                  LinearAlgebra::distributed::Vector<number>>()
  {
  }

  template <int dim, int fe_degree, typename number>
  void DTROperation<dim, fe_degree, number>::clear()
  {
    diffusion_coefficient.reinit(0, 0);
    MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number>>::
        clear();
  }

  // To initialize the coefficient, we directly give it the DiffusionCoefficient class
  // defined above and then select the method
  // <code>coefficient_function.value</code> with vectorized number (which the
  // compiler can deduce from the point data type). The use of the
  // FEEvaluation class (and its template arguments) will be explained below.
  template <int dim, int fe_degree, typename number>
  void DTROperation<dim, fe_degree, number>::evaluate_coefficients(
      const DiffusionCoefficient<dim> &diffusion_function,
      const TransportCoefficient<dim> &transport_function,
      const ReactionCoefficient<dim> &reaction_function,
      const ForcingTerm<dim> &forcing_term_function)
  {
    const unsigned int n_cells = this->data->n_cell_batches();
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> fe_eval(*this->data);

    diffusion_coefficient.reinit(n_cells, fe_eval.n_q_points);
    transport_coefficient.reinit(n_cells, fe_eval.n_q_points);
    reaction_coefficient.reinit(n_cells, fe_eval.n_q_points);
    forcing_term_coefficient.reinit(n_cells, fe_eval.n_q_points);

    for (unsigned int cell = 0; cell < n_cells; ++cell)
    {
      fe_eval.reinit(cell);
      for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
      {
        // Diffusion scalar coefficient
        diffusion_coefficient(cell, q) =
            diffusion_function.value(fe_eval.quadrature_point(q));
        // Transport vector coefficient
        evaluate_vector_function<dim, number>(
            transport_function, fe_eval.quadrature_point(q));
        // Reaction scalar coefficient
        reaction_coefficient(cell, q) =
            reaction_function.value(fe_eval.quadrature_point(q));
        // Forcing term scalar coefficient
        forcing_term_coefficient(cell, q) =
            forcing_term_function.value(fe_eval.quadrature_point(q));
      }
    }
  }

  // Here comes the main function of this class, the evaluation of the
  // matrix-vector product (or, in general, a finite element operator
  // evaluation).
  template <int dim, int fe_degree, typename number>
  void DTROperation<dim, fe_degree, number>::local_apply(
      const MatrixFree<dim, number> &data,
      LinearAlgebra::distributed::Vector<number> &dst,
      const LinearAlgebra::distributed::Vector<number> &src,
      const std::pair<unsigned int, unsigned int> &cell_range) const
  {
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> fe_eval(data);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      AssertDimension(diffusion_coefficient.size(0), data.n_cell_batches());
      AssertDimension(reaction_coefficient.size(0), data.n_cell_batches());
      AssertDimension(diffusion_coefficient.size(1), fe_eval.n_q_points);
      AssertDimension(reaction_coefficient.size(1), fe_eval.n_q_points);

      fe_eval.reinit(cell);
      fe_eval.read_dof_values(src);

      fe_eval.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
      for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
      {
        fe_eval.submit_gradient(diffusion_coefficient(cell, q) * fe_eval.get_gradient(q), q);
        fe_eval.submit_value(scalar_product(transport_coefficient(cell, q), fe_eval.get_gradient(q)), q);
        fe_eval.submit_value(reaction_coefficient(cell, q) * fe_eval.get_value(q), q);
      }
      fe_eval.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
      fe_eval.distribute_local_to_global(dst);
    }
  }

  // This function implements the loop over all cells for the
  // Base::apply_add() interface. It is used by vmult()
  template <int dim, int fe_degree, typename number>
  void DTROperation<dim, fe_degree, number>::apply_add(
      LinearAlgebra::distributed::Vector<number> &dst,
      const LinearAlgebra::distributed::Vector<number> &src) const
  {
    this->data->cell_loop(&DTROperation::local_apply, this, dst, src);
  }

  // The following function implements the computation of the diagonal of the
  // operator. Computing matrix entries of a matrix-free operator evaluation
  // turns out to be more complicated than evaluating the
  // operator.
  template <int dim, int fe_degree, typename number>
  void DTROperation<dim, fe_degree, number>::compute_diagonal()
  {
    this->inverse_diagonal_entries.reset(
        new DiagonalMatrix<LinearAlgebra::distributed::Vector<number>>());
    LinearAlgebra::distributed::Vector<number> &inverse_diagonal =
        this->inverse_diagonal_entries->get_vector();
    this->data->initialize_dof_vector(inverse_diagonal);
    unsigned int dummy = 0;
    this->data->cell_loop(&DTROperation::local_compute_diagonal,
                          this,
                          inverse_diagonal,
                          dummy);

    this->set_constrained_entries_to_one(inverse_diagonal);

    for (unsigned int i = 0; i < inverse_diagonal.locally_owned_size(); ++i)
    {
      Assert(inverse_diagonal.local_element(i) > 0.,
             ExcMessage("No diagonal entry in a positive definite operator "
                        "should be zero"));
      inverse_diagonal.local_element(i) =
          1. / inverse_diagonal.local_element(i);
    }
  }

  // In the local compute loop, we compute the diagonal by a loop over all
  // columns in the local matrix and putting the entry 1 in the i-th
  // slot and a zero entry in all other slots, i.e., we apply the cell-wise
  // differential operator on one unit vector at a time. The inner part
  // invoking FEEvaluation::evaluate, the loop over quadrature points, and
  // FEEvalution::integrate, is exactly the same as in the local_apply
  // function
  template <int dim, int fe_degree, typename number>
  void DTROperation<dim, fe_degree, number>::local_compute_diagonal(
      const MatrixFree<dim, number> &data,
      LinearAlgebra::distributed::Vector<number> &dst,
      const unsigned int &,
      const std::pair<unsigned int, unsigned int> &cell_range) const
  {
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> fe_eval(data);

    AlignedVector<VectorizedArray<number>> diagonal(fe_eval.dofs_per_cell);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      AssertDimension(diffusion_coefficient.size(0), data.n_cell_batches());
      AssertDimension(diffusion_coefficient.size(1), fe_eval.n_q_points);

      fe_eval.reinit(cell);
      for (unsigned int i = 0; i < fe_eval.dofs_per_cell; ++i)
      {
        for (unsigned int j = 0; j < fe_eval.dofs_per_cell; ++j)
          fe_eval.submit_dof_value(VectorizedArray<number>(), j);
        fe_eval.submit_dof_value(make_vectorized_array<number>(1.), i);

        fe_eval.evaluate(EvaluationFlags::gradients);
        for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
          fe_eval.submit_gradient(diffusion_coefficient(cell, q) * fe_eval.get_gradient(q),
                                  q);
        fe_eval.integrate(EvaluationFlags::gradients);
        diagonal[i] = fe_eval.get_dof_value(i);
      }
      for (unsigned int i = 0; i < fe_eval.dofs_per_cell; ++i)
        fe_eval.submit_dof_value(diagonal[i], i);
      fe_eval.distribute_local_to_global(dst);
    }
  }

  // When we initialize the finite element, we of course have to use the
  // degree specified at the top of the file as well (otherwise, an exception
  // will be thrown at some point, since the computational kernel defined in
  // the templated DTROperation class and the information from the finite
  // element read out by MatrixFree will not match). The constructor of the
  // triangulation needs to set an additional flag that tells the grid to
  // conform to the 2:1 cell balance over vertices, which is needed for the
  // convergence of the geometric multigrid routines. For the distributed
  // grid, we also need to specifically enable the multigrid hierarchy.
  template <int dim>
  LaplaceProblem<dim>::LaplaceProblem(const std::string &mesh_file_name_)
#ifdef DEAL_II_WITH_P4EST
      : triangulation(MPI_COMM_WORLD,
                      Triangulation<dim>::limit_level_difference_at_vertices,
                      parallel::distributed::Triangulation<
                          dim>::construct_multigrid_hierarchy)
#else
      : triangulation(Triangulation<dim>::limit_level_difference_at_vertices)
#endif
        ,
        fe(degree_finite_element),
        dof_handler(triangulation),
        // mapping(FE_SimplexP<dim>(1)),
        mapping(),
        mesh_file_name(mesh_file_name_),
        setup_time(0.),
        pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
        // The LaplaceProblem class holds an additional output stream that
        // collects detailed timings about the setup phase. This stream, called
        // time_details, is disabled by default through the @p false argument
        // specified here. For detailed timings, removing the @p false argument
        // prints all the details.
        time_details(std::cout, false && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
  }

  // The setup stage is in analogy to step-16 with relevant changes due to the
  // DTROperation class. The first thing to do is to set up the DoFHandler,
  // including the degrees of freedom for the multigrid levels, and to
  // initialize constraints from hanging nodes and homogeneous Dirichlet
  // conditions. Since we intend to use this programs in %parallel with MPI,
  // we need to make sure that the constraints get to know the locally
  // relevant degrees of freedom, otherwise the storage would explode when
  // using more than a few hundred millions of degrees of freedom, see
  // step-40.

  template <int dim>
  void LaplaceProblem<dim>::setup_system()
  {
    Timer time;
    setup_time = 0;

    system_matrix.clear();
    mg_matrices.clear_elements();

    dof_handler.distribute_dofs(fe);
    dof_handler.distribute_mg_dofs(); //

    pcout << "Number of degrees of freedom: " << dof_handler.n_dofs()
          << std::endl;

    const IndexSet locally_relevant_dofs =
        DoFTools::extract_locally_relevant_dofs(dof_handler);

    // Setup Dirichlet and hanging nodes constraints
    // NB: inhomogeneous Dirichlet conditions are not considered by read_dof_values during solve()
    constraints.clear();
    constraints.reinit(locally_relevant_dofs);
    // ! add each D boundary condition
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    VectorTools::interpolate_boundary_values(
        mapping, dof_handler, 0, DirichletBC(), constraints);

    constraints.close();

    setup_time += time.wall_time();
    time_details << "Distribute DoFs & B.C.     (CPU/wall) " << time.cpu_time()
                 << "s/" << time.wall_time() << 's' << std::endl;
    time.restart();

    {
      // additional data telling to reinit that no threads are being used
      typename MatrixFree<dim, double>::AdditionalData additional_data;
      additional_data.tasks_parallel_scheme = MatrixFree<dim, double>::AdditionalData::none;
      // ! changed here
      additional_data.mapping_update_flags =
          (update_values | update_gradients | update_JxW_values | update_quadrature_points);
      std::shared_ptr<MatrixFree<dim, double>> system_mf_storage(new MatrixFree<dim, double>());
      system_mf_storage->reinit(mapping,
                                dof_handler,
                                constraints,
                                QGauss<1>(fe.degree + 1),
                                additional_data);
      system_matrix.initialize(system_mf_storage);
    }

    system_matrix.evaluate_coefficients(DiffusionCoefficient<dim>(),
                                        TransportCoefficient<dim>(),
                                        ReactionCoefficient<dim>(),
                                        ForcingTerm<dim>());

    system_matrix.initialize_dof_vector(solution);   //
    system_matrix.initialize_dof_vector(system_rhs); //

    setup_time += time.wall_time();
    time_details << "Setup matrix-free system   (CPU/wall) " << time.cpu_time()
                 << "s/" << time.wall_time() << 's' << std::endl;
    time.restart();

    // Next, initialize the matrices for the multigrid method on all the
    // levels. The data structure MGConstrainedDoFs keeps information about
    // the indices subject to boundary conditions as well as the indices on
    // edges between different refinement levels as described in the step-16
    // tutorial program. We then go through the levels of the mesh and
    // construct the constraints and matrices on each level. These follow
    // closely the construction of the system matrix on the original mesh,
    // except the slight difference in naming when accessing information on
    // the levels rather than the active cells.
    const unsigned int nlevels = triangulation.n_global_levels();
    mg_matrices.resize(0, nlevels - 1);
    // ! set index
    const std::set<types::boundary_id> dirichlet_boundary_ids = {0};
    mg_constrained_dofs.initialize(dof_handler);
    mg_constrained_dofs.make_zero_boundary_constraints(dof_handler,
                                                       dirichlet_boundary_ids);

    for (unsigned int level = 0; level < nlevels; ++level)
    {
      const IndexSet relevant_dofs =
          DoFTools::extract_locally_relevant_level_dofs(dof_handler, level);
      AffineConstraints<double> level_constraints;
      level_constraints.reinit(relevant_dofs);
      level_constraints.add_lines(
          mg_constrained_dofs.get_boundary_indices(level));
      level_constraints.close();

      typename MatrixFree<dim, float>::AdditionalData additional_data;
      additional_data.tasks_parallel_scheme =
          MatrixFree<dim, float>::AdditionalData::none;
      additional_data.mapping_update_flags =
          (update_gradients | update_JxW_values | update_quadrature_points);
      additional_data.mg_level = level;
      std::shared_ptr<MatrixFree<dim, float>> mg_mf_storage_level(
          new MatrixFree<dim, float>());
      mg_mf_storage_level->reinit(mapping,
                                  dof_handler,
                                  level_constraints,
                                  QGauss<1>(fe.degree + 1),
                                  additional_data);

      mg_matrices[level].initialize(mg_mf_storage_level,
                                    mg_constrained_dofs,
                                    level);
      mg_matrices[level].evaluate_coefficients(DiffusionCoefficient<dim>(),
                                               TransportCoefficient<dim>(),
                                               ReactionCoefficient<dim>(),
                                               ForcingTerm<dim>());
    }
    setup_time += time.wall_time();
    time_details << "Setup matrix-free levels   (CPU/wall) " << time.cpu_time()
                 << "s/" << time.wall_time() << 's' << std::endl;
  }

  // The assemble function is very simple since all we have to do is to
  // assemble the right hand side. Also we need to handle the inhomogeneous Dirichlet constraints.
  // A compress() call at the end of the assembly is needed to send all the
  // contributions of the right hand side to the owner of the respective degree of freedom.
  template <int dim>
  void LaplaceProblem<dim>::assemble_rhs()
  {
    Timer time;

    system_rhs = 0;

    // Set constrained dofs to satisfy inhomogeneous Dirichlet constraints
    solution = 0;
    constraints.distribute(solution);
    solution.update_ghost_values();
    // Reference to diffusion coefficient
    const Table<2, VectorizedArray<double>> &diffusion_coefficient =
        system_matrix.get_diffusion_coefficient();
    const Table<2, VectorizedArray<double>> &forcing_term_coefficient =
        system_matrix.get_forcing_term_coefficient();

    FEEvaluation<dim, degree_finite_element> fe_eval(*system_matrix.get_matrix_free());

    for (unsigned int cell = 0; cell < system_matrix.get_matrix_free()->n_cell_batches(); ++cell)
    {
      fe_eval.reinit(cell);
      // read_dof_values_plain stores internally the values on the current cell for dofs that have no constraints
      // This is needed to leave unchanged the values on the previously setted value of the dofs
      // with inhomogeneous Dirichlet constraints
      fe_eval.read_dof_values_plain(solution);
      fe_eval.evaluate(EvaluationFlags::gradients);

      for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
      {
        fe_eval.submit_gradient(-diffusion_coefficient(cell, q) * fe_eval.get_gradient(q), q);
        fe_eval.submit_value(forcing_term_coefficient(cell, q), q);
      }

      fe_eval.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
      fe_eval.distribute_local_to_global(system_rhs);
      // no need for constraints.distribute_local_to_global since is done by the above
    }
    system_rhs.compress(VectorOperation::add);

    setup_time += time.wall_time();
    time_details << "Assemble right hand side   (CPU/wall) " << time.cpu_time()
                 << "s/" << time.wall_time() << 's' << std::endl;
  }

  // The solution process is similar as in step-16. We start with the setup of
  // the transfer. For LinearAlgebra::distributed::Vector, there is a very
  // fast transfer class called MGTransferMatrixFree that does the
  // interpolation between the grid levels with the same fast sum
  // factorization kernels that get also used in FEEvaluation.
  template <int dim>
  void LaplaceProblem<dim>::solve()
  {
    Timer time;
    MGTransferMatrixFree<dim, float> mg_transfer(mg_constrained_dofs);
    mg_transfer.build(dof_handler);
    setup_time += time.wall_time();
    time_details << "MG build transfer time     (CPU/wall) " << time.cpu_time()
                 << "s/" << time.wall_time() << "s\n";
    time.restart();

    // As a smoother, this tutorial program uses a Chebyshev iteration instead
    // of SOR in step-16. (SOR would be very difficult to implement because we
    // do not have the matrix elements available explicitly, and it is
    // difficult to make it work efficiently in %parallel.)
    using SmootherType =
        PreconditionChebyshev<LevelMatrixType,
                              LinearAlgebra::distributed::Vector<float>>;
    mg::SmootherRelaxation<SmootherType,
                           LinearAlgebra::distributed::Vector<float>>
        mg_smoother;
    MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
    smoother_data.resize(0, triangulation.n_global_levels() - 1);
    for (unsigned int level = 0; level < triangulation.n_global_levels();
         ++level)
    {
      if (level > 0)
      {
        smoother_data[level].smoothing_range = 15.;
        smoother_data[level].degree = 5;
        smoother_data[level].eig_cg_n_iterations = 10;
      }
      else
      {
        smoother_data[0].smoothing_range = 1e-3;
        smoother_data[0].degree = numbers::invalid_unsigned_int;
        smoother_data[0].eig_cg_n_iterations = mg_matrices[0].m();
      }
      mg_matrices[level].compute_diagonal();
      smoother_data[level].preconditioner =
          mg_matrices[level].get_matrix_diagonal_inverse();
    }
    mg_smoother.initialize(mg_matrices, smoother_data);

    MGCoarseGridApplySmoother<LinearAlgebra::distributed::Vector<float>>
        mg_coarse;
    mg_coarse.initialize(mg_smoother);

    // The next step is to set up the interface matrices that are needed for the
    // case with hanging nodes. The adaptive multigrid realization in deal.II
    // implements an approach called local smoothing. This means that the
    // smoothing on the finest level only covers the local part of the mesh
    // defined by the fixed (finest) grid level and ignores parts of the
    // computational domain where the terminal cells are coarser than this
    // level. As the method progresses to coarser levels, more and more of the
    // global mesh will be covered.
    mg::Matrix<LinearAlgebra::distributed::Vector<float>> mg_matrix(
        mg_matrices);

    MGLevelObject<MatrixFreeOperators::MGInterfaceOperator<LevelMatrixType>>
        mg_interface_matrices;
    mg_interface_matrices.resize(0, triangulation.n_global_levels() - 1);
    for (unsigned int level = 0; level < triangulation.n_global_levels();
         ++level)
      mg_interface_matrices[level].initialize(mg_matrices[level]);
    mg::Matrix<LinearAlgebra::distributed::Vector<float>> mg_interface(
        mg_interface_matrices);

    Multigrid<LinearAlgebra::distributed::Vector<float>> mg(
        mg_matrix, mg_coarse, mg_transfer, mg_smoother, mg_smoother);
    mg.set_edge_matrices(mg_interface, mg_interface);

    PreconditionMG<dim,
                   LinearAlgebra::distributed::Vector<float>,
                   MGTransferMatrixFree<dim, float>>
        preconditioner(dof_handler, mg, mg_transfer);

    // The setup of the multigrid routines is quite easy and one cannot see
    // any difference in the solve process as compared to step-16. All the
    // magic is hidden behind the implementation of the DTROperation::vmult
    // operation. Note that we print out the solve time and the accumulated
    // setup time through standard out, i.e., in any case, whereas detailed
    // times for the setup operations are only printed in case the flag for
    // detail_times in the constructor is changed.

    SolverControl solver_control(100, 1e-12 * system_rhs.l2_norm());
    SolverGMRES<LinearAlgebra::distributed::Vector<double>> solver(solver_control);

    time.reset();
    time.start();
    constraints.set_zero(solution); // ! why set_zero
    solver.solve(system_matrix, solution, system_rhs, preconditioner);

    constraints.distribute(solution);

    pcout << "Time solve (" << solver_control.last_step() << " iterations)"
          << (solver_control.last_step() < 10 ? "  " : " ") << "(CPU/wall) "
          << time.cpu_time() << "s/" << time.wall_time() << "s\n";
  }

  // @sect4{LaplaceProblem::output_results}

  // Here is the data output, which is a simplified version of step-5. We use
  // the standard VTU (= compressed VTK) output for each grid produced in the
  // refinement process. In addition, we use a compression algorithm that is
  // optimized for speed rather than disk usage. The default setting (which
  // optimizes for disk usage) makes saving the output take about 4 times as
  // long as running the linear solver, while setting
  // DataOutBase::CompressionLevel to
  // best_speed lowers this to only one fourth the time
  // of the linear solve.
  //
  // We disable the output when the mesh gets too large. A variant of this
  // program has been run on hundreds of thousands MPI ranks with as many as
  // 100 billion grid cells, which is not directly accessible to classical
  // visualization tools.
  template <int dim>
  void LaplaceProblem<dim>::output_results(const unsigned int cycle) const
  {
    Timer time;
    if (triangulation.n_global_active_cells() > 1000000)
      return;

    DataOut<dim> data_out;

    solution.update_ghost_values(); //
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution");
    data_out.build_patches(mapping); // no param in mpi

    DataOutBase::VtkFlags flags;
    flags.compression_level = DataOutBase::CompressionLevel::best_speed;
    data_out.set_flags(flags);
    data_out.write_vtu_with_pvtu_record(
        "./", "solution", cycle, MPI_COMM_WORLD, 3);

    time_details << "Time write output          (CPU/wall) " << time.cpu_time()
                 << "s/" << time.wall_time() << "s\n";
  }

  template <int dim>
  void LaplaceProblem<dim>::run()
  {
    // Print processor vectorization details
    {
      const unsigned int n_vect_doubles = VectorizedArray<double>::size();
      const unsigned int n_vect_bits = 8 * sizeof(double) * n_vect_doubles;

      pcout << "Vectorization over " << n_vect_doubles
            << " doubles = " << n_vect_bits << " bits ("
            << Utilities::System::get_current_vectorization_level() << ')'
            << std::endl;
    }

    // Create the mesh reading the mesh from file
    /*{
      pcout << "Initialize triangulation" << std::endl;
      // Read the mesh to a serial triangulation
      Triangulation<dim> triangulation_serial;
      {
        GridIn<dim> grid_in;
        grid_in.attach_triangulation(triangulation_serial);

        std::ifstream grid_in_file(mesh_file_name);
        grid_in.read_msh(grid_in_file);
      }
      // Copy the triangulation into a parallel one
      {
        GridTools::partition_triangulation(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), triangulation_serial);
        const auto construction_data = TriangulationDescription::Utilities::
            create_description_from_triangulation(triangulation_serial, MPI_COMM_WORLD);
        triangulation.create_triangulation(construction_data);
      }

      pcout << "  Number of elements = " << triangulation.n_active_cells()
            << std::endl;
    }*/

    for (unsigned int cycle = 0; cycle < 9 - dim; ++cycle)
    {
      pcout << "Cycle " << cycle << std::endl;

      if (cycle == 0)
      {
        GridGenerator::hyper_cube(triangulation, 0., 1.);
        // GridGenerator::convert_hypercube_to_simplex_mesh(triangulation_serial, triangulation);
        triangulation.refine_global(3 - dim);
      }

      triangulation.refine_global(1);
      setup_system();
      assemble_rhs();
      solve();
      output_results(cycle);
      pcout << std::endl;
    };
  }
}