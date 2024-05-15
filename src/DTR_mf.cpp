#include "DTR_mf.hpp"

namespace DTR_mf
{
  // This is the constructor of the @p DTROperation class.All it does is
  // to call the default constructor of the base class
  // MatrixFreeOperators::Base, which in turn is based on the Subscriptor
  // class that asserts that this class is not accessed after going out of scope
  // e.g. in a preconditioner.
  template <int dim, int fe_degree, typename number>
  DTROperation<dim, fe_degree, number>::DTROperation()
      : MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number>>()
  {
  }

  template <int dim, int fe_degree, typename number>
  void DTROperation<dim, fe_degree, number>::clear()
  {
    diffusion_coefficient.reinit(0, 0);
    transport_coefficient.reinit(0, 0);
    reaction_coefficient.reinit(0, 0);
    forcing_term_coefficient.reinit(0, 0);
    MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number>>::clear();
  }

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
      for (const unsigned int q : fe_eval.quadrature_point_indices())
      {
        // Diffusion scalar coefficient
        diffusion_coefficient(cell, q) =
            diffusion_function.value(fe_eval.quadrature_point(q));
        // Transport vector coefficient
        transport_function.tensor_value(fe_eval.quadrature_point(q),
                                        transport_coefficient(cell, q));
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
      AssertDimension(transport_coefficient.size(0), data.n_cell_batches());
      AssertDimension(reaction_coefficient.size(0), data.n_cell_batches());
      AssertDimension(diffusion_coefficient.size(1), fe_eval.n_q_points);
      AssertDimension(transport_coefficient.size(1), fe_eval.n_q_points);
      AssertDimension(reaction_coefficient.size(1), fe_eval.n_q_points);

      fe_eval.reinit(cell);
      fe_eval.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);
      for (const unsigned int q : fe_eval.quadrature_point_indices())
      {
        // Get the gradient of the FE function at quadrature point q since it will be overwritten
        Tensor<1, dim, VectorizedArray<number>> grad = fe_eval.get_gradient(q);
        // Compute the transport and reaction terms
        VectorizedArray<number> transport_value = scalar_product(transport_coefficient(cell, q), grad);
        VectorizedArray<number> reaction_value = reaction_coefficient(cell, q) * fe_eval.get_value(q);
        // Submit the term that will be tested by all basis function gradients on the current cell and integrated over
        fe_eval.submit_gradient(diffusion_coefficient(cell, q) * grad, q);
        // Submit the term that will be tested by all basis function values on the current cell and integrated over
        fe_eval.submit_value(transport_value + reaction_value, q);
      }
      fe_eval.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
    }
  }

  // This function implements the loop over all cells for the Base::apply_add() interface.
  // Used by vmult()
  template <int dim, int fe_degree, typename number>
  void DTROperation<dim, fe_degree, number>::apply_add(
      LinearAlgebra::distributed::Vector<number> &dst,
      const LinearAlgebra::distributed::Vector<number> &src) const
  {
    this->data->cell_loop(&DTROperation::local_apply, this, dst, src);
  }

  // The following function implements the computation of the diagonal of the operator
  template <int dim, int fe_degree, typename number>
  void DTROperation<dim, fe_degree, number>::compute_diagonal()
  {
    // Get and initialize the vector for the inverse diagonal entries
    this->inverse_diagonal_entries.reset(
        new DiagonalMatrix<LinearAlgebra::distributed::Vector<number>>());
    LinearAlgebra::distributed::Vector<number> &inverse_diagonal =
        this->inverse_diagonal_entries->get_vector();
    this->data->initialize_dof_vector(inverse_diagonal);
    unsigned int dummy = 0;
    // Loop over all cells to compute the diagonal entries (no source vector is needed)
    this->data->cell_loop(&DTROperation::local_compute_diagonal, this, inverse_diagonal, dummy);

    // Set all the vector entries constrained by Dirichlet BC to one
    this->set_constrained_entries_to_one(inverse_diagonal);

    for (unsigned int i = 0; i < inverse_diagonal.locally_owned_size(); ++i)
    {
      Assert(inverse_diagonal.local_element(i) > 0.,
             ExcMessage("No diagonal entry in a positive definite operator should be zero"));
      inverse_diagonal.local_element(i) = 1. / inverse_diagonal.local_element(i);
    }
  }

  // In the local compute loop, we apply the cell-wise DTR operator on one unit
  // vector at a time. The inner part is the same as in the local_apply function
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
      AssertDimension(transport_coefficient.size(0), data.n_cell_batches());
      AssertDimension(reaction_coefficient.size(0), data.n_cell_batches());
      AssertDimension(diffusion_coefficient.size(1), fe_eval.n_q_points);
      AssertDimension(transport_coefficient.size(1), fe_eval.n_q_points);
      AssertDimension(reaction_coefficient.size(1), fe_eval.n_q_points);

      fe_eval.reinit(cell);
      for (unsigned int i = 0; i < fe_eval.dofs_per_cell; ++i)
      {
        // Set to 1 all the dofs
        for (unsigned int j = 0; j < fe_eval.dofs_per_cell; ++j)
          fe_eval.submit_dof_value(VectorizedArray<number>(), j);
        fe_eval.submit_dof_value(make_vectorized_array<number>(1.), i);

        fe_eval.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
        // This section is the same as in the local_apply function
        for (const unsigned int q : fe_eval.quadrature_point_indices())
        {
          // Get the gradient of the FE function at quadrature point q since it will be overwritten
          Tensor<1, dim, VectorizedArray<number>> grad = fe_eval.get_gradient(q);
          // Compute the transport and reaction terms
          VectorizedArray<number> transport_value = scalar_product(transport_coefficient(cell, q), grad);
          VectorizedArray<number> reaction_value = reaction_coefficient(cell, q) * fe_eval.get_value(q);
          // Submit the term that will be tested by all basis function gradients on the current cell and integrated over
          fe_eval.submit_gradient(diffusion_coefficient(cell, q) * grad, q);
          // Submit the term that will be tested by all basis function values on the current cell and integrated over
          fe_eval.submit_value(transport_value + reaction_value, q);
        }
        // -- end of the same section
        fe_eval.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
        diagonal[i] = fe_eval.get_dof_value(i);
      }
      for (unsigned int i = 0; i < fe_eval.dofs_per_cell; ++i)
        fe_eval.submit_dof_value(diagonal[i], i);
      fe_eval.distribute_local_to_global(dst);
    }
  }

  // Crucial steps:
  // - The triangulation constructor needs an additional flag that tells the grid to
  //    conform to the 2:1 cell balance over vertices, which is needed for the
  //    convergence of the geometric multigrid routines.
  // - For the distributed grid we need to specifically enable the multigrid hierarchy
  template <int dim>
  DTRProblem<dim>::DTRProblem(bool verbose)
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
        setup_time(0.),
        pcout(std::cout, verbose && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
        // ! remove the false for the additional output stream for timing
        time_details(std::cout, false && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
  }

  template <int dim>
  void DTRProblem<dim>::setup_system()
  {
    Timer time;
    setup_time = 0;

    system_matrix.clear();
    mg_matrices.clear_elements();
    // Setup DoFHandler also for multigrid levels
    dof_handler.distribute_dofs(fe);
    dof_handler.distribute_mg_dofs();

    pcout << "Finite element degree:        " << fe.degree << std::endl;
    pcout << "Number of cells:              " << triangulation.n_active_cells() << std::endl;
    pcout << "Number of DoFs per cell:      " << fe.dofs_per_cell << std::endl;
    pcout << "Number of DoFs:               " << dof_handler.n_dofs() << std::endl;

    // Consider only locally relevant dofs otherwise memory will explode
    const IndexSet locally_relevant_dofs =
        DoFTools::extract_locally_relevant_dofs(dof_handler);

    // Initialize and setup Dirichlet and hanging nodes constraints
    constraints.clear();
    constraints.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    // Set all Dirichlet BC to homogeneous ones
    Functions::ZeroFunction<dim> zero_function;
    std::map<types::boundary_id, const Function<dim> *> boundary_functions;
    for (unsigned int i = 0; i < 4; ++i)
      if (bcs[i] == 'D' || bcs[i] == 'Z')
        boundary_functions[i] = &zero_function;
    VectorTools::interpolate_boundary_values(mapping,
                                             dof_handler,
                                             boundary_functions,
                                             constraints);

    constraints.close();

    setup_time += time.wall_time();
    time_details << "Distribute DoFs & B.C.     (CPU/wall) " << time.cpu_time()
                 << "s/" << time.wall_time() << 's' << std::endl;
    time.restart();

    // Setup the matrix-free instance of the problem and store it in a shared pointer
    {
      typename MatrixFree<dim, double>::AdditionalData additional_data;
      // Enable also multithreading parallelism
      additional_data.tasks_parallel_scheme = MatrixFree<dim, double>::AdditionalData::partition_color;
      // Define the flags for the needed storage
      additional_data.mapping_update_flags =
          (update_values | update_gradients | update_JxW_values | update_quadrature_points);
      additional_data.mapping_update_flags_boundary_faces =
          (update_JxW_values | update_quadrature_points);

      std::shared_ptr<MatrixFree<dim, double>> system_mf_storage(new MatrixFree<dim, double>());

      system_mf_storage->reinit(mapping,
                                dof_handler,
                                constraints,
                                QGauss<1>(fe.degree + 1),
                                additional_data);
      system_matrix.initialize(system_mf_storage);

      pcout << "Quadrature points per face    " << system_mf_storage->get_n_q_points_face() << std::endl;
      pcout << "Quadrature points per cell    " << system_mf_storage->get_n_q_points() << std::endl;
    }

    // Evaluate the coefficients on each cell and dof, save them in Tables
    system_matrix.evaluate_coefficients(DiffusionCoefficient<dim>(),
                                        TransportCoefficient<dim>(),
                                        ReactionCoefficient<dim>(),
                                        ForcingTerm<dim>());

    system_matrix.initialize_dof_vector(solution);
    system_matrix.initialize_dof_vector(lifting);
    system_matrix.initialize_dof_vector(system_rhs);

    setup_time += time.wall_time();
    time_details << "Setup matrix-free system   (CPU/wall) " << time.cpu_time()
                 << "s/" << time.wall_time() << 's' << std::endl;
    time.restart();

    // Initialize the matrices for the multigrid method on all the levels
    const unsigned int nlevels = triangulation.n_global_levels();
    mg_matrices.resize(0, nlevels - 1);
    // Set all the Dirichlet BC to homogeneous ones
    std::set<types::boundary_id> dirichlet_boundary_ids;
    for (unsigned int i = 0; i < 4; ++i)
      if (bcs[i] == 'D' || bcs[i] == 'Z')
        dirichlet_boundary_ids.emplace(i);
    // MGConstrainedDoFs keeps indices subject to BCs and indices on edges between
    // different refinement levels
    mg_constrained_dofs.initialize(dof_handler);
    mg_constrained_dofs.make_zero_boundary_constraints(dof_handler, dirichlet_boundary_ids);

    // Construct the constraints and matrices on each level following closely the
    // construction of the system matrix on the original mesh
    for (unsigned int level = 0; level < nlevels; ++level)
    {
      const IndexSet relevant_dofs =
          DoFTools::extract_locally_relevant_level_dofs(dof_handler, level);
      AffineConstraints<double> level_constraints;
      level_constraints.reinit(relevant_dofs);
      level_constraints.add_lines(mg_constrained_dofs.get_boundary_indices(level));
      level_constraints.close();

      typename MatrixFree<dim, float>::AdditionalData additional_data;
      // Enable also multithreading parallelism
      additional_data.tasks_parallel_scheme = MatrixFree<dim, float>::AdditionalData::partition_color;
      additional_data.mapping_update_flags =
          (update_values | update_gradients | update_JxW_values | update_quadrature_points);
      additional_data.mapping_update_flags_boundary_faces =
          (update_JxW_values | update_quadrature_points);
      additional_data.mg_level = level;

      std::shared_ptr<MatrixFree<dim, float>> mg_mf_storage_level(new MatrixFree<dim, float>());

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

  // Assemble rhs and handle the inhomogeneous Dirichlet constraints
  template <int dim>
  void DTRProblem<dim>::assemble_rhs()
  {
    Timer time;

    system_rhs = 0.;
    lifting = 0.;

    // Interpolate boundary values for inhomogeneous Dirichlet BC on vector lifting
    std::map<types::global_dof_index, double> boundary_values;
    std::map<types::boundary_id, const Function<dim> *> boundary_functions;

    boundary_functions[0] = &dirichletBC1;
    boundary_functions[2] = &dirichletBC2;

    VectorTools::interpolate_boundary_values(mapping,
                                             dof_handler,
                                             boundary_functions,
                                             boundary_values);
    for (const std::pair<const types::global_dof_index, double> &pair : boundary_values)
      if (lifting.locally_owned_elements().is_element(pair.first))
        lifting(pair.first) = pair.second;
    lifting.update_ghost_values();

    // Reference to coefficients
    const Table<2, VectorizedArray<double>> &diffusion_coefficient =
        system_matrix.get_diffusion_coefficient();
    const Table<2, Tensor<1, dim, VectorizedArray<double>>> &transport_coefficient =
        system_matrix.get_transport_coefficient();
    const Table<2, VectorizedArray<double>> &reaction_coefficient =
        system_matrix.get_reaction_coefficient();
    const Table<2, VectorizedArray<double>> &forcing_term_coefficient =
        system_matrix.get_forcing_term_coefficient();

    FEEvaluation<dim, degree_finite_element> fe_eval(*system_matrix.get_matrix_free());

    // Loop over the cells to add the forcing term and lifting contribution
    for (unsigned int cell = 0; cell < system_matrix.get_matrix_free()->n_cell_batches(); ++cell)
    {
      fe_eval.reinit(cell);
      // read_dof_values_plain stores internally the values on the current cell for dofs that
      // have no constraints. This is needed to leave unchanged the values on the previously
      // setted value of the dofs with inhomogeneous Dirichlet constraints
      fe_eval.read_dof_values_plain(lifting);
      fe_eval.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

      for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
      {
        // Get the gradient of the FE function at quadrature point q since it will be overwritten
        Tensor<1, dim, VectorizedArray<double>> grad = fe_eval.get_gradient(q);

        // Compute the transport and reaction terms
        VectorizedArray<double> transport_value = scalar_product(transport_coefficient(cell, q), grad);
        VectorizedArray<double> reaction_value = reaction_coefficient(cell, q) * fe_eval.get_value(q);

        // Submit the term that will be tested by all basis function gradients on the current cell and integrated over
        fe_eval.submit_gradient(-diffusion_coefficient(cell, q) * grad, q);
        // Submit the term that will be tested by all basis function values on the current cell and integrated over
        fe_eval.submit_value(-transport_value - reaction_value + forcing_term_coefficient(cell, q), q);
      }

      fe_eval.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, system_rhs);
      // no need for constraints.distribute_local_to_global since is done by the above function
    }

    // Loop over the boundary faces to add the Neumann BC contribution
    // Since the matrixfree structure does not cache internal face data, boundary data starts from index 0
    FEFaceEvaluation<dim, degree_finite_element> fe_face_eval(*system_matrix.get_matrix_free());
    for (unsigned int face = 0; face < system_matrix.get_matrix_free()->n_boundary_face_batches(); ++face)
    {
      fe_face_eval.reinit(face);

      for (const unsigned int q : fe_face_eval.quadrature_point_indices())
      {
        if (fe_face_eval.boundary_id() == 1)
          fe_face_eval.submit_value(neumannBC1.value(fe_face_eval.quadrature_point(q)), q);
        else if (fe_face_eval.boundary_id() == 3)
          fe_face_eval.submit_value(neumannBC2.value(fe_face_eval.quadrature_point(q)), q);
      }

      fe_face_eval.integrate_scatter(EvaluationFlags::values, system_rhs);
    }

    // Send the contributions to the respective owner of the dof
    system_rhs.compress(VectorOperation::add);

    setup_time += time.wall_time();
    time_details << "Assemble right hand side   (CPU/wall) " << time.cpu_time()
                 << "s/" << time.wall_time() << 's' << std::endl;
  }

  template <int dim>
  void DTRProblem<dim>::solve()
  {
    Timer time;
    // Start with the setup of the transfer using MGTransferMatrixFree that does the interpolation
    MGTransferMatrixFree<dim, float> mg_transfer(mg_constrained_dofs);
    mg_transfer.build(dof_handler);
    setup_time += time.wall_time();
    time_details << "MG build transfer time     (CPU/wall) " << time.cpu_time()
                 << "s/" << time.wall_time() << "s\n";
    time.restart();

    // Setup the Chebyshev iteration smoother
    using SmootherType =
        PreconditionChebyshev<LevelMatrixType, LinearAlgebra::distributed::Vector<float>>;
    mg::SmootherRelaxation<SmootherType, LinearAlgebra::distributed::Vector<float>>
        mg_smoother;
    MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
    smoother_data.resize(0, triangulation.n_global_levels() - 1);
    for (unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
    {
      if (level > 0)
      {
        // Use Chebyshev as smoother on all levels except the coarser one
        smoother_data[level].smoothing_range = 15.; // = factor of residual reduction
        smoother_data[level].degree = 5;
        smoother_data[level].eig_cg_n_iterations = 10;
      }
      else
      {
        // Use Chebyshev as solver on the coarser level
        // (i.e the number of iterations is internally choosen, setted on the second line)
        smoother_data[0].smoothing_range = 1e-3; // = relative tolerance
        smoother_data[0].degree = numbers::invalid_unsigned_int;
        smoother_data[0].eig_cg_n_iterations = mg_matrices[0].m();
      }
      // Compute the Jacobi preconditioner
      mg_matrices[level].compute_diagonal();
      smoother_data[level].preconditioner = mg_matrices[level].get_matrix_diagonal_inverse();
    }
    mg_smoother.initialize(mg_matrices, smoother_data);

    MGCoarseGridApplySmoother<LinearAlgebra::distributed::Vector<float>> mg_coarse;
    mg_coarse.initialize(mg_smoother);

    mg::Matrix<LinearAlgebra::distributed::Vector<float>> mg_matrix(mg_matrices);

    MGLevelObject<MatrixFreeOperators::MGInterfaceOperator<LevelMatrixType>>
        mg_interface_matrices;
    mg_interface_matrices.resize(0, triangulation.n_global_levels() - 1);

    // Initialize the interface matrices for each level
    for (unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
      mg_interface_matrices[level].initialize(mg_matrices[level]);
    mg::Matrix<LinearAlgebra::distributed::Vector<float>> mg_interface(mg_interface_matrices);

    Multigrid<LinearAlgebra::distributed::Vector<float>> mg(
        mg_matrix, mg_coarse, mg_transfer, mg_smoother, mg_smoother);
    mg.set_edge_matrices(mg_interface, mg_interface);

    PreconditionMG<dim,
                   LinearAlgebra::distributed::Vector<float>,
                   MGTransferMatrixFree<dim, float>>
        preconditioner(dof_handler, mg, mg_transfer);

    // Setup the solver
    SolverControl solver_control(100, 1e-12 * system_rhs.l2_norm());
    SolverCG<LinearAlgebra::distributed::Vector<double>> solver(solver_control);

    time.reset();
    time.start();

    // Set the correct constrained values to zero in the solution vector
    constraints.set_zero(solution);
    try
    {
      solver.solve(system_matrix, solution, system_rhs, preconditioner);
    }
    catch (std::exception &e)
    {
      Assert(false, ExcMessage(e.what()));
    }

    constraints.distribute(solution);

    // Add the lifting to the solution to set the correct inhomogeneous Dirichlet BC
    solution += lifting;

    pcout << "Time solve (" << solver_control.last_step() << " iterations)"
          << (solver_control.last_step() < 10 ? "  " : " ") << "(CPU/wall) "
          << time.cpu_time() << "s/" << time.wall_time() << "s\n";
  }

  template <int dim>
  void DTRProblem<dim>::output_results(const unsigned int cycle) const
  {
    Timer time;

    // Disable the output for too large meshes
    if (triangulation.n_global_active_cells() > 1000000)
      return;

    DataOut<dim> data_out;

    solution.update_ghost_values();
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution");
    data_out.build_patches(mapping);

    DataOutBase::VtkFlags flags;
    flags.compression_level = DataOutBase::CompressionLevel::best_speed;
    data_out.set_flags(flags);
    data_out.write_vtu_with_pvtu_record(
        output_dir, "solution", cycle, MPI_COMM_WORLD, 3);

    time_details << "Time write output          (CPU/wall) " << time.cpu_time()
                 << "s/" << time.wall_time() << "s\n";
  }

  template <int dim>
  void DTRProblem<dim>::run(unsigned int n_initial_refinements, unsigned int n_cycles)
  {
    // Print processor vectorization, MPI and multi-threading details
    {
      const unsigned int n_vect_doubles = VectorizedArray<double>::size();
      const unsigned int n_vect_bits = 8 * sizeof(double) * n_vect_doubles;

      pcout << "Vectorization over " << n_vect_doubles
            << " doubles = " << n_vect_bits << " bits ("
            << Utilities::System::get_current_vectorization_level() << ')'
            << std::endl;
      const unsigned int n_ranks = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
      pcout << "Running with " << n_ranks << " MPI process"
            << (n_ranks > 1 ? "es" : "") << ", element " << fe.get_name()
            << std::endl;
      pcout << "Using " << MultithreadInfo::n_threads() << " threads per process"
            << std::endl
            << std::endl;
    }

    // must compute at least a solution
    Assert(n_cycles > dim, ExcMessage("The number of cycles must be at least dim + 1"));


    for (unsigned int cycle = 0; cycle < n_cycles - dim; ++cycle)
    {
      pcout << "Cycle " << cycle << std::endl;

      if (cycle == 0)
      {
        // Generate the cube grid with bound index assignment
        GridGenerator::hyper_cube(triangulation, 0., 1., true);
        triangulation.refine_global(n_initial_refinements - dim);
      }

      triangulation.refine_global(1);
      setup_system();
      assemble_rhs();
      solve();
      output_results(cycle);
      pcout << std::endl;
    };
  }

  template <int dim>
  double DTRProblem<dim>::compute_error(const VectorTools::NormType &norm_type) const
  {
    solution.update_ghost_values();
    // First we compute the norm on each element, and store it in a vector.
    // To make sure we are accurate enough, we use a quadrature formula with one
    // node more than what we used in assembly
    Vector<double> error_per_cell(triangulation.n_active_cells());
    VectorTools::integrate_difference(MappingQ1<dim>(),
                                      dof_handler,
                                      solution,
                                      ExactSolution(),
                                      error_per_cell,
                                      QGauss<dim>(fe.degree + 2),
                                      norm_type);

    // Then, we add out all the cells
    const double error = std::sqrt(Utilities::MPI::sum(error_per_cell.norm_sqr(), MPI_COMM_WORLD));

    return error;
  }
}