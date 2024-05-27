#pragma once

#include "DTR_mg.hpp"

namespace DTR_mg
{

  template <int dim>
  DTRProblem<dim>::DTRProblem(unsigned int degree)
      : mpi_communicator(MPI_COMM_WORLD),
        pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)),
        time_details(std::cout, false),
        triangulation(mpi_communicator,
                      Triangulation<dim>::limit_level_difference_at_vertices,
                      parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
        mapping(), fe(degree), dof_handler(triangulation)
  {
  }

  template <int dim>
  DTRProblem<dim>::DTRProblem(unsigned int degree, std::ofstream& dimension_time_file)
      : mpi_communicator(MPI_COMM_WORLD),
        pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)),
        time_details(dimension_time_file, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)),
        triangulation(mpi_communicator,
                      Triangulation<dim>::limit_level_difference_at_vertices,
                      parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
        mapping(), fe(degree), dof_handler(triangulation)
  {
  }

  template <int dim>
  void DTRProblem<dim>::setup_system()
  {
    Timer time;
    setup_time = 0;

    dof_handler.distribute_dofs(fe);

    locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);
    locally_owned_dofs = dof_handler.locally_owned_dofs();

    pcout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
    time_details << dof_handler.n_dofs() << ',';

    solution.reinit(locally_owned_dofs, mpi_communicator);
    right_hand_side.reinit(locally_owned_dofs, mpi_communicator);

    constraints.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    Functions::ZeroFunction<dim> homogeneous_dirichlet_bc;

    std::map<types::boundary_id, const Function<dim> *> dirichlet_boundary_functions;

    dirichlet_boundary_functions[0] = &dirichletBC1;
    dirichlet_boundary_functions[2] = &dirichletBC2;

    VectorTools::interpolate_boundary_values(dof_handler,
                                             dirichlet_boundary_functions,
                                             constraints);
    constraints.close();

    TrilinosWrappers::SparsityPattern dsp(locally_owned_dofs,
                                          locally_owned_dofs,
                                          locally_relevant_dofs,
                                          mpi_communicator);
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);
    dsp.compress();
    system_matrix.reinit(dsp);

    setup_time += time.wall_time();
  }

  template <int dim>
  void DTRProblem<dim>::setup_multigrid()
  {
    Timer time;
    dof_handler.distribute_mg_dofs();

    mg_constrained_dofs.clear();
    mg_constrained_dofs.initialize(dof_handler);

    const std::set<types::boundary_id> dirichlet_boundary_ids = {0, 2};
    mg_constrained_dofs.make_zero_boundary_constraints(dof_handler, dirichlet_boundary_ids);

    const unsigned int n_levels = triangulation.n_global_levels();

    mg_matrix.resize(0, n_levels - 1);
    mg_matrix.clear_elements();
    mg_interface_in.resize(0, n_levels - 1);
    mg_interface_in.clear_elements();

    for (unsigned int level = 0; level < n_levels; ++level)
    {
      const IndexSet dof_set =
          DoFTools::extract_locally_relevant_level_dofs(dof_handler, level);
      {
        TrilinosWrappers::SparsityPattern dsp(
            dof_handler.locally_owned_mg_dofs(level),
            dof_handler.locally_owned_mg_dofs(level),
            dof_set,
            mpi_communicator);
        MGTools::make_sparsity_pattern(dof_handler, dsp, level);

        dsp.compress();
        mg_matrix[level].reinit(dsp);
      }
      {
        TrilinosWrappers::SparsityPattern dsp(
            dof_handler.locally_owned_mg_dofs(level),
            dof_handler.locally_owned_mg_dofs(level),
            dof_set,
            mpi_communicator);

        MGTools::make_interface_sparsity_pattern(dof_handler,
                                                 mg_constrained_dofs,
                                                 dsp,
                                                 level);
        dsp.compress();
        mg_interface_in[level].reinit(dsp);
      }
    }

    setup_time += time.wall_time();
  }

  template <int dim>
  void DTRProblem<dim>::assemble_system()
  {
    Timer time;
    const QGauss<dim> quadrature_formula(fe.degree + 1);
    const QGauss<dim - 1> quadrature_boundary(fe.degree + 1);

    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                                update_quadrature_points | update_JxW_values);
    FEFaceValues<dim> fe_values_boundary(fe,
                                         quadrature_boundary,
                                         update_values | update_quadrature_points |
                                             update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature_formula.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<double> rhs_values(n_q_points);

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
      {
        cell_matrix = 0.0;
        cell_rhs = 0.0;

        fe_values.reinit(cell);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
          Vector<double> b_loc(dim);
          transport_coefficient.vector_value(fe_values.quadrature_point(q), b_loc);
          Tensor<1, dim> b_loc_tensor;
          for (unsigned int i = 0; i < dim; ++i)
            b_loc_tensor[i] = b_loc[i];

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
            {
              cell_matrix(i, j) += diffusion_coefficient.value(
                                       fe_values.quadrature_point(q))         // mu(x)
                                   * fe_values.shape_grad(i, q)               // (I)
                                   * fe_values.shape_grad(j, q)               // (II)
                                   * fe_values.JxW(q);                        // (III)
              cell_matrix(i, j) += scalar_product(b_loc_tensor,               // b(x)
                                                  fe_values.shape_grad(j, q)) // (I)
                                   * fe_values.shape_value(i, q)              // (II)
                                   * fe_values.JxW(q);                        // (III)
              cell_matrix(i, j) += reaction_coefficient.value(
                                       fe_values.quadrature_point(q)) // sigma(x)
                                   * fe_values.shape_value(i, q)      // phi_i
                                   * fe_values.shape_value(j, q)      // phi_j
                                   * fe_values.JxW(q);                // dx
            }
            cell_rhs(i) += forcing_term.value(fe_values.quadrature_point(q)) *
                           fe_values.shape_value(i, q) * fe_values.JxW(q);
          }
        }

        if (cell->at_boundary())
        {
          for (unsigned int face_number = 0; face_number < cell->n_faces();
               ++face_number)
          {
            if (cell->face(face_number)->at_boundary() &&
                bcs[cell->face(face_number)->boundary_id()] == 'N')
            {
              fe_values_boundary.reinit(cell, face_number);

              for (unsigned int q = 0; q < quadrature_boundary.size(); ++q)
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  if (cell->face(face_number)->boundary_id() == 1)
                  {
                    cell_rhs(i) +=
                        neumannBC1.value(
                            fe_values_boundary.quadrature_point(q)) * // h(xq)
                        fe_values_boundary.shape_value(i, q) *        // v(xq)
                        fe_values_boundary.JxW(q);                    // Jq wq
                  }
                  else if (cell->face(face_number)->boundary_id() == 3)
                  {
                    cell_rhs(i) +=
                        neumannBC2.value(
                            fe_values_boundary.quadrature_point(q)) * // h(xq)
                        fe_values_boundary.shape_value(i, q) *        // v(xq)
                        fe_values_boundary.JxW(q);                    // Jq wq
                  }
            }
          }
        }
        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(cell_matrix,
                                               cell_rhs,
                                               local_dof_indices,
                                               system_matrix,
                                               right_hand_side);
      }
    system_matrix.compress(VectorOperation::add);
    right_hand_side.compress(VectorOperation::add);

    setup_time += time.wall_time();
  }

  template <int dim>
  void DTRProblem<dim>::assemble_multigrid()
  {
    Timer time;
    const QGauss<dim> quadrature_formula(fe.degree + 1);

    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                                update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature_formula.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<AffineConstraints<double>> boundary_constraints(
        triangulation.n_global_levels());
    for (unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
    {
      const IndexSet dof_set =
          DoFTools::extract_locally_relevant_level_dofs(dof_handler, level);
      boundary_constraints[level].reinit(dof_set);
      boundary_constraints[level].add_lines(
          mg_constrained_dofs.get_refinement_edge_indices(level));
      boundary_constraints[level].add_lines(
          mg_constrained_dofs.get_boundary_indices(level));

      boundary_constraints[level].close();
    }

    for (const auto &cell : dof_handler.cell_iterators())
      if (cell->level_subdomain_id() == triangulation.locally_owned_subdomain())
      {
        cell_matrix = 0;
        fe_values.reinit(cell);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
          Vector<double> b_loc(dim);

          transport_coefficient.vector_value(fe_values.quadrature_point(q), b_loc);

          Tensor<1, dim> b_loc_tensor;
          for (unsigned int i = 0; i < dim; ++i)
            b_loc_tensor[i] = b_loc[i];

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
            {
              cell_matrix(i, j) += diffusion_coefficient.value(
                                       fe_values.quadrature_point(q)) // mu(x)
                                   * fe_values.shape_grad(i, q)       // (I)
                                   * fe_values.shape_grad(j, q)       // (II)
                                   * fe_values.JxW(q);                // (III)

              // Transport term.
              cell_matrix(i, j) += scalar_product(b_loc_tensor,               // b(x)
                                                  fe_values.shape_grad(j, q)) // (I)
                                   * fe_values.shape_value(i, q)              // (II)
                                   * fe_values.JxW(q);                        // (III)
              // Diffusion term.
              cell_matrix(i, j) += reaction_coefficient.value(
                                       fe_values.quadrature_point(q)) // sigma(x)
                                   * fe_values.shape_value(i, q)      // phi_i
                                   * fe_values.shape_value(j, q)      // phi_j
                                   * fe_values.JxW(q);                // dx
            }
          }
        }
        cell->get_mg_dof_indices(local_dof_indices);

        boundary_constraints[cell->level()].distribute_local_to_global(
            cell_matrix, local_dof_indices, mg_matrix[cell->level()]);

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            if (mg_constrained_dofs.is_interface_matrix_entry(
                    cell->level(), local_dof_indices[i], local_dof_indices[j]))
              mg_interface_in[cell->level()].add(local_dof_indices[i],
                                                 local_dof_indices[j],
                                                 cell_matrix(i, j));
      }

    for (unsigned int i = 0; i < triangulation.n_global_levels(); ++i)
    {
      mg_matrix[i].compress(VectorOperation::add);
      mg_interface_in[i].compress(VectorOperation::add);
    }

    setup_time += time.wall_time();
  }

  template <int dim>
  void DTRProblem<dim>::solve()
  {
    Timer time;
    SolverControl solver_control(50000, 1e-10 * right_hand_side.l2_norm());

    solution = 0.;

    MGTransferPrebuilt<VectorType> mg_transfer(mg_constrained_dofs);
    mg_transfer.build(dof_handler);

    SolverControl coarse_solver_control(50000, 1e-12, false, false);
    SolverGMRES<VectorType> coarse_solver(coarse_solver_control);
    PreconditionIdentity identity;
    MGCoarseGridIterativeSolver<VectorType,
                                SolverGMRES<VectorType>,
                                MatrixType,
                                PreconditionIdentity>
      coarse_grid_solver(coarse_solver, mg_matrix[0], identity);

    using Smoother = LinearAlgebraTrilinos::MPI::PreconditionSSOR;
    MGSmootherPrecondition<MatrixType, Smoother, VectorType> smoother;

    smoother.initialize(mg_matrix, 1.);
    smoother.set_steps(3);

    mg::Matrix<VectorType> mg_m(mg_matrix);
    mg::Matrix<VectorType> mg_in(mg_interface_in);
    mg::Matrix<VectorType> mg_out(mg_interface_in);

    Multigrid<VectorType> mg(
        mg_m, coarse_grid_solver, mg_transfer, smoother, smoother);
    mg.set_edge_matrices(mg_out, mg_in);

    PreconditionMG<dim, VectorType, MGTransferPrebuilt<VectorType>>
        preconditioner(dof_handler, mg, mg_transfer);

    solution = 0.;

    SolverGMRES<VectorType> solver(solver_control);

    setup_time += time.wall_time();
    pcout << "Total setup time               (wall) " << setup_time << "s\n";
    time_details /*<< "Setup time"*/ <<Utilities::MPI::min_max_avg(setup_time, MPI_COMM_WORLD).avg << ",";

    time.reset();
    time.start();

    // Solve the linear system and distribute constraints.
    solver.solve(system_matrix,
                   solution,
                   right_hand_side,
                   preconditioner);

    constraints.distribute(solution);

    pcout << "Time solve (" << solver_control.last_step() << " iterations)"
          << (solver_control.last_step() < 10 ? "  " : " ") << "(CPU/wall) "
          << time.cpu_time() << "s/" << time.wall_time() << "s\n";
    pcout << "   Number of CG iterations:      " << solver_control.last_step()
          << std::endl;
    time_details /*<< "solve time"*/ << Utilities::MPI::min_max_avg(time.wall_time(), MPI_COMM_WORLD).avg << ",";
    time_details /*<< "iterations"*/ << solver_control.last_step() << std::endl;
  }

  template <int dim>
  void DTRProblem<dim>::output_results(const unsigned int cycle)
  {
    VectorType temp_solution;
    temp_solution.reinit(locally_owned_dofs,
                         locally_relevant_dofs,
                         mpi_communicator);
    temp_solution = solution;

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(temp_solution, "solution");
    data_out.build_patches();

    const std::string pvtu_filename = data_out.write_vtu_with_pvtu_record(
        output_dir, "solution", cycle, mpi_communicator, 2 /*n_digits*/, 1 /*n_groups*/);

    pcout << "   Wrote " << pvtu_filename << " in " << output_dir << std::endl;
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
      pcout << "Cycle " << cycle << ':' << std::endl;

      if (cycle == 0)
      {
        // Generate the cube grid with bound index assignment
        GridGenerator::hyper_cube(triangulation, 0., 1., true);
        triangulation.refine_global(n_initial_refinements - dim);
      }

      triangulation.refine_global(1);

      pcout << "   Number of active cells:       "
            << triangulation.n_global_active_cells();
      pcout << std::endl;

      setup_system();
      setup_multigrid();

      assemble_system();
      assemble_multigrid();

      solve();

      output_results(cycle);

    }
  }
}