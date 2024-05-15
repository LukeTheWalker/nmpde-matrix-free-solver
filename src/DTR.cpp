#include "DTR.hpp"

const char bcs[4] = {'Z', 'N', 'Z', 'N'};

void DTR::setup()
{
  pcout << "===============================================" << std::endl;

  // Create the mesh.
  {
    pcout << "Initializing the mesh" << std::endl;

    // First we read the mesh from file into a serial (i.e. not parallel)
    // triangulation.
    Triangulation<dim> mesh_serial;

    {
      GridIn<dim> grid_in;
      grid_in.attach_triangulation(mesh_serial);

      std::ifstream grid_in_file(mesh_file_name);
      grid_in.read_msh(grid_in_file);
    }

    // Then, we copy the triangulation into the parallel one.
    {
      GridTools::partition_triangulation(mpi_size, mesh_serial);
      const auto construction_data = TriangulationDescription::Utilities::
        create_description_from_triangulation(mesh_serial, MPI_COMM_WORLD);
      mesh.create_triangulation(construction_data);
    }

    // Notice that we write here the number of *global* active cells (across all
    // processes).
    pcout << "  Number of elements = " << mesh.n_global_active_cells()
          << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the finite element space. This is the same as in serial codes.
  {
    pcout << "Initializing the finite element space" << std::endl;

    fe = std::make_unique<FE_SimplexP<dim>>(r);

    pcout << "  Degree                     = " << fe->degree << std::endl;
    pcout << "  DoFs per cell              = " << fe->dofs_per_cell
          << std::endl;

    quadrature = std::make_unique<QGaussSimplex<dim>>(r + 1);

    pcout << "  Quadrature points per cell = " << quadrature->size()
          << std::endl;

    quadrature_boundary = std::make_unique<QGaussSimplex<dim - 1>>(r + 1);

    pcout << "  Quadrature points per boundary cell = "
              << quadrature_boundary->size() << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the DoF handler.
  {
    pcout << "Initializing the DoF handler" << std::endl;

    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fe);

    // We retrieve the set of locally owned DoFs,whose indices are global, 
    // which will be useful when initializing linear algebra classes.
    locally_owned_dofs = dof_handler.locally_owned_dofs();

    pcout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the linear system.
  {
    pcout << "Initializing the linear system" << std::endl;

    pcout << "  Initializing the sparsity pattern" << std::endl;

    // To initialize the sparsity pattern, we use Trilinos' class, that manages
    // some of the inter-process communication.
    TrilinosWrappers::SparsityPattern sparsity(locally_owned_dofs,
                                               MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler, sparsity);

    // After initialization, we need to call compress, so that all process
    // retrieve the information they need for the rows they own (i.e. the rows
    // corresponding to locally owned DoFs).
    // It handles the kind of cache that allow the writing of the computed values,
    // that single processors couldn't write due to a lack of information (the 
    // corresponding dofs were assigned to another processor).
    sparsity.compress();

    // Then, we use the sparsity pattern to initialize the system matrix. Since
    // the sparsity pattern is partitioned by row, so will the matrix.
    pcout << "  Initializing the system matrix" << std::endl;
    system_matrix.reinit(sparsity);

    // Finally, we initialize the right-hand side and solution vectors.
    pcout << "  Initializing the system right-hand side" << std::endl;
    system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    pcout << "  Initializing the solution vector" << std::endl;
    solution.reinit(locally_owned_dofs, MPI_COMM_WORLD);
  }
}

void DTR::assemble()
{
  pcout << "===============================================" << std::endl;

  pcout << "  Assembling the linear system" << std::endl;

  // Number of local DoFs for each element.
  const unsigned int dofs_per_cell = fe->dofs_per_cell;

  // Number of quadrature points for each element.
  const unsigned int n_q = quadrature->size();

  // FEValues instance. This object allows to compute basis functions, their
  // derivatives, the reference-to-current element mapping and its
  // derivatives on all quadrature points of all elements.
  FEValues<dim> fe_values(
      *fe,
      *quadrature,
      // Here we specify what quantities we need FEValues to compute on
      // quadrature points. For our test, we need:
      // - the values of shape functions (update_values);
      // - the derivative of shape functions (update_gradients);
      // - the position of quadrature points (update_quadrature_points);
      // - the product J_c(x_q)*w_q (update_JxW_values).
      update_values | update_gradients | update_quadrature_points |
          update_JxW_values);

  // Since we need to compute integrals on the boundary for Neumann conditions,
  // we also need a FEValues object to compute quantities on boundary edges
  // (faces).
  FEFaceValues<dim> fe_values_boundary(*fe,
                                       *quadrature_boundary,
                                       update_values |
                                           update_quadrature_points |
                                           update_JxW_values);

  // Local matrix and right-hand side vector. We will overwrite them for
  // each element within the loop.
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);

  // We will use this vector to store the global indices of the DoFs of the
  // current element within the loop.
  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  // Reset the global matrix and vector, just in case.
  system_matrix = 0.0;
  system_rhs = 0.0;

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    // If current cell is not owned locally, we skip it.
      if (!cell->is_locally_owned())
        continue;

    // On all other cells (which are owned by current process) reinitialize the FEValues
    // object on current element. This precomputes all the quantities we requested when 
    // constructing FEValues (see the update_* flags above) for all quadrature nodes of
    // the current cell.
    fe_values.reinit(cell);

    // We reset the cell matrix and vector (discarding any leftovers from
    // previous element).
    cell_matrix = 0.0;
    cell_rhs = 0.0;

    for (unsigned int q = 0; q < n_q; ++q)
    {
      // Here we assemble the local contribution for current cell and
      // current quadrature point, filling the local matrix and vector.

      // Here we iterate over *local* DoF indices.

      Vector<double> b_loc(dim);

      transport_coefficient.vector_value(fe_values.quadrature_point(q), b_loc);

      Tensor<1, dim> b_loc_tensor;
      for (unsigned int i = 0; i < dim; ++i)
        b_loc_tensor[i] = b_loc[i];

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        for (unsigned int j = 0; j < dofs_per_cell; ++j)
        {
          // Diffusion term.
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
          // Reaction term.
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

    // If the cell is adjacent to the boundary...
    if (cell->at_boundary())
      {
        // ...we loop over its edges (referred to as faces in the deal.II
        // jargon).
        for (unsigned int face_number = 0; face_number < cell->n_faces();
             ++face_number)
          {
            // If current face lies on the boundary, and its boundary ID (or
            // tag) is that of one of the Neumann boundaries, we assemble the
            // boundary integral.
            if (cell->face(face_number)->at_boundary() &&
                bcs[cell->face(face_number)->boundary_id()] == 'N')

              {
                fe_values_boundary.reinit(cell, face_number);

                for (unsigned int q = 0; q < quadrature_boundary->size(); ++q)
                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    if (cell->face(face_number)->boundary_id() == 1){
                      cell_rhs(i) +=
                        neumannBC1.value(
                          fe_values_boundary.quadrature_point(q)) * // h(xq)
                        fe_values_boundary.shape_value(i, q) *      // v(xq)
                        fe_values_boundary.JxW(q);                  // Jq wq
                    }
                    else if (cell->face(face_number)->boundary_id() == 3){
                      cell_rhs(i) +=
                        neumannBC2.value(
                          fe_values_boundary.quadrature_point(q)) * // h(xq)
                        fe_values_boundary.shape_value(i, q) *      // v(xq)
                        fe_values_boundary.JxW(q);                  // Jq wq
                    }
              }
          }
      }

    // At this point the local matrix and vector are constructed: we
    // need to sum them into the global matrix and vector. To this end,
    // we need to retrieve the global indices of the DoFs of current
    // cell.
    cell->get_dof_indices(dof_indices);

    // Then, we add the local matrix and vector into the corresponding
    // positions of the global matrix and vector.
    system_matrix.add(dof_indices, cell_matrix);
    system_rhs.add(dof_indices, cell_rhs);
  }

  // Each process might have written to some rows it does not own (for instance,
  // if it owns elements that are adjacent to elements owned by some other
  // process). Therefore, at the end of the assembly, processes need to exchange
  // information: the compress method allows to do this.
  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);

  // Boundary conditions.
  {
    // We construct a map that stores, for each DoF corresponding to a
    // Dirichlet condition, the corresponding value. E.g., if the Dirichlet
    // condition is u_i = b, the map will contain the pair (i, b).
    std::map<types::global_dof_index, double> boundary_values;

    // Then, we build a map that, for each boundary tag, stores the
    // corresponding boundary function.

    std::map<types::boundary_id, const Function<dim> *> boundary_functions;

    for (unsigned int i = 0; i < 4; ++i)
      if (bcs[i] == 'D')
        boundary_functions[i] = &dirichletBC;

    VectorTools::interpolate_boundary_values(dof_handler,
                                             boundary_functions,
                                             boundary_values);

    boundary_functions.clear();
    Functions::ZeroFunction<dim> zero_function(dim + 1);
    for (unsigned int i = 0; i < 4; ++i)
      if (bcs[i] == 'Z')
        boundary_functions[i] = &zero_function;

    VectorTools::interpolate_boundary_values(dof_handler,
                                             boundary_functions,
                                             boundary_values);
    // Finally, we modify the linear system to apply the boundary
    // conditions. This replaces the equations for the boundary DoFs with
    // the corresponding u_i = 0 equations.
    MatrixTools::apply_boundary_values(
        boundary_values, system_matrix, solution, system_rhs, true);
  }
}

void DTR::solve()
{
  pcout << "===============================================" << std::endl;

  // Here we specify the maximum number of iterations of the iterative solver,
  // and its tolerance.
  SolverControl solver_control(10000, 1e-6 * system_rhs.l2_norm());

  // The linear solver is basically the same as in serial, in terms of
  // interface: we only have to use appropriate classes, compatible with
  // Trilinos linear algebra.
  SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);

  TrilinosWrappers::PreconditionSSOR preconditioner;
  preconditioner.initialize(
    system_matrix, TrilinosWrappers::PreconditionSSOR::AdditionalData(1.0));

  pcout << "  Solving the linear system" << std::endl;
  solver.solve(system_matrix, solution, system_rhs, preconditioner);
  pcout << "  " << solver_control.last_step() << " CG iterations" << std::endl;
}

void DTR::output() const
{
  pcout << "===============================================" << std::endl;

  // Union of owned and ghost dofs
  IndexSet locally_relevant_dofs;
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

  // To correctly export the solution, each process needs to know the solution
  // DoFs it owns, and the ones corresponding to elements adjacent to the ones
  // it owns (the locally relevant DoFs, or ghosts). We create a vector to store
  // them.
  TrilinosWrappers::MPI::Vector solution_ghost(locally_owned_dofs,
                                               locally_relevant_dofs,
                                               MPI_COMM_WORLD);

  // This performs the necessary communication so that the locally relevant DoFs
  // are received from other processes and stored inside solution_ghost.
  solution_ghost = solution;

  // Then, we build and fill the DataOut class as usual.
  DataOut<dim> data_out;
  data_out.add_data_vector(dof_handler, solution_ghost, "solution");

  // We also add a vector to represent the parallel partitioning of the mesh.
  std::vector<unsigned int> partition_int(mesh.n_active_cells());
  GridTools::get_subdomain_association(mesh, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();

  const std::filesystem::path mesh_path(mesh_file_name);
  const std::string output_file_name = "output-" + mesh_path.stem().string();

  // Finally, we need to write in a format that supports parallel output. This
  // can be achieved in multiple ways (e.g. XDMF/H5). We choose VTU/PVTU files,
  // because the interface is nice and it is quite robust.
  data_out.write_vtu_with_pvtu_record("./",
                                      output_file_name,
                                      0,
                                      MPI_COMM_WORLD);

  pcout << "Output written to " << output_file_name << std::endl;

  pcout << "===============================================" << std::endl;
}

double
DTR::compute_error(const VectorTools::NormType &norm_type) const
{
  FE_SimplexP<dim> fe_linear(1);
  MappingFE mapping(fe_linear);

  // The error is an integral, and we approximate that integral using a
  // quadrature formula. To make sure we are accurate enough, we use a
  // quadrature formula with one node more than what we used in assembly.
  const QGaussSimplex<dim> quadrature_error(r + 2);

  // First we compute the norm on each element, and store it in a vector.
  Vector<double> error_per_cell(mesh.n_active_cells());
  VectorTools::integrate_difference(mapping,
                                    dof_handler,
                                    solution,
                                    ExactSolution(),
                                    error_per_cell,
                                    quadrature_error,
                                    norm_type);

  // Then, we add out all the cells.
  const double error =
      VectorTools::compute_global_error(mesh, error_per_cell, norm_type);

  return error;
}