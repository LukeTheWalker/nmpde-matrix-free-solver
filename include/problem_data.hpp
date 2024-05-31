#include <deal.II/base/function.h>

namespace problem_data
{
  using namespace dealii;

  /**
   * @brief Boundary conditions: 'D' for Dirichlet and 'N' for Neumann. Order: left, right, bottom, top.
   */
  const char bcs[4] = {'D', 'N', 'D', 'N'};

  /**
   * @brief Diffusion coefficient function.
   *
   * This class defines a constant diffusion coefficient for a PDE.
   * 
   * @tparam dim Dimension of the problem.
   */
  template <int dim>
  class DiffusionCoefficient : public Function<dim>
  {
  public:
    /**
     * @brief Returns the value of the diffusion coefficient at a given point.
     *
     * @param p Point at which the coefficient is evaluated.
     * @param component Component index.
     * @return Coefficient value at point p.
     */
    virtual double value(const Point<dim> &p, const unsigned int component = 0) const override
    {
      return value<double>(p, component);
    }

    /**
     * @brief Templated version of value() for different number types.
     *
     * @tparam number Number type (e.g., double, vectorArray).
     * @param p Point at which the coefficient is evaluated.
     * @param component Component index.
     * @return Coefficient value at point p.
     */
    template <typename number>
    number value(const Point<dim, number> & /*p*/, const unsigned int /*component*/ = 0) const
    {
      return 1.;
    }
  };

  /**
   * @brief Transport coefficient function.
   *
   * This class defines a constant transport coefficient for a PDE.
   * 
   * @tparam dim Dimension of the problem.
   */
  template <int dim>
  class TransportCoefficient : public Function<dim>
  {
  public:
    /**
     * @brief Returns the vector value of the transport coefficient at a given point.
     *
     * @param p Point at which the coefficient is evaluated.
     * @param values Vector to store the coefficient values.
     */
    virtual void vector_value(const Point<dim> &p, Vector<double> &values) const override
    {
      return vector_value<double>(p, values);
    }

    /**
     * @brief Templated version of vector_value() for different number types.
     *
     * @tparam number Number type (e.g., double, vectorArray).
     * @param p Point at which the coefficient is evaluated.
     * @param values Vector to store the coefficient values.
     */
    template <typename number>
    void vector_value(const Point<dim, number> & /*p*/, Vector<number> &values) const
    {
      values[0] = 1.;
      values[1] = 0.;
    }

    /**
     * @brief Returns the value of the transport coefficient for a specific component.
     *
     * @param p Point at which the coefficient is evaluated.
     * @param component Component index.
     * @return Coefficient value at point p for the specified component.
     */
    virtual double value(const Point<dim> &p, const unsigned int component = 0) const override
    {
      return value<double>(p, component);
    }

    /**
     * @brief Templated version of value() for different number types.
     *
     * @tparam number Number type (e.g., double, vectorArray).
     * @param p Point at which the coefficient is evaluated.
     * @param component Component index.
     * @return Coefficient value at point p for the specified component.
     */
    template <typename number>
    number value(const Point<dim, number> & /*p*/, const unsigned int component = 0) const
    {
      if (component == 0)
        return 1.;
      else
        return 0.;
    }

    /**
     * @brief Returns the tensor value of the transport coefficient at a given point.
     *
     * @param p Point at which the coefficient is evaluated.
     * @param values Tensor to store the coefficient values.
     */
    template <typename number>
    void tensor_value(const Point<dim, number> & /*p*/, Tensor<1, dim, number> &values) const
    {
      values[0] = 1.;
      values[1] = 0.;
    }
  };

  /**
   * @brief Reaction coefficient function.
   *
   * This class defines a constant reaction coefficient for a PDE.
   * 
   * @tparam dim Dimension of the problem.
   */
  template <int dim>
  class ReactionCoefficient : public Function<dim>
  {
  public:
    /**
     * @brief Returns the value of the reaction coefficient at a given point.
     *
     * @param p Point at which the coefficient is evaluated.
     * @param component Component index.
     * @return Coefficient value at point p.
     */
    virtual double value(const Point<dim> &p, const unsigned int component = 0) const override
    {
      return value<double>(p, component);
    }

    /**
     * @brief Templated version of value() for different number types.
     *
     * @tparam number Number type (e.g., double, vectorArray).
     * @param p Point at which the coefficient is evaluated.
     * @param component Component index.
     * @return Coefficient value at point p.
     */
    template <typename number>
    number value(const Point<dim, number> & /*p*/, const unsigned int /*component*/ = 0) const
    {
      return 1.;
    }
  };

  /**
   * @brief Forcing term function.
   *
   * This class defines a forcing term for a PDE, representing an external source or sink.
   * 
   * @tparam dim Dimension of the problem.
   */
  template <int dim>
  class ForcingTerm : public Function<dim>
  {
  public:
    /**
     * @brief Returns the value of the forcing term at a given point.
     *
     * @param p Point at which the term is evaluated.
     * @param component Component index.
     * @return Forcing term value at point p.
     */
    virtual double value(const Point<dim> &p, const unsigned int component = 0) const override
    {
      return value<double>(p, component);
    }

    /**
     * @brief Templated version of value() for different number types.
     *
     * @tparam number Number type (e.g., double, vectorArray).
     * @param p Point at which the term is evaluated.
     * @param component Component index.
     * @return Forcing term value at point p.
     */
    template <typename number>
    number value(const Point<dim, number> &p, const unsigned int /*component*/ = 0) const
    {
      return number(1.) - number(2.) * exp(p[0]);
    }
  };

  /**
   * @brief Dirichlet boundary condition (left and bottom) function.
   *
   * This class defines Dirichlet boundary conditions on the left and bottom boundaries.
   * 
   * @tparam dim Dimension of the problem.
   */
  template <int dim>
  class DirichletBC1 : public Function<dim>
  {
  public:
    /**
     * @brief Returns the value of the Dirichlet BC at a given point.
     *
     * @param p Point at which the BC is evaluated.
     * @param component Component index.
     * @return BC value at point p.
     */
    virtual double value(const Point<dim> &p, const unsigned int component = 0) const override
    {
      return value<double>(p, component);
    }

    /**
     * @brief Templated version of value() for different number types.
     *
     * @tparam number Number type (e.g., double, vectorArray).
     * @param p Point at which the BC is evaluated.
     * @param component Component index.
     * @return BC value at point p.
     */
    template <typename number>
    double value(const Point<dim, number> &p, const unsigned int /*component*/ = 0) const
    {
      return number(2.) * exp(p[1]) - number(1.);
    }
  };

  /**
   * @brief Dirichlet boundary condition (right and top) function.
   *
   * This class defines Dirichlet boundary conditions on the right and top boundaries.
   * 
   * @tparam dim Dimension of the problem.
   */
  template <int dim>
  class DirichletBC2 : public Function<dim>
  {
  public:
    /**
     * @brief Returns the value of the Dirichlet BC at a given point.
     *
     * @param p Point at which the BC is evaluated.
     * @param component Component index.
     * @return BC value at point p.
     */
    virtual double value(const Point<dim> &p, const unsigned int component = 0) const override
    {
      return value<double>(p, component);
    }

    /**
     * @brief Templated version of value() for different number types.
     *
     * @tparam number Number type (e.g., double, vectorArray).
     * @param p Point at which the BC is evaluated.
     * @param component Component index.
     * @return BC value at point p.
     */
    template <typename number>
    double value(const Point<dim, number> &p, const unsigned int /*component*/ = 0) const
    {
      return number(2.) * exp(p[0]) - number(1.);
    }
  };

  /**
   * @brief Neumann boundary condition (right and top) function.
   *
   * This class defines Neumann boundary conditions on the right and top boundaries.
   * 
   * @tparam dim Dimension of the problem.
   */
  template <int dim>
  class NeumannBC1 : public Function<dim>
  {
  public:
    /**
     * @brief Returns the value of the Neumann BC at a given point.
     *
     * @param p Point at which the BC is evaluated.
     * @param component Component index.
     * @return BC value at point p.
     */
    virtual double value(const Point<dim> &p, const unsigned int component = 0) const override
    {
      return value<double>(p, component);
    }

    /**
     * @brief Templated version of value() for different number types.
     *
     * @tparam number Number type (e.g., double, vectorArray).
     * @param p Point at which the BC is evaluated.
     * @param component Component index.
     * @return BC value at point p.
     */
    template <typename number>
    number value(const Point<dim, number> &p, const unsigned int /*component*/ = 0) const
    {
      return number(2.) * exp(p[0]) * (number(2.) * exp(p[1]) - number(1.));
    }
  };

  /**
   * @brief Neumann boundary condition (left and bottom) function.
   *
   * This class defines Neumann boundary conditions on the left and bottom boundaries.
   * 
   * @tparam dim Dimension of the problem.
   */
  template <int dim>
  class NeumannBC2 : public Function<dim>
  {
  public:
    /**
     * @brief Returns the value of the Neumann BC at a given point.
     *
     * @param p Point at which the BC is evaluated.
     * @param component Component index.
     * @return BC value at point p.
     */
    virtual double value(const Point<dim> &p, const unsigned int component = 0) const override
    {
      return value<double>(p, component);
    }

    /**
     * @brief Templated version of value() for different number types.
     *
     * @tparam number Number type (e.g., double, vectorArray).
     * @param p Point at which the BC is evaluated.
     * @param component Component index.
     * @return BC value at point p.
     */
    template <typename number>
    number value(const Point<dim, number> &p, const unsigned int /*component*/ = 0) const
    {
      return number(2.) * exp(p[1]) * (number(2.) * exp(p[0]) - number(1.));
    }
  };

  /**
   * @brief Exact solution function.
   *
   * This class defines the exact solution for comparison with the numerical solution.
   * 
   * @tparam dim Dimension of the problem.
   */
  template <int dim>
  class ExactSolution : public Function<dim>
  {
  public:
    /**
     * @brief Returns the exact solution value at a given point.
     *
     * @param p Point at which the solution is evaluated.
     * @param component Component index.
     * @return Exact solution value at point p.
     */
    virtual double value(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
    {
      return (2. * exp(p[0]) - 1.) * (2. * exp(p[1]) - 1.);
    }

    /**
     * @brief Returns the gradient of the exact solution at a given point.
     *
     * @param p Point at which the gradient is evaluated.
     * @param component Component index.
     * @return Gradient of the exact solution at point p.
     */
    virtual Tensor<1, dim> gradient(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
    {
      Tensor<1, dim> result;

      result[0] = 2. * exp(p[0]) * (2. * exp(p[1]) - 1.);
      result[1] = 2. * exp(p[1]) * (2. * exp(p[0]) - 1.);

      return result;
    }
  };
} // namespace problem_data
