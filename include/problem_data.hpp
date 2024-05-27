#include <deal.II/base/function.h>

namespace problem_data
{
  using namespace dealii;

  const char bcs[4] = {'D', 'N', 'D', 'N'}; // left, right, bottom, top

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
      values[1] = 0.;
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
        return 0.;
    }

    template <typename number>
    void tensor_value(const Point<dim, number> & /*p*/, Tensor<1, dim, number> &values) const
    {
      values[0] = 1.;
      values[1] = 0.;
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
      return number(1.) - number(2.) * exp(p[0]);
    }
  };

  template <int dim>
  class DirichletBC1 : public Function<dim>
  {
public:
    virtual double value(const Point<dim> &p, const unsigned int component = 0) const override
    {
      return value<double>(p, component);
    }
    template <typename number>
    double value(const Point<dim, number> &p, const unsigned int /*component*/ = 0) const
    {
      return number(2.) * exp(p[1]) - number(1.);
    }
  };

  template <int dim>
  class DirichletBC2 : public Function<dim>
  {
public:
    virtual double value(const Point<dim> &p, const unsigned int component = 0) const override
    {
      return value<double>(p, component);
    }
    template <typename number>
    double value(const Point<dim, number> &p, const unsigned int /*component*/ = 0) const
    {
      return number(2.) * exp(p[0]) - number(1.);
    }
  };

  template <int dim>
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
      return number(2.) * exp(p[0]) * (number(2.) * exp(p[1]) - number(1.));
    }
  };

  template <int dim>
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
      return number(2.) * exp(p[1]) * (number(2.) * exp(p[0]) - number(1.));
    }
  };

  template <int dim>
  class ExactSolution : public Function<dim>
  {
public:
    virtual double value(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
    {
      return (2. * exp(p[0]) - 1.) * (2. * exp(p[1]) - 1.);
    }

    virtual Tensor<1, dim> gradient(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
    {
      Tensor<1, dim> result;

      result[0] = 2. * exp(p[0]) * (2. * exp(p[1]) - 1.);
      result[1] = 2. * exp(p[1]) * (2. * exp(p[0]) - 1.);

      return result;
    }
  };
} // namespace problem_data