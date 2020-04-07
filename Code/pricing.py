class Curve:
   """
   Mother class of a curve.
   Each derived class should implement the constructor and a method curve(date).
   """

   def __init__(self, **kwargs):
      """Constructor"""
      raise Exception('do not instantiate this class.')

   def __call__(self, date):
      """Value of the Curve at date."""
      return self.curve(date)


class EquityForwardCurve(Curve):
   def __init__(self, spot=None, reference=None, discounting_curve=None,
                repo_rates=None, repo_dates=None, dividend_rates=None, dividend_dates=None):
      pass

   def curve(self, date):
      pass


class DiscountingCurve(Curve):
   def __init__(self, reference=None, discounts=None, dates=None):
      pass

   def curve(self, date):
      pass


class PricingModel:
   """Every Pricing model must be derived from this class."""

   def __init__(self, **kwargs):
      """Constructor"""
      raise Exception('model not implemented.')

   def simulate(self, fixings=None, Nsim=1, seed=14, **kwargs):
      """Simulate Spot at fixing dates."""
      raise Exception('simulate not implemented.')


class Black(PricingModel):
   """Black model"""

   def __init__(self, volatility=None, forward_curve=None, **kwargs):
      pass

   def simulate(self, fixings=None, Nsim=1, seed=14, **kwargs):
      pass