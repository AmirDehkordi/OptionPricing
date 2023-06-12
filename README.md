# Options Pricing Module 

## Introduction
This python module provides a comprehensive framework to price various types of options: American, European, and Asian, using advanced financial models. These models are implemented in an Object-Oriented Programming (OOP) paradigm, encapsulated in different classes, each representing different types of options and the underlying assets.

The classes provided in this module are designed to be used individually or in combinations, depending on the specific needs of your options pricing problems. The flexibility and modularity of these classes also make them easily extendable for future development and customization.

## Classes
The module consists of the following main classes:

1. Asset
2. Option
3. PathIndependentOption
4. MonteCarloValuedOption
5. BinomialValuedOption
6. EuropeanOption
7. AmericanOption
8. AsianEuropeanOption

Each class has its own attributes and methods, which are briefly discussed below. 

### Asset Class
The Asset class is designed to represent a financial underlying asset, with its own name, current price, and dividend yield. The 'simulate_path' method allows users to simulate a price path using Geometric Brownian Motion. 

```python
asset = Asset(name='AAPL', current_price=150.00, dividend_yield=0.01)
```

The 'simulate_path' method is a key function in this class that simulates a price path for the asset using Geometric Brownian Motion. The method generates a list of asset prices, representing a simulated path based on the input parameters. 

### Option Class
The Option class is a fundamental building block for various types of options. It represents an option contract with a specific underlying asset, exercise price, option type ('call' or 'put'), and maturity time. 

```python
option = Option(name='AAPL_Call', underlying=asset, exercise_price=155.00, option_type='call', maturity_time=1.0)
```

If an incorrect option_type ('call' or 'put') is provided, a ValueError will be raised. 

### PathIndependentOption Class
The PathIndependentOption class is a subclass of the Option class and represents a path-independent option. The '_payoff' method calculates the payoff of the option given the exercise price, current price, and option type. 

### MonteCarloValuedOption Class
The MonteCarloValuedOption class is a subclass of Option and represents an option valued using Monte Carlo simulation. The 'monte_carlo_value' method calculates the value of the option using Monte Carlo simulation, given the number of simulated paths, the length of the simulated paths, the interest rate, and the volatility.

### BinomialValuedOption Class
The BinomialValuedOption class is a subclass of PathIndependentOption. It includes methods to calculate binomial node values and binomial value for the option.

### EuropeanOption Class
The EuropeanOption class is a subclass of both BinomialValuedOption and MonteCarloValuedOption, and represents a European Option.

### AmericanOption Class
The AmericanOption class is a subclass of BinomialValuedOption and represents an American Option, characterized by the possibility of early exercise.

### AsianEuropeanOption Class
The AsianEuropeanOption class is a subclass of MonteCarloValuedOption and represents an Asian European Option.

## Usage
First, import the module into your script.

```python
import options_pricing
```

### Pricing a European Option
Here is an example of pricing a European call option using this module:

```python
asset = Asset(name='AAPL', current_price=150.00, dividend_yield=0.01)
option = EuropeanOption(name='AAPL_Call', underlying=asset, exercise_price=155.00, option_type='call', maturity_time=1.0)
```

# Using binomial model
binomial_price = option.binomial_value(interest_rate=0.05, volatility=0.2, steps=200)
print(f'Binomial model price: {binomial_price}')

# Using Monte Carlo model
monte_carlo_price = option.monte_carlo_value(interest_rate=0.05, volatility=0.2, paths=10000, length=1)
print(f'Monte Carlo model price: {monte_carlo_price}')

In the above example, both the Binomial and Monte Carlo methods are used to calculate the price of the European call option. You will notice that both methods should provide very similar results as the number of steps (for the binomial model) or paths (for the Monte Carlo model) increase. This is due to the Law of Large Numbers, which suggests that as more observations are included in the calculation, the result tends to converge to the expected value.

## Note
It's important to know that the volatility and interest rates used for pricing are assumed to be constant over the life of the option. While this is a simplification (real-world volatility and interest rates change over time), these models still provide useful theoretical prices. For more accurate pricing, you might consider models that allow for changing volatility and interest rates, or employ a calibration process to historical market data.











