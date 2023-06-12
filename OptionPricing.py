"""
@author: Amir Dehkordi.
"""

# Python version (3, 8, 8)


import numpy as np
import random


# Asset
class Asset:
    """
    A class to represent a financial underlying asset.

    Attributes
    ----------
    name : str
        Name of the asset.
    current_price : float
        The current price of the asset.
    dividend_yield : float
        The yield of the asset's dividend. Default is 0.

    Methods
    -------
    simulate_path(path_length, final_time, interest_rate, volatility)
        Simulates a price path using Geometric Brownian Motion.
    """

    def __init__(self, name, current_price, dividend_yield=0):
        """
        Construct all the necessary attributes for the asset object.

        Parameters
        ----------
        name : str
            Name of the asset.
        current_price : float
            The current price of the asset.
        dividend_yield : float, optional
            The yield of the asset's dividend. Default is 0.
        """
        # Determine the attributes
        self.name = name
        self.current_price = current_price
        self.dividend_yield = dividend_yield

    def simulate_path(self, path_length, final_time, *, interest_rate,
                      volatility):
        """
        Simulate a price path for the asset using Geometric Brownian Motion.

        Parameters
        ----------
        path_length : int
            The number of steps to simulate.
        final_time : float
            The final time point of the simulation.
        interest_rate : float, keyword-only
            The risk-free interest rate.
        volatility : float, keyword-only
            The volatility of the asset.

        Returns
        -------
        list
            A list of asset prices representing a simulated path.

        Notes
        -----
        1. The drift rate for the random walk is
        (interest_rate - dividend_yield).
        2. 'dt' can be calculated using the total time devided by the number of
        steps.
        3. Since W_{t+s} - W_{t} follows a normal distribution with mean=0 and
        variance=dt,therefore, W_{t+dt} - W{t} ~ N(0, 1)*(dt)^1/2.
        """
        # Determine the drift rate
        drift_rate = interest_rate - self.dividend_yield

        # Calculate the step size
        dt = final_time / path_length

        # Initialize path with the current price
        path = [self.current_price]
        Stn_1 = self.current_price

        # Simulate the path
        for i in range(1, path_length+1):
            # Generate Wiener process from normal distribution
            W_process = random.normalvariate(0, 1)*(np.sqrt(dt))

            # Calculate the next price
            Stn = Stn_1 * np.exp((drift_rate - (volatility**2)/2)*(dt) +
                                 volatility*(W_process))

            # Append the price to the path
            path.append(Stn)

            # Update the last price for the next iteration
            Stn_1 = Stn

        return path  # Return the simulated path


# Option
class Option:
    """
    A class to represent an Option.

    Attributes
    ----------
    name : str
        The name of the option.
    underlying : Asset
        The underlying asset for the option.
    exercise_price : float
        The exercise price of the option.
    option_type : str
        The type of the option ('call' or 'put').
    maturity_time : float
        The maturity time of the option.
    """

    def __init__(self, name, underlying, *, exercise_price, option_type,
                 maturity_time):
        """
        Construct all the necessary attributes for the Option object.

        Parameters
        ----------
        name : str
            The name of the option.
        underlying : Asset
            The underlying asset for the option.
        exercise_price : float, keyword-only
            The price at which the option can be exercised.
        option_type : str, keyword-only
            The type of the option. Must be either 'call' or 'put'.
        maturity_time : float, keyword-only
            The time when the option can be exercised.

        Raises
        ------
        ValueError
            If `option_type` is not 'call' or 'put'.
        """
        # Determine the attributes
        self.name = name
        self.underlying = underlying
        self.exercise_price = exercise_price

        # Check that option_type is 'call' or 'put'
        if option_type.lower() not in ['call', 'put']:
            raise ValueError("option_type must be 'call' or 'put'")

        self.option_type = option_type
        self.maturity_time = maturity_time


class PathIndependentOption(Option):
    """
    A class used to represent a Path Independent Option (Subclass of  Option).

    Attributes
    ----------
    Inherits all attributes from the Option class

    Methods
    -------
    _payoff(exercise_price, current_price, option_type)
        Calculates the payoff of the option given the exercise price,
        current price, and option type.
    payoff()
        Computes the payoff of the option using the _payoff method.
    """

    @staticmethod
    def _payoff(exercise_price, current_price, option_type):
        """
        Use Static method to calculate the payoff of the option.

        Parameters
        ----------
        exercise_price : float
            The price at which the option can be exercised.
        current_price : float
            The current price of the underlying asset.
        option_type : str
            The type of the option, either 'call' or 'put'.

        Returns
        -------
        float
            The calculated payoff of the option.
        """
        # Calcluate payoff based on the type of the option ('call' or 'put')
        if option_type.lower() == 'call':
            return max(0, current_price - exercise_price)
        elif option_type.lower() == 'put':
            return max(0, exercise_price - current_price)

    def payoff(self):
        """
        Compute the payoff of the option using the _payoff method.

        Returns
        -------
        float
            The payoff of the option.
        """
        # Return the applied payoff
        return self._payoff(self.exercise_price, self.underlying.current_price,
                            self.option_type)


# MonteCarloValuedOption
class MonteCarloValuedOption(Option):
    """
    Represent option valued using Monte Carlo simulation (subclass of Option).

    Methods
    -------
    _monte_carlo_sim_value(exercise_price, current_price, option_type,
                           interest_rate, maturity_time)
        Calculate the payoff of the option using Monte Carlo simulation.
    monte_carlo_value(num_paths, path_length, interest_rate, volatility)
        Calculate the value of the option using Monte Carlo simulation.
    """

    def monte_carlo_value(self, num_paths, path_length, *, interest_rate,
                          volatility):
        """
        Calculate the value of the option using Monte Carlo simulation.

        Parameters
        ----------
        num_paths : int
            The number of simulated paths to construct.
        path_length : int
            The length of the simulated paths.
        interest_rate : float, keyword-only
            The annualized, continuously-compounded risk-free interest rate.
        volatility : float, keyword-only
            The annualized volatility of the underlying asset.

        Returns
        -------
        float
            The estimated value of the option for the Monte Carlo simulation.

        Notes
        -----
        1. NumPy has been used for more optimized implication of the for loop
        since we simulate hundreds of thousands of paths to get good results.
        2. The fair value of the option in Monte Carlo method is the present
        value of the expected values of payoff at maturity time=T.
        """
        # Create vector of zeros to store the payoff of the paths
        mc_value = np.zeros(num_paths)

        # For loop to construct paths
        for i in range(num_paths):
            # Simulate & apply the 'presentvalue'*'Expected value' to payoff
            mc_payoff = self._monte_carlo_sim_value(
                self.exercise_price,
                path_length,
                self.option_type,
                interest_rate,
                self.maturity_time,
                volatility
                )
            # Append the values to the vector of the paths
            mc_value[i] = mc_payoff

        return np.mean(mc_value)  # Return the average of the values


# BinomialValuedOption
class BinomialValuedOption(PathIndependentOption):
    """
    Represents a Binomial Valued Option (Subclass of PathIndependentOption).

    This class contains methods to calculate binomial node values and
    binomial value for the option.
    """

    def binomial_value(self, n, u=None, d=None, p=None, *, interest_rate,
                       volatility, method=None):
        """
        Calculate the binomial value of the option.

        Parameters
        ----------
        n : int
            The maximum level in the tree.
        u : float, optional
            The upward movement in the price.
        d : float, optional
            The downward movement in the price.
        p : float, optional
            The probability that the underlying moves up.
        interest_rate : float, keyword-only
            The annualized continuously-compounded interest rate.
        volatility : float, keyword-only
            The annualized volatility.
        method : str, keyword-only, optional
            The method of constructing the binomial tree. Can be either
            'symmetrical' or 'equal probability'.

        Raises
        ------
        ValueError
            If incorrect parameters are provided.

        Returns
        -------
        float
            The binomial value of the option.

        Notes
        -----
        1. (interest_rate - dividend_yield) is being used instead of
        interest_rate for evaluating the values of u, d and p. However,
        it is not the case for stepping down the tree, and interest_rate
        is solely being used.
        2. dt is calculated using the formula dt = T/n, which is
        maturity_time devided by number of steps.
        3. Since volatility and interest_rate are given values, dt has to
        be determined such that satisfies the following constrains for the
        binomial tree:
            3.1. The upward movement must be bigger than 1 (u>1)
            3.2. The downward movement must be between 0 and 1 (0<d<1)
            3.3. The prob of the upward movement must between 0 and 1 (0<p<1)
        If dt is not small enough, the constraints will not be satisfied and
        as a result the solution is not valid.
        4. If the values of u, d and p are given in the funciton and are not
        satisfying the constraints, another error will be raised, indicating
        the parameters are not appropriate.
        """
        # Check that only 'method' or set of 'u, d, p' is provided
        if method is not None:
            if any([u is not None, d is not None, p is not None]):
                raise ValueError(
                    "If 'method' provided, 'u', 'd', 'p' must not be provided."
                    )
        else:
            if any([param is None for param in [u, d, p]]):
                raise ValueError(
                    "If 'method' not provided, 'u','d', 'p' must be provided."
                    )
            # Check that the parameters are having appropriate values or not
            if not (u > 1 and d < 1 and d > 0 and p < 1 and p > 0):
                raise ValueError(
                    "The values of u, d and p are not appropriate."
                )

        # Assign values of u, d and p based on the method
        if method == 'symmetrical':
            A = ((np.exp(-(interest_rate-self.underlying.dividend_yield) *
                         self.maturity_time/n) +
                 np.exp(((interest_rate-self.underlying.dividend_yield)
                         + volatility**2)*self.maturity_time/n))/2)
            u = A + np.sqrt(A**2 - 1)
            d = A - np.sqrt(A**2 - 1)
            p = ((np.exp((interest_rate-self.underlying.dividend_yield) *
                         self.maturity_time/n) - d) / (u - d))

        elif method == 'equal probability':
            p = 0.5
            u = (np.exp((interest_rate-self.underlying.dividend_yield) *
                        self.maturity_time/n) *
                 (1 + np.sqrt(np.exp(volatility**2 * self.maturity_time/n)
                              - 1)))

            d = (np.exp((interest_rate-self.underlying.dividend_yield) *
                        self.maturity_time/n) *
                 (1 - np.sqrt(np.exp(volatility**2 * self.maturity_time/n)
                              - 1)))

        # Check that only 'symmetrical' and 'equal probability' are allowed.
        if method not in ['symmetrical', 'equal probability', None]:
            raise ValueError(
                "method must be 'symmetrical' or 'equal probability'"
                )

        # Check that the parameters are having appropriate values or not
        if not (u > 1 and d < 1 and d > 0 and p < 1 and p > 0):
            raise ValueError(
                "The timestep is too large,increase the number of 'n'."
                )

        # Calculate the underlying price at top of the tree
        price_at_T = [
            self.underlying.current_price*(u**(i))*(d**(n-i))
            for i in range(0, n+1)
            ]
        # Apply the payoff function at top of the tree
        payoff_T = [
            self._payoff(self.exercise_price, i, self.option_type)
            for i in price_at_T
            ]

        # Initialize the option values V at top of the tree
        option_value = payoff_T
        # For loop to get down to the tree
        for j in range(len(option_value)-1):
            # Create a list of V-1 (One step lower)
            one_step_down = []
            # For loop to calculate each Vi using the upper Vs' value
            for i in range(len(option_value)-1):
                # Calculate the price at each node (for American Options)
                S = (u**(i))*(d**(n-j-i-1))*self.underlying.current_price
                # Calculate V value
                v = self._binomial_node_value(option_value[i+1],
                                              option_value[i], S, p,
                                              interest_rate,
                                              self.maturity_time/n)
                # Append the V value to the list of V-1
                one_step_down.append(v)

            # Assign the V-1 to V and doing the above again to get to the V00
            option_value = one_step_down

        return option_value[0]  # Return value of the option (V00)


# EuropeanOption
class EuropeanOption(BinomialValuedOption, MonteCarloValuedOption):
    """
    Represents a European Option (Subclass of  BVO and MCVO).

    Methods
    -------
    _binomial_node_value(Vk1, Vk, Sk, prob, interest_rate, timestep)
        Calculate the value of a node in a binomial tree.
    _monte_carlo_sim_value(exercise_price, path, option_type,
                           interest_rate, maturity_time)
        Calculate the value of the option using a Monte Carlo simulation.
    """

    def _binomial_node_value(self, Vk1, Vk, Sk, prob, interest_rate, timestep):
        """
        Calculate the value of a node in a binomial tree.

        Parameters
        ----------
        Vk1 : float
            The value of the option at the next time step in the up state.
        Vk : float
            The value of the option at the next time step in the down state.
        Sk : float
            The price of the underlying asset.
        prob : float
            The probability of an upward movement.
        interest_rate : float
            The risk-free interest rate.
        timestep : float
            The length of the time step.

        Returns
        -------
        float
            The value of the option at the current node.

        Notes
        -----
        1. Only interest_rate is being used as the discount rate to
        get down on the tree.
        """
        # Return V based on two nodes above
        return np.exp(-interest_rate*timestep)*(prob*Vk1 + (1-prob)*Vk)

    def _monte_carlo_sim_value(self, exercise_price, path_length, option_type,
                               interest_rate, maturity_time, volatility):
        """
        Calculate the value of the option using a Monte Carlo simulation.

        Parameters
        ----------
        exercise_price : float
            The exercise price of the option.
        path : list
            The simulated path of the underlying asset price.
        option_type : str
            The type of the option ('call' or 'put').
        interest_rate : float
            The risk-free interest rate.
        maturity_time : float
            The time when the option can be exercised.

        Returns
        -------
        float
            The value of the option.

        Raises
        ------
        ValueError
            If `option_type` is not 'call' or 'put'.

        Notes
        -----
        1. To get the Monte Carlo method's simulation value for each path,
        the present value of the payoff at time T (maturity_time) is being
        used for each path.
        2. The last value of the path list is being used as the simulated
        price of the underlying at time T.
        """
        # Simulate a path
        mc_path = self.underlying.simulate_path(path_length,
                                                self.maturity_time,
                                                interest_rate=interest_rate,
                                                volatility=volatility)

        # Calculate the present value of the payoff based on option_type
        if option_type.lower() == 'call':
            return (np.exp(-interest_rate*maturity_time) *
                    max(0, mc_path[-1] - exercise_price))
        elif option_type.lower() == 'put':
            return (np.exp(-interest_rate*maturity_time) *
                    max(0, exercise_price - mc_path[-1]))


# AmericanOption
class AmericanOption(BinomialValuedOption):
    """
    Respresents an American Option, which is a type of BinomialValuedOption.

    This class inherits from the BinomialValuedOption class and overrides
    the _binomial_node_value method to account for the possibility of early
    exercise, which is characteristic of American options.

    Methods
    -------
    _binomial_node_value(Vk1, Vk, Sk, prob, interest_rate, timestep)
        Calculate the value of a node in the binomial tree, taking into
        account the possibility of early exercise.
    """

    def _binomial_node_value(self, Vk1, Vk, Sk, prob, interest_rate, timestep):
        """
        Take into account the possibility of early exercise.

        The value of an American option at a node is the maximum of the
        option's payoff and its expected future value discounted back
        to the present.

        Parameters
        ----------
        Vk1 : float
            The value of the option at the next time step in the up state.
        Vk : float
            The value of the option at the next time step in the down state.
        Sk : float
            The price of the underlying asset at the current node.
        prob : float
            The risk-neutral probability of an up move.
        interest_rate : float
            The risk-free interest rate.
        timestep : float
            The length of the time step.

        Returns
        -------
        float
            The value of the option at the current node.

        Notes
        -----
        1. Only interest_rate is being used as the discount rate
        to get down on the tree.
        """
        # Return V based on two nodes above
        return max(self._payoff(self.exercise_price, Sk, self.option_type),
                   np.exp(-interest_rate*timestep)*(prob*Vk1 + (1-prob)*Vk))


# AsianEuropeanOption
class AsianEuropeanOption(MonteCarloValuedOption):
    """
    Represents an Asian European Option (Subclass of MonteCarloValuedOption).

    Attributes
    ----------
    averaging_method : str
        The method used to average  price path ('arithmetic' or 'geometric').

    Methods
    -------
    __init__(self, name, underlying, exercise_price, option_type,
             maturity_time, averaging_method)
        Constructs all the necessary attributes for the AsianEuropeanOption
        object.
    _monte_carlo_sim_value(self, exercise_price, path, option_type,
                           interest_rate, maturity_time)
        Simulates the value of the Asian option using Monte Carlo simulation.
    """

    def __init__(self, name, underlying, *, exercise_price, option_type,
                 maturity_time, averaging_method):
        """
        Construct all the necessary attributes of AsianEuropeanOption object.

        Parameters
        ----------
        name : str
            The name of the option.
        underlying : Asset
            The underlying asset for the option.
        exercise_price : float, keyword-only
            The price at which the option can be exercised.
        option_type : str, keyword-only
            The type of the option. Must be either 'call' or 'put'.
        maturity_time : float, keyword-only
            The time when the option can be exercised.
        averaging_method : str, keyword-only
            The method used to average the path ('arithmetic' or 'geometric').
        """
        # Determine the attributes
        super().__init__(name, underlying, exercise_price=exercise_price,
                         option_type=option_type,
                         maturity_time=maturity_time)
        self.averaging_method = averaging_method

    def _monte_carlo_sim_value(self, exercise_price, path_length, option_type,
                               interest_rate, maturity_time, volatility):
        """
        Simulate the value of the Asian option using Monte Carlo simulation.

        Parameters
        ----------
        exercise_price : float
            The exercise price of the option.
        path : list of float
            The simulated price path of the underlying asset.
        option_type : str
            The type of the option ('call' or 'put').
        interest_rate : float
            The risk-free interest rate.
        maturity_time : float
            The time to maturity of the option.

        Returns
        -------
        float
            The simulated value of the Asian option.

        Raises
        ------
        ValueError
            If averaging_method is not 'arithmetic' or 'geometric'.

        Notes
        -----
        1. To get the Monte Carlo method's simulation value for each path
        , the present value of the  payoff at time T (maturity_time) is being
        used for each path.
        2. The last value of the path list and the average of the path is being
        used to calculate the payoff since it is an Asian option.
        3. Since we have 'average strike' Asian options, it is not dependent on
        the exercise price of the option and it is only dependent on the path
        of the price.
        """
        # Simulate a path
        mc_path = self.underlying.simulate_path(path_length,
                                                self.maturity_time,
                                                interest_rate=interest_rate,
                                                volatility=volatility)

        # Check that averaging_method is 'arithmetic' or 'geometric'
        if self.averaging_method.lower() not in ['arithmetic', 'geometric']:
            raise ValueError(
                "averaging_method must be 'arithmetic' or 'geometric'"
                )

        # Calculate the presented value of payoff based on the averaging_method
        if self.averaging_method.lower() == 'arithmetic':
            # Calculate the arithmetic average of the path
            S_avg = np.mean(mc_path)
            # Apply present value of the payoff based on option_type
            if option_type == 'call':
                return (np.exp(-interest_rate*maturity_time) *
                        max(0, mc_path[-1] - S_avg))
            elif option_type == 'put':
                return (np.exp(-interest_rate*maturity_time) *
                        max(0, S_avg - mc_path[-1]))

        elif self.averaging_method.lower() == 'geometric':
            # Calculate the geometric average of the path
            S_avg = np.exp(np.mean(np.log(mc_path)))
            # Apply the present value of the payoff based on option_type
            if option_type == 'call':
                return (np.exp(-interest_rate*maturity_time) *
                        max(0, mc_path[-1] - S_avg))
            elif option_type == 'put':
                return (np.exp(-interest_rate*maturity_time) *
                        max(0, S_avg - mc_path[-1]))


# Test
if __name__ == '__main__':
    current_p = 40
    asset = Asset('AAPL', current_p, dividend_yield=0.2)
    int_rate = 0.05
    vol = 0.25
    T = 2
    ot = ['call', 'put']
    name = 'AAPL_option'

    # For loop based on the exercise price
    for j in ot:
        for i in range(current_p - 20, current_p + 21):
            american = AmericanOption(name, asset, exercise_price=i,
                                      option_type=j, maturity_time=T)
            a = american.binomial_value(100, interest_rate=int_rate,
                                        volatility=vol,
                                        method='equal probability')
            euro = EuropeanOption(name, asset, exercise_price=i,
                                  option_type=j, maturity_time=T)
            eb = euro.binomial_value(100, interest_rate=int_rate,
                                     volatility=vol,
                                     method='equal probability')
            em = euro.monte_carlo_value(1000, path_length=100,
                                        interest_rate=int_rate, volatility=vol)
            print(i)
            print(f'American {j} Option:', a)
            print(f'Euro Binom {j} Optn:', eb)
            print(f'Euro Monte {j} Optn:', em)
            for z in ['geometric', 'arithmetic']:
                asian = AsianEuropeanOption(name, asset, exercise_price=i,
                                            option_type=j, maturity_time=T,
                                            averaging_method=z)
                ae = asian.monte_carlo_value(1000, path_length=100,
                                             interest_rate=int_rate,
                                             volatility=vol)

                print(f'Asian Euro {z} {j} Optn:', ae)
