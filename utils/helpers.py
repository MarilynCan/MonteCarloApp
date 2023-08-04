"""A magnitude model based on the lognormal distribution define by a 90% confidence interval
"""

#   Copyright 2019-2020 Netflix, Inc.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import math
from scipy.stats import norm
from scipy.stats import lognorm
import scipy.stats as stats
import numpy as np

class LognormalMagnitude(object):
    def __init__(self, low_loss, high_loss):
        """:param  low_loss = Low loss estimate
        :param high_loss = High loss estimate

        The range low_loss -> high_loss should represent the 90% confidence interval
        that the loss will fall in that range.

        These values are then fit to a lognormal distribution so that they fall at the 5% and
        95% cumulative probability points.
        """
        if low_loss >= high_loss:
            # High loss must exceed low loss
            raise AssertionError
        self.low_loss = low_loss
        self.high_loss = high_loss
        self._setup_lognormal(low_loss, high_loss)

    def _setup_lognormal(self, low_loss, high_loss):
        # Set up the lognormal distribution
        factor = -0.5 / norm.ppf(0.05)
        mu = (math.log(low_loss) + math.log(high_loss)) / 2.  # Average of the logn of low/high
        shape = factor * (math.log(high_loss) - math.log(low_loss))  # Standard deviation
        self.distribution = lognorm(shape, scale=math.exp(mu))

    def draw(self, n=1):
        return self.distribution.rvs(size=n)

    def mean(self):
        return self.distribution.mean()



"""A Poisson model suitable for frequency. Returns an array of ints.
"""

#   Copyright 2019-2020 Netflix, Inc.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.




class PoissonFrequency(object):
    def __init__(self, frequency):
        """:param frequency = Mean rate per interval"""
        if frequency < 0:
            raise AssertionError("Frequency must be non-negative.")
        self.frequency = frequency

    def draw(self, n=1):
        np.random.seed(68) # Set the seed value (optional)
        return np.random.poisson(self.frequency, n)

    def mean(self):
        return self.frequency



"""A generic loss container that can be populated with different models.
"""

#   Copyright 2019-2020 Netflix, Inc.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import numpy as np
import scipy


class Loss(object):
    def __init__(self, frequency_model, magnitude_model):
        """:param frequency_model: A class with method draw(n=1) to draw a list of n int values
        :param magnitude_model: A class with method draw(n=1) to draw a list of n float values
        """
        self.frequency_model = frequency_model
        self.magnitude_model = magnitude_model

    def annualized_loss(self):
        return self.frequency_model.mean() * self.magnitude_model.mean()

    def simulate_losses_one_year(self):
        """:return List of zero or more loss magnitudes for a single simulated year"""
        num_losses = self.frequency_model.draw()[0]  # Draw a single number of events
        return list(self.magnitude_model.draw(num_losses))

    def simulate_years(self, n):
        """:param n = Number of years to simulate
        :return A list of length n, each entry is the sum of losses for that simulated year"""
        num_losses = self.frequency_model.draw(n)  # Draw a list of the number of events in each year
        loss_values = self.magnitude_model.draw(sum(num_losses))
        losses = [0] * n
        losses_used = 0
        for i in range(n):
            new_losses = num_losses[i]
            losses[i] = sum(loss_values[losses_used:losses_used + new_losses])
            losses_used += new_losses
        return losses

    @staticmethod
    def summarize_loss(loss_array):
        """Get statistics about a numpy array.
        Risk is a range of possibilities, not just one outcome.

        :arg: loss_array = Numpy array of simulated losses
        :returns: Dictionary of statistics about the loss
        """
        percentiles = np.percentile(loss_array, [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 98, 99, 99.5, 99.9, 99.99]).astype(int)
        mode_result = stats.mode(loss_array)
        mode_values = mode_result.mode.tolist()
        mode_value = mode_values[0] if mode_values else 0
        loss_summary = {'Min': np.min(loss_array).astype(int),
                        'Moda': mode_value,
                        'Mediana': percentiles[4],
                        'Max': np.max(loss_array).astype(int),
                        'Media': np.mean(loss_array).astype(int),
                        'P10': percentiles[0],
                        'P20': percentiles[1],
                        'P30': percentiles[2],
                        'P40': percentiles[3],
                        'P50': percentiles[4],
                        'P60': percentiles[5],
                        'P70': percentiles[6],
                        'P80': percentiles[7],
                        'P90': percentiles[8],
                        'P95': percentiles[9],
                        'P98': percentiles[10],
                        'P99': percentiles[11],
                        'P99.5': percentiles[12],
                        'P99.9': percentiles[13],
                        'P99.99': percentiles[14]}
        return loss_summary


    def scatter_plot_losses_curve(self,
                     n,
                     individual_losses,
                     salto_eje_x,
                     salto_eje_y):
        """Generate the Scatter Plot - Simulated Losses Curve for the list of losses for one event
        :arg:   n = Number of years to simulate and display the LEC for.
                individual_losses = simulated losses in a range of n years
        :returns: the Scatter Plot
        """
        plt.figure(figsize=(8, 6))
        plt.scatter(np.arange(1, n+1),individual_losses, color='blue', alpha=0.5)
        plt.xticks(np.arange(0,n+1,salto_eje_x))
        plt.yticks(np.arange(0,max(individual_losses)+1000,salto_eje_y))
        plt.title("Scatter Plot - Simulated Losses")
        plt.xlabel("Simulation")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.show()





"""A simple loss model based on a single loss scenario with
* label = An identifier for the scenario
* name = A descriptive name for the scenario
* p = Probability of occurring within one year
* low_loss = Low loss amount
* high_loss = High loss amount

The range low_loss -> high_loss should represent the 90% confidence interval
that the loss will fall in that range.

These values are then fit to a lognormal
distribution so that they fall at the 5% and 95% cumulative probability points.
"""

#   Copyright 2019-2020 Netflix, Inc.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

#from riskquant import loss
#from riskquant.model import lognormal_magnitude, poisson_frequency


class SimpleLoss(Loss):
    def __init__(self, label, name, frequency, low_loss, high_loss):
        self.label = label
        self.name = name
        self.frequency = frequency
        self.low_loss = low_loss
        self.high_loss = high_loss
        super(SimpleLoss, self).__init__(
            PoissonFrequency(frequency),
            LognormalMagnitude(low_loss, high_loss))



#   Copyright 2019-2020 Netflix, Inc.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import sys

from matplotlib import pyplot as plt
from matplotlib import ticker as mtick
import numpy as np


class MultiLoss(object):
    """A container for a list of loss objects and methods for generating summaries of them."""

    def __init__(self, loss_list):
        self.loss_list = loss_list

    def prioritized_losses(self):
        """Generate a prioritized list of losses from the loss list.

        :returns: List of [(name, annualized_loss), ...] in descending order of annualized_loss.
        """
        result = [(loss.label, loss.name, loss.annualized_loss()) for loss in self.loss_list]
        return sorted(result, key=lambda x: x[2], reverse=True)

    def simulate_years(self, n):
        """Simulate n years across all the losses in the list.

        :arg: n = The number of years to simulate

        :returns: List of [loss_year_1, loss_year_2, ...] where each is a sum of all
                  losses experienced that year."""

        return np.array([loss.simulate_years(n) for loss in self.loss_list]).sum(axis=0)


    def loss_exceedance_curve(self,
                              n,
                              title="Aggregated Loss Exceedance",
                              xlim=[1000000, 10000000000],
                              savefile=None):
        """Generate the Loss Exceedance Curve for the list of losses. (Uses simulate_years)

        :arg: n = Number of years to simulate and display the LEC for.
              [title] = An alternative title for the plot.
              [xlim] = An alternative lower and upper limit for the plot's x axis.
              [savefile] = Save a PNG version to this file location instead of displaying.

        :returns: None. If display=False, returns the matplotlib axis array
                  (for customization)."""
        # Calcula los valores de pérdida para diferentes percentiles
        losses = np.array([np.percentile(self.simulate_years(n), x) for x in range(1, 100, 1)])
        percentiles = np.array([float(100 - x) / 100.0 for x in range(1, 100, 1)])
        # Crea una nueva figura
        _ = plt.figure()
        ax = plt.gca()
        # Grafica los valores de pérdida y percentiles
        ax.plot(losses, percentiles)
        plt.title(title)
        # Configura la escala del eje x en logarítmica
        ax.set_xscale("log")
        # Establece los límites del eje y para que se ajusten a los valores de pérdida y percentiles
        ax.set_ylim(0.0, percentiles[np.argmax(losses > 0.0)] + 0.05)
        # Establece los límites del eje x utilizando los valores proporcionados en xlim
        ax.set_xlim(xlim[0], xlim[1])
        # Formatea el eje x como una etiqueta con el símbolo de moneda y formato numérico
        xtick = mtick.StrMethodFormatter('S/.{x:,.0f}')
        ax.xaxis.set_major_formatter(xtick)
        # Formatea el eje y como un porcentaje
        ytick = mtick.StrMethodFormatter('{x:.000%}')
        ax.yaxis.set_major_formatter(ytick)
        # Muestra las líneas de cuadrícula en ambos ejes
        plt.grid(which='both')
        # Guarda el gráfico como un archivo PNG si se proporciona un nombre de archivo, de lo contrario, muestra el gráfico
        if savefile:
            sys.stderr.write("Saving plot to {}\n".format(savefile))
            plt.savefig(savefile)
        else:
            plt.show()

    def scatter_plot_losses_curve(self,
                     n,
                     individual_losses,
                     salto_eje_x,
                     salto_eje_y):
        """Generate the Scatter Plot - Simulated Losses Curve for the list of losses for one event
        :arg:   n = Number of years to simulate and display the LEC for.
                individual_losses = simulated losses in a range of n years
        :returns: the Scatter Plot
        """
        plt.figure(figsize=(8, 6))
        plt.scatter(np.arange(1, n+1),individual_losses, color='blue', alpha=0.5)
        plt.xticks(np.arange(0,n+1,salto_eje_x))
        plt.yticks(np.arange(0,max(individual_losses)+1000,salto_eje_y))
        plt.title("Scatter Plot - Simulated Losses")
        plt.xlabel("Simulation")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.show()