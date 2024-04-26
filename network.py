from collections.abc import Collection
from contextlib import contextmanager
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
import torch
from networkx import DiGraph, draw, topological_generations, multipartite_layout
from pomegranate.bayesian_network import BayesianNetwork as _BayesianNetwork, Distribution, Categorical, \
    ConditionalCategorical


def _category_to_index(data: pd.Series, strict: bool = True) -> pd.Series:
    """Convert categorical values into indices."""
    result = data.cat.codes
    if strict and (result == -1).any():
        raise ValueError('Some data is not in the known categories (or is missing)')
    return result


class BayesianNetworkStateError(Exception):
    """
    The Bayesian network is in an invalid state for the current action to work.
    """
    def __init__(self, message: str, *args):
        super().__init__(message, *args)


class CategoricalBayesianNetwork:
    """
    A Bayesian network that handles categorical data.

    Internally, this is a convenience class that automatically converts categorical input data to PyTorch tensors
    for input into a Pomegranate Bayesian network.
    """
    def __init__(self) -> None:
        self.structure = DiGraph()
        """The structure of the network."""
        self._nodes_to_indices: dict[str, int] = {}
        """A mapping from node names to numerical indices."""
        self._ordered_nodes: list[str] = []
        """The nodes in the order that the internal network expects."""

        self._fitted = False
        """Whether the network is fitted."""
        self.bayes_net: _BayesianNetwork | None = None
        """The internal Bayesian network."""
        self._nodes_to_categories: dict[str, list[Any]] = {}
        """A mapping from nodes to the list of categories for that node."""

        self._device: torch.device | None = None
        """The device that tensors will be created on. If None, the default device is used."""

    @property
    def device(self) -> torch.device | None:
        """The device that tensors will be created on, or None if the default device is being used."""
        return self._device

    @device.setter
    def device(self, new_device: torch.device | None) -> None:
        """Set the device that tensors will be created on."""
        self._device = new_device

    @property
    def variables(self) -> list[str]:
        """The variables in this Bayesian network."""
        return self._ordered_nodes

    def add_variable(self, name: str) -> None:
        """
        Add a variable to the Bayesian network.

        :param name: The name of the variable.
        """
        if name not in self._ordered_nodes:
            self.structure.add_node(name)
            self._ordered_nodes.append(name)
            self._nodes_to_indices[name] = len(self._ordered_nodes) - 1

        self._fitted = False

    def add_variables(self, *names: str) -> None:
        """
        Add multiple variables to the Bayesian network.
        :param names: The names of the variables.
        """
        names = [name for name in names if name not in self._ordered_nodes]
        self.structure.add_nodes_from(names)
        self._ordered_nodes.extend(names)
        for name in names:
            self._nodes_to_indices[name] = self._ordered_nodes.index(name)

        self._fitted = False

    def add_variables_from_data(self, data: pd.DataFrame) -> None:
        """
        Add variables to the Bayesian network for each column in the data frame.
        :param data: A pandas DataFrame.
        :return:
        """
        self.add_variables(*data.columns)

    def add_dependency_to(self, variable: str, dependency: str) -> None:
        """
        Add a dependency to a variable in the network.
        :param variable: The name of the variable to which the dependency will be added.
        :param dependency: The name of the variable that the first variable will depend on.
        """
        if variable not in self.structure.nodes:
            raise KeyError(variable)
        if dependency not in self.structure.nodes:
            raise KeyError(dependency)
        self.structure.add_edge(dependency, variable)

    def add_dependencies_to(self, variable: str, dependencies: Collection[str]) -> None:
        """
        Add dependencies to a variable in the network.
        :param variable: The name of the variable to which the dependencies will be added.
        :param dependencies: The names of the variables that the first variable will depend on.
        """
        for dependency in dependencies:
            self.add_dependency_to(variable, dependency)

    def draw(self, **kwargs) -> None:
        """Draw the structure of this Bayesian network."""
        # This code is from https://networkx.org/documentation/stable/auto_examples/graph/plot_dag_layout.html
        for layer, nodes in enumerate(topological_generations(self.structure)):
            # `multipartite_layout` expects the layer as a node attribute, so add the
            # numeric layer value as a node attribute
            for node in nodes:
                self.structure.nodes[node]["layer"] = layer

        # Compute the multipartite_layout using the "layer" node attribute
        pos = multipartite_layout(self.structure, subset_key="layer")

        kwargs = dict(pos=pos, with_labels=True) | kwargs
        draw(self.structure, **kwargs)

    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit the Bayesian network to the provided data.
        :param data: A DataFrame containing the training data.
        """
        self._make_bayes_net()

        data: pd.DataFrame = data[self._ordered_nodes].apply(lambda col: col.astype('category'), axis=0)
        self._nodes_to_categories = {name: col.cat.categories.to_list() for name, col in data.items()}
        X = torch.as_tensor(data.apply(_category_to_index, axis=0).to_numpy(dtype=np.int32), device=self.device)
        self.bayes_net.fit(X)
        self._fitted = True

    def summarize(self, data: pd.DataFrame) -> None:
        """
        Summarize the given data and add it to the internal summary info.
        :param data: The data to summarize.
        """
        self._make_bayes_net()

        data: pd.DataFrame = data[self._ordered_nodes].apply(lambda col: col.astype('category'), axis=0)
        self._nodes_to_categories = {name: col.cat.categories.to_list() for name, col in data.items()}
        X = torch.as_tensor(data.apply(_category_to_index, axis=0).to_numpy(dtype=np.int32), device=self.device)
        self.bayes_net.summarize(X)

    def from_summaries(self) -> None:
        """Fit the Bayesian network to the summarized data."""
        if self.bayes_net is None:
            raise ValueError('The Bayesian net has not been created yet')

        self.bayes_net.from_summaries()

    def probability(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate the probability of the given data occurring given the fitted Bayesian network's model.
        :param data: A pandas DataFrame containing data for which the occurrence probability will be calculated.
        :return: A pandas Series that contains the probability of each state occurring for each scenario in the given
                 data.
        :raises BayesianNetworkStateError: If the Bayesian network has not been fitted to any data.
        """
        if not self._fitted:
            raise BayesianNetworkStateError('Not fitted yet')

        return pd.Series(self.bayes_net.probability(self._get_tensor_from_dataframe(data)).detach().cpu().numpy(),
                         index=data.index,
                         name='probability')

    def log_probability(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate the log-probability of the given data occurring given the fitted Bayesian network's model. This is
        more numerically stable than calling the probability function directly.
        :param data: A pandas DataFrame containing data for which the occurrence log-probability will be calculated.
        :return: A pandas Series that contains the log-probability of each state occurring for each scenario in the
                 given data.
        :raises BayesianNetworkStateError: If the Bayesian network has not been fitted to any data.
        """
        if not self._fitted:
            raise BayesianNetworkStateError('Not fitted yet')

        return pd.Series(self.bayes_net.log_probability(self._get_tensor_from_dataframe(data)).detach().cpu().numpy(),
                         index=data.index,
                         name='log_probability')

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict the value of the missing data in the given DataFrame using information from the fitted Bayesian
        network's model.
        :param data: A pandas DataFrame containing data with missing values.
        :return: A pandas DataFrame with the missing values filled in with the predicted values.
        :raises BayesianNetworkStateError: If the Bayesian network has not been fitted to any data.
        """
        if not self._fitted:
            raise BayesianNetworkStateError('Not fitted yet')

        original_column_order = data.columns
        X = self._get_masked_tensor_from_dataframe(data)
        X_filled = self.bayes_net.predict(X)

        return self._get_dataframe_from_tensor(X_filled)[original_column_order]

    def predict_proba(self, data: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """
        Get the probability distributions for the possible entries of each value in the given data. Given values will
        have a deterministic distribution, and missing values will have a distribution informed by the fitted Bayesian
        network.
        :param data: A pandas DataFrame containing data with missing values.
        :return: A mapping from names of nodes in the Bayesian network to DataFrames containing the probability
        distributions for each scenario in the given DataFrame.
        :raises BayesianNetworkStateError: If the Bayesian network has not been fitted to any data.
        """
        if not self._fitted:
            raise BayesianNetworkStateError('Not fitted yet')

        return {name: self._get_dataframe_from_probabilities(name, probs, index=data.index)
                for name, probs in zip(self._ordered_nodes,
                                       self.bayes_net.predict_proba(self._get_masked_tensor_from_dataframe(data)))}

    def predict_log_proba(self, data: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """
        Get the log-probability distributions for the possible entries of each value in the given data. Given values
        will have a deterministic distribution, and missing values will have a distribution informed by the fitted
        Bayesian network. This is more numerically stable than predict_proba.
        :param data: A pandas DataFrame containing data with missing values.
        :return: A mapping from names of nodes in the Bayesian network to DataFrames containing the log-probability
        distributions for each scenario in the given DataFrame.
        :raises BayesianNetworkStateError: If the Bayesian network has not been fitted to any data.
        """
        if not self._fitted:
            raise BayesianNetworkStateError('Not fitted yet')

        return {name: self._get_dataframe_from_probabilities(name, probs, index=data.index)
                for name, probs in zip(self._ordered_nodes,
                                       self.bayes_net.predict_log_proba(self._get_masked_tensor_from_dataframe(data)))}

    def sample(self, num_samples: int) -> pd.DataFrame:
        """
        Get samples from the fitted Bayesian network.
        :param num_samples: The number of samples.
        :return: A pandas DataFrame containing the samples.
        :raises BayesianNetworkStateError: If the Bayesian network has not been fitted to any data.
        """
        if not self._fitted:
            raise BayesianNetworkStateError('Not fitted yet')

        return self._get_dataframe_from_tensor(self.bayes_net.sample(num_samples))

    def get_distribution_for(self, variable: str) -> Distribution:
        """
        Get the probability distribution for the given variable.
        :param variable: The name of the variable.
        :return: A Pomegranate Distribution object for the given variable's distribution.
        :raises BayesianNetworkStateError: If the Bayesian network has not been fitted to any data.
        """
        if not self._fitted:
            raise BayesianNetworkStateError('Not fitted yet')

        return self.bayes_net.distributions[self._nodes_to_indices[variable]]

    def get_distribution_probs_for(self, variable: str) -> pd.DataFrame | pd.Series:
        """
        Get the probability distribution probabilities for the given variable.
        :param variable: The name of the variable.
        :return: A Series if the distribution is non-conditional, and a DataFrame if it is conditional.
        :raises BayesianNetworkStateError: If the Bayesian network has not been fitted to any data.
        """
        if not self._fitted:
            raise BayesianNetworkStateError('Not fitted yet')

        distribution = self.get_distribution_for(variable)
        if isinstance(distribution, Categorical):
            probs = distribution.probs.detach().cpu().numpy()[0]
            return pd.Series(probs, name=variable, index=self._nodes_to_categories[variable])
        elif isinstance(distribution, ConditionalCategorical):
            probs = distribution.probs[0].detach().cpu().numpy()
            dependencies = [self._ordered_nodes[index]
                            for index in self.bayes_net.structure[self._nodes_to_indices[variable]]]
            multiindex = pd.MultiIndex.from_product([self._nodes_to_categories[dependency]
                                                     for dependency in dependencies],
                                                    names=dependencies)
            categories = self._nodes_to_categories[variable]
            return pd.DataFrame(probs.reshape((-1, len(categories))), index=multiindex, columns=categories)
        else:
            raise ValueError('Unrecognized distribution type')

    def get_categories_for(self, variable: str) -> list[Any]:
        """
        Get the categories for the given variable.
        :param variable: The name of the variable.
        :return: The list of categories for the given variable.
        """
        if not self._fitted:
            raise BayesianNetworkStateError('Not fitted yet')

        return self._nodes_to_categories[variable]

    class _NetworkAnalyser:
        """
        A class for analysing Bayesian networks.
        """

        def __init__(self, network: 'CategoricalBayesianNetwork'):
            self._network = network
            self._predictor_input = np.full(shape=(1, len(network.variables)), fill_value=-1, dtype=np.int32)
            self._predictor_output = None
            self._freeze_log: dict[str, Any] = {}

            self._calc_predictor_output()

        @property
        def currently_frozen_variables(self) -> dict[str, Any]:
            """All the currently frozen variables and their values."""
            return self._freeze_log

        @contextmanager
        def freezing(self, **variables_to_values) -> None:
            """
            A context manager that freezes a set of variables to given values. The keyword arguments to this context
            manager should have the variable name as the key and the value as the value.
            :raises KeyError: If the value of the variable is not one of the categories for that variable or the
            variable itself is not in the network.
            """
            freeze_log_old_state = self._freeze_log.copy()
            self._freeze_log.update(variables_to_values)
            for variable, value in variables_to_values.items():
                try:
                    self._predictor_input[0, self._network._nodes_to_indices[variable]] = (
                        self._network._nodes_to_categories[variable].index(value))
                except KeyError as e:
                    e.add_note(f'The variable \'{variable}\' could not be found.')
                    raise
                except ValueError as e:
                    e.add_note(f'The value you entered for the variable \'{variable}\' was invalid. '
                               f'It must be one of {self._network._nodes_to_categories[variable]}')
                    raise
            old_output = self._predictor_output.copy() if self._predictor_output is not None else None
            self._calc_predictor_output()

            try:
                yield
            finally:
                self._predictor_output = old_output
                for variable in variables_to_values.keys():
                    self._predictor_input[0, self._network._nodes_to_indices[variable]] = -1
                self._freeze_log = freeze_log_old_state

        def get_probability_for(self, variable: str) -> pd.Series:
            """
            Get the probability distribution for a certain variable with the current state of the network.
            :param variable: The name of the variable.
            :return: The probability distribution for the network as a pandas Series.
            """
            probs = self._predictor_output[self._network._nodes_to_indices[variable]].detach().cpu().numpy()[0]
            return pd.Series(probs, name=variable, index=self._network._nodes_to_categories[variable])

        def _calc_predictor_output(self) -> None:
            data = torch.tensor(self._predictor_input, device=self._network.device)
            masked_data = torch.masked.masked_tensor(data, mask=(data != -1))
            self._predictor_output = self._network.bayes_net.predict_proba(masked_data)

    def analyze(self) -> _NetworkAnalyser:
        """
        Return an analyzer for analyzing this network. The analyzer can freeze variables to certain states and tell you
        the probability distribution of each variable individually.
        :return: A network analyzer.
        :raises BayesianNetworkStateError: If the Bayesian network has not been fitted to any data.
        """
        if not self._fitted:
            raise BayesianNetworkStateError('Not fitted yet')

        return self._NetworkAnalyser(self)

    def _make_bayes_net(self) -> None:
        """Create a Bayesian network using the structure on the object."""
        structure = [[] for _ in range(len(self._ordered_nodes))]
        for dependency, variable in self.structure.in_edges:
            structure[self._nodes_to_indices[variable]].append(self._nodes_to_indices[dependency])
        self.bayes_net = _BayesianNetwork(structure=[tuple(deps) for deps in structure])

    def _get_tensor_from_dataframe(self, data: pd.DataFrame, strict: bool = True) -> torch.Tensor:
        """Convert a DataFrame to a tensor."""
        def convert_to_category_dtype(col: pd.Series) -> pd.Series:
            result = col.astype('category')
            result.cat.set_categories(self._nodes_to_categories[str(col.name)])
            return result

        data = data[self._ordered_nodes].apply(convert_to_category_dtype, axis=0)
        return torch.as_tensor(data.apply(partial(_category_to_index, strict=strict), axis=0).to_numpy(dtype=np.int32),
                               device=self.device)

    def _get_masked_tensor_from_dataframe(self, data: pd.DataFrame) -> torch.masked.MaskedTensor:
        """Convert a DataFrame with missing values to a masked tensor."""
        full_tensor = self._get_tensor_from_dataframe(data, strict=False)
        return torch.masked.masked_tensor(full_tensor, mask=(full_tensor != -1))

    def _get_dataframe_from_tensor(self,
                                   data: torch.Tensor,
                                   index: pd.Index | None = None) -> pd.DataFrame:
        """Convert a tensor to a DataFrame."""
        def code_to_category(col: pd.Series) -> pd.Series:
            return col.apply(lambda entry: self._nodes_to_categories[col.name][entry])

        return (pd.DataFrame.from_records(data.detach().cpu().numpy(), columns=self._ordered_nodes, index=index)
                .apply(code_to_category, axis=0))

    def _get_dataframe_from_probabilities(self,
                                          node_name: str,
                                          data: torch.Tensor,
                                          index: pd.Index | None = None) -> pd.DataFrame:
        """Convert a probability distribution tensor to a DataFrame."""
        return pd.DataFrame.from_records(data.detach().cpu().numpy(),
                                         columns=self._nodes_to_categories[node_name], index=index)
