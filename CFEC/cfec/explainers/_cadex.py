from numpy.typing import NDArray

from cfec.constraints import ValueMonotonicity, Freeze, OneHot
from cfec.base import BaseExplainer

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy



from typing import Union, List, Any, Callable, Optional

_CallableTransform = Callable[[NDArray[np.float32]], NDArray[np.float32]]


class Cadex(BaseExplainer):
    """
    Creates a counterfactual explanation based on a pre-trained model using CADEX method
    The model has to be a Keras classifier model
    """

    def __init__(self,
                 pretrained_model,
                 n_changed: int = 5,
                 columns_to_change: Optional[List[Union[int, str]]] = None,
                 max_epochs: int = 1000,
                 optimizer: tf.keras.optimizers.legacy.Optimizer = tf.keras.optimizers.legacy.Adam(0.01),
                 loss: tf.keras.losses.Loss = CategoricalCrossentropy(),
                 transform: Optional[_CallableTransform] = None,
                 inverse_transform: Optional[_CallableTransform] = None,
                 constraints: Optional[List[Any]] = None) -> None:

        self._constraints = constraints if constraints is not None else []
        self._opt = optimizer
        self._loss = loss
        self._model = pretrained_model
        self._max_epochs = max_epochs
        self._n_changed = n_changed
        self._columns_to_change = columns_to_change  # type: ignore

        self._transform = transform
        self._inverse_transform = inverse_transform

        self._mask: NDArray[np.int32]
        self._C: NDArray[np.int32]
        self._columns: List[str]
        self._dtype: str

    def generate(self, x: pd.Series) -> Union[pd.DataFrame, None]:
        x = self._transform_input(x)
        cf = self._gradient_descent(x)
        if cf is None:
            return None
        x_inv = self._inverse_transform_input(cf)
        assert x_inv is not None
        return pd.DataFrame([x_inv], columns=self._columns)

    def _gradient_descent(self, x: tf.Variable) -> tf.Variable:
        y = self._get_predicted_class(x)
        y_expected: NDArray[Any] = np.array([0, 1]) if y == 0 else np.array([1, 0])

        input_shape = x.shape[1:]
        gradient = self._get_gradient(x, y_expected)
        self._initialize_mask(input_shape, gradient.numpy()[0])
        self._initialize_c(input_shape)

        for _ in range(self._max_epochs):
            gradient = self._get_gradient(x, y_expected)
            updated_mask = self._update_mask(gradient)
            gradient = tf.convert_to_tensor(gradient * updated_mask)
            self._opt.apply_gradients(zip([gradient], [x]))
            if any(isinstance(constraint, OneHot) for constraint in self._constraints):
                corrected_input = self._correct_categoricals(x)
            else:
                corrected_input = x
            if self._get_predicted_class(corrected_input) == np.argmax(y_expected):
                return corrected_input

    def _get_predicted_class(self, x: tf.Variable):
        return np.argmax(self._model(x), axis=1)[0]

    def _get_gradient(self, x, y_true):
        with tf.GradientTape() as t:
            t.watch(x)
            y_pred = self._model(x)
            loss = self._loss(tf.constant([y_true]), y_pred)

        return t.gradient(loss, x)

    def _update_mask(self, gradient):
        new_mask = self._mask.copy()
        for i in range(len(gradient)):
            if not ((self.C[i] > 0 > gradient[i]) or (self.C[i] < 0 < gradient[i]) or self.C[i] == 0):
                new_mask[i] = 0
        return new_mask

    def _correct_categoricals(self, x) -> tf.Variable:
        corrected_x = x.numpy()[0]
        if self._inverse_transform is not None:
            corrected_x = self._inverse_transform(corrected_x)

        for constraint in self._constraints:
            if isinstance(constraint, OneHot):
                feature = corrected_x[constraint.start_column:constraint.end_column + 1]
                max_feature = np.argmax(feature)
                corrected_x[constraint.start_column:constraint.end_column + 1] = 0
                corrected_x[constraint.start_column + max_feature] = 1

        if self._transform is not None:
            corrected_x = self._transform(corrected_x)
            assert self._inverse_transform is not None
            self._inverse_transform(corrected_x)

        return tf.convert_to_tensor([corrected_x], dtype=self._dtype)

    def _transform_input(self, x: pd.Series) -> tf.Variable:
        self._columns = list(x.index)
        self._dtype = x.dtype
        x = x.to_numpy(dtype=self._dtype)
        if self._transform is not None:
            x = self._transform(x)
        return tf.Variable(x[np.newaxis, :], dtype=self._dtype)

    def _inverse_transform_input(self, _x: tf.Variable) -> NDArray[np.float32]:
        x: NDArray[np.float32] = _x.numpy()[0]
        if self._inverse_transform is not None:
            return self._inverse_transform(x)
        return x

    def _initialize_mask_with_columns(self, shape):
        assert self._columns_to_change is not None
        if all(isinstance(column, int) for column in self._columns_to_change):
            column_indices = self._columns_to_change
        else:
            column_indices = []
            for column in self._columns_to_change:
                assert isinstance(column, str)
                column_indices.append(self._columns.index(column))

        self._mask = np.zeros(shape, dtype=self._dtype)
        self._mask[column_indices] = 1

    def _apply_mask_constraints(self, shape, gradient):
        self._mask = np.ones(shape, dtype=self._dtype)
        for constraint in self._constraints:
            if isinstance(constraint, Freeze):
                for column in constraint.columns:
                    column_index = column if isinstance(column, int) else self._columns.index(column)  # type: ignore
                    # if constraint freezes a column which is one-hot encoded, freeze all one-hot columns
                    for constraint_one_hot in self._constraints:
                        if isinstance(constraint, OneHot):
                            if constraint_one_hot.start_column <= column_index <= constraint_one_hot.end_column:
                                self._mask[constraint_one_hot.start_column: constraint_one_hot.end_column + 1] = 0
                    self._mask[column_index] = 0

            if isinstance(constraint, ValueMonotonicity):
                for column in constraint.columns:
                    column_index = column if isinstance(column, int) else self._columns.index(column)  # type: ignore
                    if (constraint.direction == "increasing" and gradient[column_index] > 0) or \
                            (constraint.direction == "decreasing" and gradient[column_index] < 0):
                        self._mask[column_index] = 0

    def _choose_n_features(self, gradient):
        indices = np.argsort(np.absolute(gradient))[::-1]
        categoricals = []
        count = 0
        for i in indices:
            if count < self._n_changed:
                if self._mask[i] == 1:
                    is_categorical = False
                    for constraint in self._constraints:
                        if isinstance(constraint, OneHot):
                            if constraint.start_column <= i <= constraint.end_column:
                                is_categorical = True
                                if constraint not in categoricals:
                                    categoricals.append(constraint)
                                    count += 1
                                    break
                    if not is_categorical:
                        count += 1
                    continue
            self._mask[i] = 0

        for constraint in categoricals:
            self._mask[constraint.start_column: constraint.end_column + 1] = 1

    def _initialize_mask(self, shape, gradient):
        if self._columns_to_change is not None:
            self._initialize_mask_with_columns(shape)

        self._apply_mask_constraints(shape, gradient)
        self._choose_n_features(gradient)

    def _initialize_c(self, shape):
        self.C = np.zeros(shape)
        for constraint in self._constraints:
            if isinstance(constraint, ValueMonotonicity):
                val = 1 if constraint.direction == "increasing" else -1
                for column in constraint.columns:
                    column_index = column if isinstance(column, int) else self._columns.index(column)  # type: ignore
                    self.C[column_index] = val
