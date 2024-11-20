import pykx as kx
import pandas as pd
import datetime
from typing import Any

class QWrapper:
    def __init__(self, connection):
        """
        Initialize the QWrapper with a PyKX connection.

        :param connection: PyKX sync connection to KDB+.
        """
        self.connection = connection

    def q(self, function_name: str, **kwargs: Any):
        """
        Call a Q function with provided arguments and return the result.

        :param function_name: Name of the Q function to call.
        :param kwargs: Keyword arguments to pass to the Q function.
        :return: Result of the query, typically a K object.
        """
        try:
            # Convert arguments to Q-compatible types
            converted_args = {key: self._to_q_value(value) for key, value in kwargs.items()}
            # Prepare the arguments as a dictionary for the Q call
            args_dict = kx.q.dict(converted_args)
            # Call the Q function
            result = self.connection(f'{function_name}[{args_dict}]')
            return QResult(result)
        except Exception as e:
            raise RuntimeError(f"Error executing Q function '{function_name}': {e}")

    def _to_q_value(self, value: Any):
        """
        Convert Python values to Q-compatible values.

        :param value: Python value to convert.
        :return: Q-compatible value.
        """
        if value is None:
            return kx.q.null
        if isinstance(value, bool):
            return kx.q.boolean(value)
        if isinstance(value, int):
            return kx.q.long(value)
        if isinstance(value, float):
            return kx.q.float(value)
        if isinstance(value, str):
            return kx.q.symbol(value)
        if isinstance(value, datetime.date) and not isinstance(value, datetime.datetime):
            return kx.q.date(value.isoformat())
        if isinstance(value, datetime.datetime):
            return kx.q.datetime(value.isoformat())
        if isinstance(value, datetime.time):
            return kx.q.time(value.isoformat())
        if isinstance(value, list):
            # Handle lists of uniform types
            if all(isinstance(v, bool) for v in value):
                return [kx.q.boolean(v) for v in value]
            if all(isinstance(v, int) for v in value):
                return [kx.q.long(v) for v in value]
            if all(isinstance(v, float) for v in value):
                return [kx.q.float(v) for v in value]
            if all(isinstance(v, str) for v in value):
                return [kx.q.symbol(v) for v in value]
            if all(isinstance(v, datetime.date) for v in value):
                return [kx.q.date(v.isoformat()) for v in value]
            if all(isinstance(v, datetime.datetime) for v in value):
                return [kx.q.datetime(v.isoformat()) for v in value]
            if all(isinstance(v, datetime.time) for v in value):
                return [kx.q.time(v.isoformat()) for v in value]
        if isinstance(value, dict):
            # Convert Python dict to Q dictionary
            return kx.q.dict({self._to_q_value(k): self._to_q_value(v) for k, v in value.items()})
        if isinstance(value, kx.K):
            # Already a Q type
            return value
        raise TypeError(f"Unsupported value type: {type(value)}")

class QResult:
    def __init__(self, result):
        """
        Wrap the result of a Q query.

        :param result: K object returned by the query.
        """
        self.result = result

    def pd(self):
        """
        Convert the K object result to a pandas DataFrame.

        :return: Pandas DataFrame.
        """
        try:
            return pd.DataFrame(self.result)
        except Exception as e:
            raise RuntimeError(f"Error converting result to pandas DataFrame: {e}")
