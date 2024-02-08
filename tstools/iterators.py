from itertools import islice
from typing import Optional, List

import numpy as np
import pandas as pd


class SequenceIterator:

    def __init__(self, data: pd.DataFrame, sequence_length: int, stride: int = 1,
                 shuffle: bool = False, shuffle_random_state: Optional[int] = None, copy: bool = False):
        self.data = data if not copy else data.copy(deep=True)

        # Definition of sequence length, must be ni [1, len(data)]
        if (sequence_length <= 0) or (sequence_length > len(data)):
            raise ValueError("sequence length out or range")
        self.sequence_length = sequence_length

        # Last valid index is the index of the first element of the last sequence (otherwise the last sequence wouldn't
        # have length of sequence_length)
        self.last_valid_index = len(self.data) - self.sequence_length

        # Maximum stride is reached when you take the first sequence, and then the last sequence
        if (stride <= 0) or (stride > self.last_valid_index):
            raise ValueError("stride out of range")
        else:
            self.stride = stride

        # Computation of the index of the last sequence in dimension of sequence index
        # (index 0 = first sequence, index 1 = second sequence, and so on)
        # computed here to avoid call of __len__ each time __getitem__ is called
        self.last_sequence_index = self.last_valid_index - (self.last_valid_index % self.stride)

        # Definition of the index of sequences
        self.index = np.arange(len(self))

        # If shuffling, we just shuffle the previous index
        if shuffle:
            np.random.shuffle(shuffle_random_state)
            np.random.shuffle(self.index)
            np.random.seed(None)

    def __len__(self):
        # Following works too, kept as comment for beauty of maths
        # return math.floor(self.last_sequence_index / self.stride) + 1
        return self.last_sequence_index + 1

    def __getitem__(self, item):
        if isinstance(item, slice):
            start, stop, step = item.indices(len(self))
            return [self[i] for i in range(start, stop, step)]
        elif np.issubdtype(type(item), np.integer):
            if item <= -len(self) - 1:
                raise IndexError("index out of range")
            elif item <= -1:
                return self[len(self) + item]
            elif (item + self.stride) <= (len(self) - 1):
                # self.index[item] allows here to retrieve random index in shuffling case
                start = self.index[item] * self.stride
                return self.data.iloc[start: start + self.sequence_length].copy(deep=True)
            else:
                raise IndexError("index out of range")
        else:
            raise TypeError(f"SequenceIterator indices must be integers or slices, not {type(item)}")

    def __iter__(self):
        self.iter_step = 0
        return self

    def __next__(self):
        if self.iter_step < len(self):
            self.iter_step += 1
            return self[self.iter_step - 1]
        else:
            raise StopIteration


class MultiSequenceIterator:

    def __init__(self, data: List[pd.DataFrame], sequence_length: int, stride: int = 1, shuffle: bool = False,
                 shuffle_random_state: Optional[int] = None, copy: bool = False):
        self.iterators = [SequenceIterator(dataframe, sequence_length, stride, False, None, copy)
                          for dataframe in data]

        # Save the length of each iterator
        self.lengths = [len(iterator) for iterator in self.iterators]

        # Compute cumulated sum of lengths
        self.lengths_cum_sum = np.cumsum(self.lengths)

        # Computation of the index of the last sequence for each iterator
        self.iterators_last_sequence_index = self.lengths_cum_sum - 1

        # Definition of the index of sequences
        self.index = np.arange(len(self))

        # If shuffling, we just shuffle the previous index
        if shuffle:
            np.random.seed(shuffle_random_state)
            np.random.shuffle(self.index)
            np.random.seed(None)

    def __len__(self):
        return self.lengths_cum_sum[-1]

    def __getitem__(self, item):
        if isinstance(item, slice):
            start, stop, step = item.indices(len(self))
            return [self[i] for i in range(start, stop, step)]
        elif np.issubdtype(type(item), np.integer):
            if item <= -len(self) - 1:
                raise IndexError("index out of range")
            elif item <= -1:
                return self[len(self) + item]
            elif item <= (len(self) - 1):
                index = self.index[item]
                # We find the iterator containing the sequence index
                iterator_index = np.searchsorted(self.iterators_last_sequence_index, index, side="left")
                # If first iterator, we simply return the given sequence
                if iterator_index == 0:
                    return self.iterators[iterator_index][index]
                else:
                    return self.iterators[iterator_index][index - self.lengths_cum_sum[iterator_index - 1]]
            else:
                raise IndexError("index out of range")
        else:
            raise TypeError(f"SequenceIterator indices must be integers or slices, not {type(item)}")

    def __iter__(self):
        self.iter_step = 0
        return self

    def __next__(self):
        if self.iter_step <= len(self):
            self.iter_step = 0
            return self[self.iter_step - 1]
        else:
            raise StopIteration


def batched(iterable, n):
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch
