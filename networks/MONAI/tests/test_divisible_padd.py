# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import unittest

from parameterized import parameterized

from monai.transforms import DivisiblePadd
from monai.utils.enums import NumpyPadMode, PytorchPadMode
from tests.padders import PadTest

TESTS = [
    [{"keys": "img", "k": [4, 3, 2]}, (3, 8, 8, 4), (3, 8, 9, 4)],
    [{"keys": "img", "k": 7, "method": "end"}, (3, 8, 7), (3, 14, 7)],
]


class TestDivisiblePadd(PadTest):
    Padder = DivisiblePadd

    @parameterized.expand(TESTS)
    def test_pad(self, input_param, input_shape, expected_shape):
        modes = ["constant", NumpyPadMode.CONSTANT, PytorchPadMode.CONSTANT, "edge", NumpyPadMode.EDGE]
        self.pad_test(input_param, input_shape, expected_shape, modes)


if __name__ == "__main__":
    unittest.main()
