# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import pytest

from llamafactory.data import Role
from llamafactory.data.converter import get_dataset_converter
from llamafactory.data.parser import DatasetAttr
from llamafactory.hparams import DataArguments


@pytest.mark.runs_on(["cpu", "mps"])
def test_alpaca_converter():
    dataset_attr = DatasetAttr("hf_hub", "llamafactory/tiny-supervised-dataset")
    data_args = DataArguments()
    example = {
        "instruction": "Solve the math problem.",
        "input": "3 + 4",
        "output": "The answer is 7.",
    }
    dataset_converter = get_dataset_converter("alpaca", dataset_attr, data_args)
    result = dataset_converter(example)
    assert {key: value for key, value in result.items() if not key.startswith("_score_")} == {
        "_prompt": [{"role": Role.USER.value, "content": "Solve the math problem.\n3 + 4"}],
        "_response": [{"role": Role.ASSISTANT.value, "content": "The answer is 7."}],
        "_system": "",
        "_tools": "",
        "_images": None,
        "_videos": None,
        "_audios": None,
    }
    assert math.isnan(result["_score_chosen"])
    assert math.isnan(result["_score_rejected"])


@pytest.mark.runs_on(["cpu", "mps"])
def test_sharegpt_converter():
    dataset_attr = DatasetAttr("hf_hub", "llamafactory/tiny-supervised-dataset")
    data_args = DataArguments()
    example = {
        "conversations": [
            {"from": "system", "value": "You are a helpful assistant."},
            {"from": "human", "value": "Solve the math problem.\n3 + 4"},
            {"from": "gpt", "value": "The answer is 7."},
        ]
    }
    dataset_converter = get_dataset_converter("sharegpt", dataset_attr, data_args)
    result = dataset_converter(example)
    assert {key: value for key, value in result.items() if not key.startswith("_score_")} == {
        "_prompt": [{"role": Role.USER.value, "content": "Solve the math problem.\n3 + 4"}],
        "_response": [{"role": Role.ASSISTANT.value, "content": "The answer is 7."}],
        "_system": "You are a helpful assistant.",
        "_tools": "",
        "_images": None,
        "_videos": None,
        "_audios": None,
    }
    assert math.isnan(result["_score_chosen"])
    assert math.isnan(result["_score_rejected"])


@pytest.mark.runs_on(["cpu", "mps"])
def test_sharegpt_pairwise_converter_with_scores():
    dataset_attr = DatasetAttr("hf_hub", "llamafactory/tiny-ranking-dataset")
    dataset_attr.ranking = True
    dataset_attr.chosen = "chosen"
    dataset_attr.rejected = "rejected"
    dataset_attr.score_chosen = "score_chosen"
    dataset_attr.score_rejected = "score_rejected"
    data_args = DataArguments()
    example = {
        "conversations": [{"from": "human", "value": "Question"}],
        "chosen": {"from": "gpt", "value": "Better answer"},
        "rejected": {"from": "gpt", "value": "Worse answer"},
        "score_chosen": 4.0,
        "score_rejected": 1.5,
    }
    dataset_converter = get_dataset_converter("sharegpt", dataset_attr, data_args)
    assert dataset_converter(example) == {
        "_prompt": [{"role": Role.USER.value, "content": "Question"}],
        "_response": [
            {"role": Role.ASSISTANT.value, "content": "Better answer"},
            {"role": Role.ASSISTANT.value, "content": "Worse answer"},
        ],
        "_system": "",
        "_tools": "",
        "_score_chosen": 4.0,
        "_score_rejected": 1.5,
        "_images": None,
        "_videos": None,
        "_audios": None,
    }
