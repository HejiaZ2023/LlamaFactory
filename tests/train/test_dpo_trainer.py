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

import pytest
import torch


pytest.importorskip("trl")

from llamafactory.train.dpo.trainer import CustomDPOTrainer


@pytest.mark.runs_on(["cpu", "mps"])
def test_beta_weighted_dpo_loss_uses_score_diff():
    trainer = CustomDPOTrainer.__new__(CustomDPOTrainer)
    trainer.beta = 0.1
    losses, chosen_rewards, rejected_rewards = trainer.beta_weighted_dpo_loss(
        policy_chosen_logps=torch.tensor([2.0, 0.5]),
        policy_rejected_logps=torch.tensor([1.0, 0.25]),
        reference_chosen_logps=torch.tensor([1.5, 0.4]),
        reference_rejected_logps=torch.tensor([0.8, 0.3]),
        score_diff=torch.tensor([3.0, float("nan")]),
    )

    expected_logratio_diff = torch.tensor([(2.0 - 1.5) - (1.0 - 0.8), (0.5 - 0.4) - (0.25 - 0.3)])
    expected_beta_star = torch.tensor([0.3, 0.1])
    expected_losses = -torch.log(torch.sigmoid(expected_beta_star * expected_logratio_diff))
    expected_chosen_rewards = expected_beta_star * torch.tensor([2.0 - 1.5, 0.5 - 0.4])
    expected_rejected_rewards = expected_beta_star * torch.tensor([1.0 - 0.8, 0.25 - 0.3])

    assert torch.allclose(losses, expected_losses)
    assert torch.allclose(chosen_rewards, expected_chosen_rewards)
    assert torch.allclose(rejected_rewards, expected_rejected_rewards)
