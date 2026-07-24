from __future__ import annotations

import unittest
from unittest import mock

import torch

from methods.LG.official_lg import LocalityGuidance
from methods.LG.runtime import (
    StaticGuidanceController,
    create_student,
    finalize_args,
    official_lg_parameter_groups,
    parse_args,
)


class ToyDeiT(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cls_token = torch.nn.Parameter(torch.ones(1, 1, 4))
        self.pos_embed = torch.nn.Parameter(torch.ones(1, 5, 4))
        self.norm = torch.nn.LayerNorm(4)
        self.body = torch.nn.Linear(4, 4)
        self.head = torch.nn.Linear(4, 3)


class OfficialLgRuntimeTest(unittest.TestCase):
    def test_classifier_head_is_zero_initialized(self) -> None:
        student = ToyDeiT()
        fake_timm = mock.Mock()
        fake_timm.create_model.return_value = student
        result = create_student(fake_timm, 3, 0.1)
        self.assertIs(result, student)
        self.assertEqual(torch.count_nonzero(student.head.weight).item(), 0)
        self.assertEqual(torch.count_nonzero(student.head.bias).item(), 0)

    def test_official_weight_decay_exclusions_and_complete_coverage(self) -> None:
        student = ToyDeiT()
        guidance = LocalityGuidance()
        groups = official_lg_parameter_groups(
            student,
            guidance,
            lr=5e-4,
            weight_decay=0.05,
        )
        by_name = {group["name"]: group for group in groups}
        self.assertEqual(by_name["head_no_decay"]["weight_decay"], 0.0)
        self.assertEqual(by_name["body_no_decay"]["weight_decay"], 0.0)
        self.assertEqual(by_name["head_decay"]["weight_decay"], 0.05)
        self.assertEqual(by_name["body_decay"]["weight_decay"], 0.05)

        no_decay_ids = {
            id(parameter)
            for name in ("head_no_decay", "body_no_decay")
            for parameter in by_name[name]["params"]
        }
        self.assertIn(id(student.cls_token), no_decay_ids)
        self.assertIn(id(student.pos_embed), no_decay_ids)
        self.assertIn(id(student.norm.weight), no_decay_ids)
        self.assertIn(id(student.norm.bias), no_decay_ids)
        self.assertIn(id(student.head.bias), no_decay_ids)
        expected = {
            id(parameter)
            for parameter in list(student.parameters()) + list(guidance.parameters())
            if parameter.requires_grad
        }
        actual = {
            id(parameter)
            for group in groups
            for parameter in group["params"]
        }
        self.assertEqual(actual, expected)
        self.assertEqual(
            sum(len(group["params"]) for group in groups),
            len(expected),
        )

    def test_static_lg_never_disables_guidance(self) -> None:
        controller = StaticGuidanceController(beta=2.5)
        for epoch in range(1, 6):
            self.assertEqual(controller.beta_for_epoch(epoch), 2.5)
            controller.observe(epoch, 1.0 / epoch)
            self.assertTrue(controller.active)
            self.assertIsNone(controller.stop_epoch)

    def test_lg_default_run_name_is_not_labeled_alg(self) -> None:
        with mock.patch("sys.argv", ["train.py", "--dataset", "cub200"]):
            args = parse_args()
        args.method = "LG"
        finalize_args(args)
        self.assertEqual(args.run_name, "lg_cub200_deit_ti_300ep")


if __name__ == "__main__":
    unittest.main()
