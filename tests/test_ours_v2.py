from __future__ import annotations

import unittest

import torch

from methods.Ours.ours import CBAMConvCrossAttention
from methods.OursV2.cifar100.protocol import protocol_defaults
from methods.OursV2.grid_permutation.model import (
    GridPermutedCrossAttention,
    GridPermutedOurs,
)
from methods.OursV2.ours import GridAwareCrossAttention, Ours
from methods.OursV2.token_space.model import (
    TokenSpaceCrossAttention,
    TokenSpaceOurs,
)


def copy_conv_attention_weights(
    source: GridAwareCrossAttention,
    target: CBAMConvCrossAttention,
) -> None:
    target.cbam.load_state_dict(source.cbam.state_dict())
    target.q_conv.load_state_dict(source.q_conv.state_dict())
    target.k_conv.load_state_dict(source.k_conv.state_dict())
    target.v_conv.load_state_dict(source.v_conv.state_dict())
    target.out_conv.load_state_dict(source.out_conv.state_dict())


def copy_to_token_attention(
    source: GridAwareCrossAttention,
    target: TokenSpaceCrossAttention,
) -> None:
    target.cbam.load_state_dict(source.cbam.state_dict())
    with torch.no_grad():
        for linear, convolution in (
            (target.q_linear, source.q_conv),
            (target.k_linear, source.k_conv),
            (target.v_linear, source.v_conv),
            (target.out_linear, source.out_conv),
        ):
            linear.weight.copy_(convolution.weight[:, :, 0, 0])
            linear.bias.copy_(convolution.bias)


class OursV2AttentionTest(unittest.TestCase):
    def test_zero_initialized_bias_starts_from_v1_attention(self) -> None:
        torch.manual_seed(3)
        v2 = GridAwareCrossAttention(
            embed_dim=8,
            num_heads=2,
            max_grid_size=4,
        ).eval()
        v1 = CBAMConvCrossAttention(embed_dim=8, num_heads=2).eval()
        copy_conv_attention_weights(v2, v1)
        self.assertEqual(
            int(torch.count_nonzero(v2.relative_position_bias.bias_table)),
            0,
        )

        student = torch.randn(2, 8, 4, 4)
        teacher = torch.randn(2, 8, 4, 4)
        with torch.no_grad():
            v1_output = v1(student, teacher)
            v2_output = v2(student, teacher)
        torch.testing.assert_close(v2_output, v1_output, rtol=1e-5, atol=1e-6)

    def test_token_space_matches_v2_only_while_2d_bias_is_zero(self) -> None:
        torch.manual_seed(5)
        grid = GridAwareCrossAttention(
            embed_dim=8,
            num_heads=2,
            max_grid_size=4,
        ).eval()
        token = TokenSpaceCrossAttention(embed_dim=8, num_heads=2).eval()
        copy_to_token_attention(grid, token)

        student = torch.randn(2, 8, 4, 4)
        teacher = torch.randn(2, 8, 4, 4)
        with torch.no_grad():
            zero_bias_output = grid(student, teacher)
            token_output = token(student, teacher)
        torch.testing.assert_close(
            zero_bias_output,
            token_output,
            rtol=1e-5,
            atol=1e-6,
        )

        with torch.no_grad():
            grid.relative_position_bias.bias_table.copy_(
                torch.linspace(
                    -0.5,
                    0.5,
                    grid.relative_position_bias.bias_table.numel(),
                ).reshape_as(grid.relative_position_bias.bias_table)
            )
            position_aware_output = grid(student, teacher)
        self.assertFalse(torch.allclose(position_aware_output, token_output))

    def test_learned_2d_bias_breaks_joint_kv_permutation_invariance(self) -> None:
        torch.manual_seed(7)
        block = GridAwareCrossAttention(
            embed_dim=8,
            num_heads=2,
            max_grid_size=4,
        ).eval()
        student = torch.randn(1, 8, 4, 4)
        teacher = torch.randn(1, 8, 4, 4)
        permutation = torch.randperm(16)
        permuted_teacher = teacher.flatten(2).index_select(
            2,
            permutation,
        ).reshape_as(teacher)

        with torch.no_grad():
            zero_bias_original = block(student, teacher)
            zero_bias_permuted = block(student, permuted_teacher)
        torch.testing.assert_close(
            zero_bias_original,
            zero_bias_permuted,
            rtol=1e-5,
            atol=1e-6,
        )

        with torch.no_grad():
            block.relative_position_bias.bias_table.copy_(
                torch.linspace(
                    -1.0,
                    1.0,
                    block.relative_position_bias.bias_table.numel(),
                ).reshape_as(block.relative_position_bias.bias_table)
            )
            aware_original = block(student, teacher)
            aware_permuted = block(student, permuted_teacher)
        self.assertFalse(torch.allclose(aware_original, aware_permuted))

    def test_relative_position_bias_receives_gradient(self) -> None:
        block = GridAwareCrossAttention(
            embed_dim=8,
            num_heads=2,
            max_grid_size=4,
        )
        student = torch.randn(2, 8, 4, 4, requires_grad=True)
        teacher = torch.randn(2, 8, 4, 4)
        block(student, teacher).square().mean().backward()
        gradient = block.relative_position_bias.bias_table.grad
        self.assertIsNotNone(gradient)
        assert gradient is not None
        self.assertGreater(float(gradient.abs().sum()), 0.0)


class OursV2ControlTest(unittest.TestCase):
    def test_grid_permutation_changes_kv_only_not_mse_targets(self) -> None:
        torch.manual_seed(11)
        module = GridPermutedOurs(
            student_channels=8,
            teacher_channels=(8,),
            num_student_blocks=2,
            num_heads=2,
            max_grid_size=4,
            permutation_seed=1,
        )
        student = [torch.randn(1, 8, 4, 4) for _ in range(2)]
        teacher = [torch.randn(1, 8, 4, 4)]
        _, _, _, _, targets = module(student, teacher)
        torch.testing.assert_close(targets[0], teacher[0])

        block = module.fusion_blocks[0]
        self.assertIsInstance(block, GridPermutedCrossAttention)
        assert isinstance(block, GridPermutedCrossAttention)
        self.assertFalse(
            torch.equal(
                block.permute_teacher(teacher[0]),
                teacher[0],
            )
        )

    def test_token_control_disables_only_v2_position_term(self) -> None:
        full = Ours(
            student_channels=8,
            teacher_channels=(8,),
            num_student_blocks=2,
            num_heads=2,
            max_grid_size=4,
        )
        token = TokenSpaceOurs(
            student_channels=8,
            teacher_channels=(8,),
            num_student_blocks=2,
            num_heads=2,
            max_grid_size=4,
        )
        self.assertTrue(bool(full.ours_v2_relative_position_enabled))
        self.assertFalse(bool(token.ours_v2_relative_position_enabled))
        self.assertIsInstance(full.fusion_blocks[0], GridAwareCrossAttention)
        self.assertIsInstance(token.fusion_blocks[0], TokenSpaceCrossAttention)

    def test_all_cifar100_variants_share_the_same_training_defaults(self) -> None:
        names = (
            "cifar100_deit_ti_ours_v2_relative_position_v1",
            "cifar100_deit_ti_ours_v2_grid_permutation_v1",
            "cifar100_deit_ti_ours_v2_token_space_v1",
        )
        defaults = [dict(protocol_defaults(name)) for name in names]
        for values in defaults:
            values.pop("--protocol-name")
        self.assertEqual(defaults[0], defaults[1])
        self.assertEqual(defaults[0], defaults[2])


if __name__ == "__main__":
    unittest.main()
