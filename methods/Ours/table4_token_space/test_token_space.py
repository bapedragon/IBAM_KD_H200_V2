"""Structural tests for the isolated Table 4 token-space control."""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from methods.Ours.ours import CBAMConvCrossAttention, Ours
from methods.Ours.table4_token_space.model import (
    TokenSpaceCrossAttention,
    TokenSpaceOurs,
)


class TokenSpaceControlTests(unittest.TestCase):
    def test_replaces_only_fusion_block_type(self) -> None:
        grid = Ours()
        token = TokenSpaceOurs()
        self.assertTrue(
            all(isinstance(block, CBAMConvCrossAttention) for block in grid.fusion_blocks)
        )
        self.assertTrue(
            all(isinstance(block, TokenSpaceCrossAttention) for block in token.fusion_blocks)
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in grid.parameters()),
            sum(parameter.numel() for parameter in token.parameters()),
        )
        self.assertTrue(
            all(
                isinstance(block.q_linear, nn.Linear)
                and isinstance(block.k_linear, nn.Linear)
                and isinstance(block.v_linear, nn.Linear)
                for block in token.fusion_blocks
            )
        )

    def test_forward_shapes_and_losses(self) -> None:
        module = TokenSpaceOurs()
        student = [torch.randn(2, 192, 14, 14) for _ in range(12)]
        teacher = [
            torch.randn(2, 16, 32, 32),
            torch.randn(2, 32, 16, 16),
            torch.randn(2, 64, 8, 8),
        ]
        alignment, fusion, aligned, fused, targets = module(student, teacher)
        expected = [(2, 16, 32, 32), (2, 32, 16, 16), (2, 64, 14, 14)]
        self.assertEqual([tuple(tensor.shape) for tensor in aligned], expected)
        self.assertEqual([tuple(tensor.shape) for tensor in fused], expected)
        self.assertEqual([tuple(tensor.shape) for tensor in targets], expected)
        self.assertTrue(torch.isfinite(alignment + fusion))

    def test_linear_token_block_is_equivalent_to_conv1x1_when_weights_match(
        self,
    ) -> None:
        """Document the exact mathematical relationship of this substitution."""

        torch.manual_seed(7)
        grid = CBAMConvCrossAttention(embed_dim=16, num_heads=4).eval()
        token = TokenSpaceCrossAttention(embed_dim=16, num_heads=4).eval()
        token.cbam.load_state_dict(grid.cbam.state_dict())
        with torch.no_grad():
            for linear, convolution in (
                (token.q_linear, grid.q_conv),
                (token.k_linear, grid.k_conv),
                (token.v_linear, grid.v_conv),
                (token.out_linear, grid.out_conv),
            ):
                linear.weight.copy_(convolution.weight[:, :, 0, 0])
                linear.bias.copy_(convolution.bias)

        student = torch.randn(2, 16, 5, 5)
        teacher = torch.randn(2, 16, 5, 5)
        with torch.no_grad():
            grid_output = grid(student, teacher)
            token_output = token(student, teacher)
        torch.testing.assert_close(
            token_output,
            grid_output,
            rtol=1e-5,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
