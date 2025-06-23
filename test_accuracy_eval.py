#!/usr/bin/env python3

"""
Test script to verify the new accuracy evaluation function works correctly.
"""

import torch
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.base_model import LatentProgramNetwork
from training import evaluate_accuracy

def test_accuracy_evaluation():
    """Test the accuracy evaluation function for both single and multi-encoder models."""
    print("Testing accuracy evaluation functions...")
    
    # Create sample data
    batch_size = 4
    input_seq = torch.randn(batch_size, 902)
    target_seq = torch.randn(batch_size, 902)
    
    # Make target_seq have valid shape and grid data
    for i in range(batch_size):
        target_seq[i, 900] = 2  # rows
        target_seq[i, 901] = 3  # cols
        target_seq[i, :6] = torch.randint(0, 10, (6,))  # 6 grid pixels for 2x3
    
    # Create a simple dataloader-like structure
    test_dataloader = [(input_seq, target_seq)]
    device = torch.device('cpu')
    
    print("\n=== Testing Single Encoder ===")
    # Test single encoder model
    single_encoder_model = LatentProgramNetwork(num_encoders=1)
    single_encoder_model.eval()
    
    try:
        single_accuracy = evaluate_accuracy(
            single_encoder_model, test_dataloader, device,
            is_multi_encoder=False, encoder_idx=None,
            optimize_z=False, logger=None
        )
        print(f"✓ Single encoder accuracy evaluation successful")
        print(f"  Shape: {single_accuracy['shape_accuracy']:.4f}")
        print(f"  Grid: {single_accuracy['grid_accuracy']:.4f}")
        print(f"  Overall: {single_accuracy['overall_accuracy']:.4f}")
        print(f"  Sample Exact: {single_accuracy['sample_exact_accuracy']:.4f}")
        print(f"  Evaluation Name: {single_accuracy['evaluation_name']}")
    except Exception as e:
        print(f"✗ Single encoder accuracy evaluation failed: {e}")
        return False
    
    print("\n=== Testing Multi-Encoder ===")
    # Test multi-encoder model
    multi_encoder_model = LatentProgramNetwork(num_encoders=3)
    multi_encoder_model.eval()
    
    # Test individual encoder evaluation
    for encoder_idx in range(3):
        try:
            encoder_accuracy = evaluate_accuracy(
                multi_encoder_model, test_dataloader, device,
                is_multi_encoder=True, encoder_idx=encoder_idx,
                optimize_z=False, logger=None
            )
            print(f"✓ Encoder {encoder_idx} accuracy evaluation successful")
            print(f"  Shape: {encoder_accuracy['shape_accuracy']:.4f}")
            print(f"  Grid: {encoder_accuracy['grid_accuracy']:.4f}")
            print(f"  Overall: {encoder_accuracy['overall_accuracy']:.4f}")
            print(f"  Sample Exact: {encoder_accuracy['sample_exact_accuracy']:.4f}")
            print(f"  Evaluation Name: {encoder_accuracy['evaluation_name']}")
        except Exception as e:
            print(f"✗ Encoder {encoder_idx} accuracy evaluation failed: {e}")
            return False
    
    # Test PoE evaluation
    try:
        poe_accuracy = evaluate_accuracy(
            multi_encoder_model, test_dataloader, device,
            is_multi_encoder=True, encoder_idx=None,
            optimize_z=False, logger=None
        )
        print(f"✓ PoE accuracy evaluation successful")
        print(f"  Shape: {poe_accuracy['shape_accuracy']:.4f}")
        print(f"  Grid: {poe_accuracy['grid_accuracy']:.4f}")
        print(f"  Overall: {poe_accuracy['overall_accuracy']:.4f}")
        print(f"  Sample Exact: {poe_accuracy['sample_exact_accuracy']:.4f}")
        print(f"  Evaluation Name: {poe_accuracy['evaluation_name']}")
    except Exception as e:
        print(f"✗ PoE accuracy evaluation failed: {e}")
        return False
    
    print("\n✓ All accuracy evaluation tests passed!")
    return True

if __name__ == "__main__":
    test_accuracy_evaluation() 