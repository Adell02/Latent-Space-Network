#!/bin/bash

# Example script for running evaluate_all_artifacts.py
# Make sure to activate your virtual environment first

echo "=== LPN Model Evaluation Examples ==="
echo ""

# Example 1: Evaluate multiple checkpoints of the same model
echo "Example 1: Multiple checkpoints of same model"
echo "python evaluate_all_artifacts.py --artifacts \\"
echo "  ga624-imperial-college-london/LPN_specialist_paper/model:epoch_10 \\"
echo "  ga624-imperial-college-london/LPN_specialist_paper/model:epoch_20 \\"
echo "  ga624-imperial-college-london/LPN_specialist_paper/model:final"
echo ""

# Example 2: Use evaluation config file
echo "Example 2: Using evaluation config file"
echo "python evaluate_all_artifacts.py --artifacts \\"
echo "  ga624-imperial-college-london/LPN_specialist_paper/model:epoch_20 \\"
echo "  ga624-imperial-college-london/LPN_specialist_paper/model:final \\"
echo "  --config evaluation_config.json"
echo ""

# Example 3: Specialist models
echo "Example 3: Specialist models"
echo "python evaluate_all_artifacts.py --artifacts \\"
echo "  ga624-imperial-college-london/LPN_specialist_paper/specialist_model:latest \\"
echo "  --specialist --config evaluation_config.json"
echo ""

# Example 4: Custom dataset size
echo "Example 4: Custom dataset size"
echo "python evaluate_all_artifacts.py --artifacts \\"
echo "  ga624-imperial-college-london/LPN_specialist_paper/model:latest \\"
echo "  --n_samples 10 --n_queries 20 \\"
echo "  --config evaluation_config.json"
echo ""

# Example 5: Different models comparison
echo "Example 5: Different models comparison"
echo "python evaluate_all_artifacts.py --artifacts \\"
echo "  entity1/project1/single_encoder:latest \\"
echo "  entity2/project2/specialist_model:v1 \\"
echo "  --config evaluation_config.json"
echo ""

# Example 6: No WandB logging
echo "Example 6: No WandB logging"
echo "python evaluate_all_artifacts.py --artifacts \\"
echo "  ga624-imperial-college-london/LPN_specialist_paper/model:latest \\"
echo "  --no_wandb --config evaluation_config.json"
echo ""

echo "=== Configuration File ==="
echo "The evaluation_config.json file contains:"
echo "- Evaluation settings (keys, samples, queries)"
echo "- Data settings (batch size, seeds)"
echo "- Model settings (architecture parameters)"
echo "- Optimization settings (steps, learning rate)"
echo "- Visualization settings (t-SNE, plots)"
echo "- WandB settings (project, entity, logging)"
echo "- Output settings (formats, compression)"
echo "- Comparison settings (metrics, ranking)"
echo "- Dataset settings (unified, caching)"
echo "- Performance settings (batching, memory)"
echo ""

echo "=== Output Files ==="
echo "The script generates:"
echo "- all_model_results.json: Detailed results per model"
echo "- unified_metrics.json: Comparable metrics across models"
echo "- model_metadata.json: Model information and artifacts"
echo "- unified_dataset_info.json: Dataset statistics"
echo "- comparative_accuracies.png: Bar chart comparison"
echo "- support_vs_query_comparison.png: Performance comparison"
echo ""

echo "=== WandB Integration ==="
echo "Results are logged to WandB with:"
echo "- Unified metrics under unified_metrics/*"
echo "- Dataset info under unified_dataset/*"
echo "- Comparative visualizations under comparative_visualizations/*"
echo "- Model-specific prefixes for easy comparison"
echo ""

echo "=== Tips ==="
echo "1. Use the same config file for consistent evaluation"
echo "2. Set fixed seeds for reproducible results"
echo "3. Use unified dataset for fair comparison"
echo "4. Check WandB dashboard for interactive comparisons"
echo "5. Adjust n_samples and n_queries based on your needs"
echo ""
