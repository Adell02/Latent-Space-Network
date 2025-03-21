from models.initial_model import main
from utils.visualizers import visualize_all_results

# Run the training for the initial model
print("Starting training...")
results, model = main()
print("Training complete. Results saved in the run directory.")

print("Visualizing results...")
visualize_all_results(results)
print("Results visualized.")

