import torch

from utils.model_utils import set_seed, prepare_dataloader
from re_arc.main import generate_and_process_tasks
from utils.settings_manager import settings
from utils.latent_functions import get_optimized_z
from models.base_model import compute_loss


training_settings = settings.get_training_settings()
BATCH_SIZE = training_settings['batch_size']


##############################
# Run Inference
##############################

def main_test(model, keys, n_samples, n_queries, seed, device='cuda'):
    """
    Generate new data and evaluate the model on it.
    
    Args:
        model: The trained model
        keys: List of problem keys
        n_samples: List of numbers of input-output pairs to generate
        n_queries: List of numbers of queries to do inference
        device: Device to run evaluation on
    
    Returns:
        dict: Dictionary containing evaluation results for each key and n_samples   
    """

    set_seed(seed)
    results = {}
    
    for key in keys:
        results[key] = {}
        print(f"\nEvaluating key {key} with {n_samples} samples and {n_queries} queries...")
        
        # Generate new data
        _, _, _, input_samples_sequences, output_samples_sequences = generate_and_process_tasks(key, n_samples)
        samples_dataloader = prepare_dataloader(input_samples_sequences, output_samples_sequences, BATCH_SIZE)

        # Generate queries
        _, _, _, input_queries_sequences, output_queries_sequences = generate_and_process_tasks(key, n_queries)
        queries_dataloader = prepare_dataloader(input_queries_sequences, output_queries_sequences, BATCH_SIZE)

        # Evaluate overall performance
        metrics = evaluate_model(model, samples_dataloader, queries_dataloader, device=device)

        results[key] = metrics
        results[key]['reconstruction_results'] = {
            'input_samples_sequences': input_samples_sequences,
            'output_samples_sequences': output_samples_sequences,
            'input_queries_sequences': input_queries_sequences,
            'output_queries_sequences': output_queries_sequences,
            'support_reconstructions': results[key]['reconstruction_results']['support_reconstructions'],
            'query_reconstructions': results[key]['reconstruction_results']['query_reconstructions']
        }

    return results


def evaluate_model(model, samples_dataloader, queries_dataloader, device='cuda'):
    """
    Evaluate model performance on a dataloader.
    # ... (rest of docstring) ...
    """
    latent_optimization = settings.get_latent_optimization()
    optimize_z_inference = latent_optimization['inference']['enabled']

    model.eval()
    shape_correct, shape_tokens = 0, 0
    grid_correct, grid_tokens = 0, 0
    sample_exact_correct = 0
    total_samples = 0
    
    support_losses = []
    query_losses = []
    support_reconstructions = []
    query_reconstructions = []
    z_optimization_logs = []

    z_from_last_sample_batch = None # Will store z from the last processed sample batch

    for batch_input_s, batch_target_s in samples_dataloader:
        batch_input_s = batch_input_s.to(device)
        batch_target_s = batch_target_s.to(device)
        
        current_z_for_this_sample_batch = None
        if optimize_z_inference:
            current_z_for_this_sample_batch, losses_opt = get_optimized_z(
                model, batch_input_s, batch_target_s, context='inference'
            )
            if losses_opt is not None: # get_optimized_z can return (z, None)
                 z_optimization_logs.append(losses_opt)
        else:
            with torch.no_grad():
                mu, log_var = model.encoder(batch_input_s, batch_target_s)
                current_z_for_this_sample_batch = model.reparameterize(mu, log_var)
        
        z_from_last_sample_batch = current_z_for_this_sample_batch

        # Calculate support loss (original way, which re-derives z internally via model call)
        # This part is not directly related to the RuntimeError but is a point of consistency.
        s_loss_val = compute_loss(model, batch_input_s, batch_target_s)
        support_losses.append(s_loss_val.item())

        # Calculate support reconstructions using the specific z for this sample batch
        if current_z_for_this_sample_batch is not None:
            with torch.no_grad():
                # 'current_z_for_this_sample_batch' already has the correct batch dim for these samples
                shape_logits_s, grid_logits_s = model.decoder(current_z_for_this_sample_batch, batch_input_s, target_seq=batch_target_s)
                support_reconstructions.append((
                    shape_logits_s.cpu(),
                    grid_logits_s.cpu()
                ))
        else:
             support_reconstructions.append((None, None)) # Should not happen if dataloader has items

    # After processing all sample batches:
    if z_from_last_sample_batch is None:
        # Check if queries_dataloader has any data. list(dataloader) can be slow if it's large.
        # A more efficient check might be needed if performance is critical here.
        # For now, a simple check or assuming queries_dataloader won't run if samples are empty.
        # If there are queries to process, but no z from samples, it's an issue.
        has_queries = False
        for _ in queries_dataloader: # Check if dataloader is not empty
            has_queries = True
            break
        
        if has_queries:
            print("Error in evaluate_model: No z obtained from samples_dataloader, but queries exist. Cannot proceed.")
            return {
                'metrics': {'error': 'No z from samples for query evaluation', 'support_loss': sum(support_losses)/len(support_losses) if support_losses else 0, 'query_loss': 0, 'shape_accuracy': 0, 'grid_accuracy': 0, 'overall_accuracy': 0, 'sample_exact_accuracy': 0, 'losses_gradient_ascent': z_optimization_logs, 'used_latent_optimization': optimize_z_inference},
                'reconstruction_results': {'support_reconstructions': support_reconstructions, 'query_reconstructions': []}
            }
        else: # No samples and no queries
            print("Warning in evaluate_model: samples_dataloader was empty. No z obtained. No queries to process.")
            # Fall through, query loop won't run.

    # If optimize_z_inference was false for the entire run, and no logs were added
    # (e.g., if samples_dataloader itself was empty or if no optimization happened)
    if not optimize_z_inference and not z_optimization_logs:
        z_optimization_logs.append(None)

    z_for_queries_prototype = None
    if z_from_last_sample_batch is not None:
        # Average the z from the last sample batch to create a [1, latent_dim] prototype
        z_for_queries_prototype = z_from_last_sample_batch.mean(dim=0, keepdim=True)

    for batch_input_q, batch_target_q in queries_dataloader:
        batch_input_q = batch_input_q.to(device)
        batch_target_q = batch_target_q.to(device)
        query_batch_size = batch_input_q.size(0)

        if z_for_queries_prototype is None:
            # This implies z_from_last_sample_batch was None, and the earlier check should have caught it if queries exist.
            # This is a safeguard.
            print("Critical Error: z_for_queries_prototype is None inside query loop. Skipping query batch.")
            continue
            
        with torch.no_grad():
            # Expand the [1, latent_dim] prototype to [query_batch_size, latent_dim]
            z_query_expanded = z_for_queries_prototype.expand(query_batch_size, -1)
            
            shape_logits, grid_logits = model.decoder(z_query_expanded, batch_input_q, target_seq=batch_target_q)
            
            # Compute query loss (original way, re-derives z from query data)
            q_loss_val = compute_loss(model, batch_input_q, batch_target_q)
            query_losses.append(q_loss_val.item())
            
            query_reconstructions.append((
                shape_logits.cpu(), # These are from z_query_expanded
                grid_logits.cpu()   # These are from z_query_expanded
            ))

            shape_pred = shape_logits.argmax(dim=-1)
            grid_pred = grid_logits.argmax(dim=-1)
            shape_tgt = batch_target_q[:, 900:902].long()
            grid_tgt = batch_target_q[:, :900].long()

            shape_correct += (shape_pred == shape_tgt).sum().item()
            shape_tokens += shape_tgt.numel()

            for i in range(query_batch_size):
                tgt_rows = int(batch_target_q[i, 900].item())
                tgt_cols = int(batch_target_q[i, 901].item())
                active_pixels = tgt_rows * tgt_cols
                if active_pixels > 0 : # Check before indexing grid_pred and grid_tgt
                    grid_correct += (grid_pred[i, :active_pixels] == grid_tgt[i, :active_pixels]).sum().item()
                    grid_tokens += active_pixels
                    if torch.all(shape_pred[i] == shape_tgt[i]) and torch.all(grid_pred[i, :active_pixels] == grid_tgt[i, :active_pixels]):
                        sample_exact_correct += 1
                elif torch.all(shape_pred[i] == shape_tgt[i]): # If grid is empty, only shape matters
                     sample_exact_correct +=1


            total_samples += query_batch_size

    avg_support_loss = sum(support_losses) / len(support_losses) if support_losses else 0.0
    avg_query_loss = sum(query_losses) / len(query_losses) if query_losses else 0.0

    return {
        'metrics': {
            'support_loss': avg_support_loss,
            'query_loss': avg_query_loss,
            'shape_accuracy': shape_correct / shape_tokens if shape_tokens > 0 else 0.0,
            'grid_accuracy': grid_correct / grid_tokens if grid_tokens > 0 else 0.0,
            'overall_accuracy': (shape_correct + grid_correct) / (shape_tokens + grid_tokens) if (shape_tokens + grid_tokens) > 0 else 0.0,
            'sample_exact_accuracy': sample_exact_correct / total_samples if total_samples > 0 else 0.0,
            'losses_gradient_ascent': z_optimization_logs,
            'used_latent_optimization': optimize_z_inference,
        },
        'reconstruction_results': {
            'support_reconstructions': support_reconstructions,
            'query_reconstructions': query_reconstructions,
        }
    }