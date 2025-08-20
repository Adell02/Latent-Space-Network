import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from collections import defaultdict

def compute_task_distance_metrics(latent_data, task_keys, encoder_indices=None, 
                                distance_metric='cosine', normalize=True):
    """
    Compute distance metrics within and between tasks.
    
    Args:
        latent_data: List of latent vectors [N, latent_dim]
        task_keys: List of task keys corresponding to each latent
        encoder_indices: Optional list of encoder indices for multi-encoder models
        distance_metric: 'cosine' or 'euclidean'
        normalize: Whether to normalize latents before computing distances
    
    Returns:
        dict: Dictionary containing distance metrics
    """
    print(f"[DEBUG] compute_task_distance_metrics called with {len(latent_data) if latent_data is not None else 0} latents")
    
    if latent_data is None or len(latent_data) == 0 or not task_keys or len(task_keys) == 0:
        print(f"[DEBUG] Early return: latent_data empty or task_keys empty")
        return {}
    
    if len(latent_data) != len(task_keys):
        print(f"[DEBUG] Early return: latent_data length ({len(latent_data)}) != task_keys length ({len(task_keys)})")
        return {}
    
    if encoder_indices is not None and len(encoder_indices) != len(latent_data):
        print(f"[DEBUG] Early return: encoder_indices length ({len(encoder_indices)}) != latent_data length ({len(latent_data)})")
        return {}
    
    # Convert to numpy array
    print(f"[DEBUG] Converting latent_data to numpy array...")
    latents = np.array(latent_data)
    print(f"[DEBUG] Latents shape: {latents.shape}")
    
    # Normalize latents if requested
    if normalize:
        print(f"[DEBUG] Normalizing latents...")
        latents = latents / (np.linalg.norm(latents, axis=1, keepdims=True) + 1e-8)
    
    # Group latents by task key
    print(f"[DEBUG] Grouping latents by task key...")
    task_to_latents = defaultdict(list)
    task_to_indices = defaultdict(list)
    
    for i, (latent, key) in enumerate(zip(latents, task_keys)):
        task_to_latents[key].append(latent)
        task_to_indices[key].append(i)
    
    print(f"[DEBUG] Found {len(task_to_latents)} unique tasks")
    
    # Convert to numpy arrays
    print(f"[DEBUG] Converting task latents to numpy arrays...")
    for key in task_to_latents:
        task_to_latents[key] = np.array(task_to_latents[key])
    
    # Compute distance matrix
    print(f"[DEBUG] Computing {distance_metric} distance matrix...")
    if distance_metric == 'cosine':
        distance_matrix = cosine_distances(latents)
    else:  # euclidean
        distance_matrix = euclidean_distances(latents)
    print(f"[DEBUG] Distance matrix shape: {distance_matrix.shape}")
    
    # Compute within-task distances
    within_task_distances = []
    within_task_counts = []
    
    for key, indices in task_to_indices.items():
        if len(indices) > 1:
            # Get submatrix for this task
            task_distances = distance_matrix[np.ix_(indices, indices)]
            # Get upper triangular (excluding diagonal)
            upper_tri = task_distances[np.triu_indices(len(indices), k=1)]
            within_task_distances.extend(upper_tri.tolist())
            within_task_counts.append(len(upper_tri))
    
    # Compute between-task distances
    between_task_distances = []
    between_task_counts = []
    
    task_keys_list = list(task_to_latents.keys())
    for i, key1 in enumerate(task_keys_list):
        for j, key2 in enumerate(task_keys_list[i+1:], i+1):
            indices1 = task_to_indices[key1]
            indices2 = task_to_indices[key2]
            
            # Get submatrix for between-task distances
            between_distances = distance_matrix[np.ix_(indices1, indices2)]
            between_task_distances.extend(between_distances.flatten().tolist())
            between_task_counts.append(len(between_distances.flatten()))
    
    # Compute statistics
    metrics = {}
    
    if len(within_task_distances) > 0:
        within_distances = np.array(within_task_distances)
        metrics.update({
            'within_task_mean': float(np.mean(within_distances)),
            'within_task_std': float(np.std(within_distances)),
            'within_task_min': float(np.min(within_distances)),
            'within_task_max': float(np.max(within_distances)),
            'within_task_median': float(np.median(within_distances)),
            'within_task_count': len(within_distances)
        })
    
    if len(between_task_distances) > 0:
        between_distances = np.array(between_task_distances)
        metrics.update({
            'between_task_mean': float(np.mean(between_distances)),
            'between_task_std': float(np.std(between_distances)),
            'between_task_min': float(np.min(between_distances)),
            'between_task_max': float(np.max(between_distances)),
            'between_task_median': float(np.median(between_distances)),
            'between_task_count': len(between_distances)
        })
    
    # Compute separation ratio (between_task / within_task)
    if len(within_task_distances) > 0 and len(between_task_distances) > 0:
        within_mean = np.mean(within_task_distances)
        between_mean = np.mean(between_task_distances)
        if within_mean > 0:
            metrics['separation_ratio'] = float(between_mean / within_mean)
        else:
            metrics['separation_ratio'] = 0.0
    
    # Add task-level statistics
    metrics['num_tasks'] = len(task_to_latents)
    metrics['total_samples'] = len(latents)
    
    # Add encoder-specific metrics if available
    if encoder_indices is not None and len(encoder_indices) > 0:
        encoder_to_metrics = defaultdict(lambda: defaultdict(list))
        
        for i, (latent, key, enc_idx) in enumerate(zip(latents, task_keys, encoder_indices)):
            if enc_idx is not None:
                encoder_to_metrics[enc_idx]['latents'].append(latent)
                encoder_to_metrics[enc_idx]['keys'].append(key)
        
        for enc_idx, enc_data in encoder_to_metrics.items():
            if len(enc_data['latents']) > 1:
                enc_latents = np.array(enc_data['latents'])
                enc_keys = enc_data['keys']
                
                # Compute encoder-specific distances
                enc_metrics = compute_task_distance_metrics(
                    latent_data=enc_latents, 
                    task_keys=enc_keys, 
                    encoder_indices=None,  # No nested encoder indices for this recursive call
                    distance_metric=distance_metric, 
                    normalize=normalize
                )
                
                # Add encoder prefix
                for key, value in enc_metrics.items():
                    metrics[f'encoder_{enc_idx}_{key}'] = value
    
    return metrics
