# Out-of-Distribution Sampling Implementation

## Overview

This implementation adds support for using original ARC evaluation tasks as out-of-distribution samples during both training and evaluation. This allows the model to be tested on truly unseen data that is different from the generated re-arc samples.

## Key Features

1. **Automatic Detection**: The system automatically detects if original ARC evaluation tasks are available
2. **Fallback Mechanism**: If out-of-distribution sampling fails, it gracefully falls back to generated samples
3. **Settings Integration**: Controlled via the `out_of_distribution` setting in `evaluation_settings`
4. **Comprehensive Support**: Works for both training and evaluation phases
5. **Multi-Encoder Support**: Compatible with both single and multi-encoder architectures

## Implementation Details

### 1. Core Functions Added

#### `load_and_process_original_arc_evaluation_tasks()` (re_arc/main.py)
- Loads original ARC evaluation tasks from `re_arc/arc_original/evaluation/`
- Processes them into the same format as generated samples
- Supports random sampling of examples per task
- Returns sequences compatible with the existing pipeline

#### `get_available_evaluation_tasks()` (re_arc/main.py)
- Scans the evaluation directory for available original ARC tasks
- Returns a list of task keys that can be used for out-of-distribution sampling

### 2. Modified Components

#### Evaluation (evaluation.py)
- Modified `main_test()` function to check for out-of-distribution setting
- Uses original ARC tasks when `out_of_distribution=True`
- Falls back to generated samples if out-of-distribution sampling fails

#### Training (training.py)
- Modified data generation for both single and multi-encoder training
- Supports out-of-distribution sampling in all training modes:
  - Single encoder training
  - Multi-encoder with mixed dataset
  - Multi-encoder with key-based splitting

#### Data Preparation (utils/data_preparation.py)
- Modified `_generate_key_data()` function to support out-of-distribution sampling
- Integrated with the key-based splitting for multi-encoder training

### 3. Settings Configuration

The feature is controlled by the `out_of_distribution` setting in `evaluation_settings`:

```json
{
  "evaluation_settings": {
    "out_of_distribution": true,
    // ... other settings
  }
}
```

## Usage

### Enabling Out-of-Distribution Sampling

1. Set `"out_of_distribution": true` in your settings file
2. Ensure original ARC evaluation tasks are available in `re_arc/arc_original/evaluation/`
3. Run training or evaluation as normal

### Disabling Out-of-Distribution Sampling

1. Set `"out_of_distribution": false` in your settings file
2. The system will use generated samples as before

## Data Structure Differences

### Original ARC Evaluation Tasks
- Located in `re_arc/arc_original/evaluation/`
- Structure: `{"train": [...], "test": [...]}`
- Each example: `{"input": [[...]], "output": [[...]]}`
- File naming: `{task_id}.json`

### Generated re-arc Tasks
- Located in `re_arc/re_arc/tasks/`
- Structure: `[{...}, {...}, ...]`
- Each example: `{"input": [[...]], "output": [[...]]}`
- File naming: `{generator_name}.json`

## Error Handling

The implementation includes comprehensive error handling:

1. **Missing Directory**: If `re_arc/arc_original/evaluation/` doesn't exist, falls back to generated samples
2. **No Tasks Available**: If no original ARC tasks are found, falls back to generated samples
3. **Insufficient Samples**: If not enough out-of-distribution samples are available, falls back to generated samples
4. **Loading Errors**: If individual tasks fail to load, continues with remaining tasks

## Testing

A test script `test_out_of_distribution.py` is provided to verify the implementation:

```bash
python test_out_of_distribution.py
```

This script tests:
- Availability of original ARC evaluation tasks
- Loading and processing of original tasks
- Settings integration
- Sequence format compatibility

## Benefits

1. **True Out-of-Distribution Testing**: Uses genuinely unseen data from the original ARC dataset
2. **Better Generalization Assessment**: Tests model performance on real-world ARC tasks
3. **Flexible Configuration**: Can be easily enabled/disabled via settings
4. **Robust Fallback**: Never breaks existing functionality
5. **Comprehensive Coverage**: Works across all training and evaluation modes

## Limitations

1. **Limited Original Tasks**: Only 400 original ARC evaluation tasks available
2. **Fixed Task Set**: Cannot generate new original ARC tasks
3. **Potential Overfitting**: If used for training, may lead to overfitting to the original task set
4. **Memory Usage**: Loading all original tasks may require significant memory

## Future Improvements

1. **Task Filtering**: Add ability to select specific original tasks
2. **Difficulty Stratification**: Group tasks by difficulty level
3. **Dynamic Loading**: Load tasks on-demand to reduce memory usage
4. **Task Augmentation**: Apply transformations to original tasks for more variety 