import numpy as np

import os
import sys
import random

# add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from re_arc import generators
from re_arc.utils import plot_task

example1 = getattr(generators, 'generate_pattern_task')

n_samples = 2
n_pairs = 4

for n in range(n_samples):
    _PATTERN_COLORS = []
    for i in range(4):
        row = []
        for j in range(4):
            # Generate random colors (0-9) for the pattern
            row.append(random.randint(0, 9))
        _PATTERN_COLORS.extend(row)

    generated_examples = [example1(0, 1, _PATTERN_COLORS) for _ in range(n_pairs)]
    
    plot_task(generated_examples)
