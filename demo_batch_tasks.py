#from main import *
#demo_generator('00d62c1b')

# import
from re_arc.main import *
from re_arc.utils import *

from tabulate import tabulate


from utils.data_preparation import prepare_input_output_pair
from utils.visualizers import visualize_full_transformation, print_sequence_info

# Set the key for the task
key = '00d62c1b'  # Example key, can be changed to other keys like '007bbfb7'

# Generate tasks
generated_tasks,input_grids,output_grids,input_sequences,output_sequences = generate_and_process_tasks(key, 3, plot=False, print_data=True)

# Transform it
sequences = prepare_input_output_pair(input_grids,output_grids)

# Print sequence info
print_sequence_info(input_grids,output_grids,sequences)

# Visualize full transformation
visualize_full_transformation(input_grids[0],output_grids[0],sequences[0])



