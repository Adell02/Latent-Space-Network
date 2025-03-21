# Latent Space Search

This repository contains code and resources for exploring latent program spaces, as described in the paper "Searching Latent Program Spaces" by Bonnet and Macfarlane (2024).

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The project focuses on searching latent spaces of programs, leveraging machine learning models to generate, evaluate, and refine program structures. The goal is to explore how latent spaces can be used to discover novel program solutions.

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/latent-space-search.git
cd latent-space-search
pip install -r requirements.txt
```

## Usage

The main functionality of the project is encapsulated in the `re_arc` module, which includes various generators and verifiers for program tasks. You can run the demo script to see the project in action:

```bash
python demo_batch_tasks.py
```

## Project Structure

- **re_arc/**: Contains the main modules for generating and processing tasks.
  - `main.py`: Main script for task generation and evaluation.
  - `generators.py`: Functions to generate tasks with varying difficulty.
  - `dsl.py`: Domain-specific language functions for task manipulation.
- **models/**: Contains the machine learning models used for latent space exploration.
  - `initial_model.py`: Defines the architecture and training routines for the models.
- **utils/**: Utility functions for data preparation and processing.
  - `data_preparation.py`: Functions to transform and prepare data for model input.
- **runs_re_arc/**: Directory for storing experiment runs and results.
- **requirements.txt**: Lists the Python dependencies required for the project.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes. Ensure that your code adheres to the project's coding standards and includes appropriate tests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
