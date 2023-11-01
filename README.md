# ML Models Training: Optuna + SHAP + Calibration

<!--[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.txt)-->

## Overview

This project is currently in development and aims to aggregate a set of training scripts created for various machine learning projects. The primary focus is on unsupervised model training (full cycle) to expedite the model development process. Additionally, the project includes a custom algorithm for feature selection
based on SHAP values, chosen for its efficiency. Furthermore, a 'patience' feature has been implemented with Optuna to accelerate optimization tasks. Please note that the code is a work in progress, and further improvements are planned.

## Key Features

- Unsupervised model training to streamline machine learning model development.
- Custom algorithm for feature selection using SHAP values.
- Integration of 'patience' feature in Optuna for faster hyperparameter optimization.
- Added calibration step to improve model performance for classification tasks.
- Customizable and adaptable for different machine learning tasks and datasets.
- Ongoing development with planned enhancements and improvements.

## Getting Started

...
<!--Follow these initial instructions to set up and begin using the project for your machine learning tasks.-->

### Prerequisites

- Python 3.8 (did not test other versions yet)

<!--- [Virtual environment](https://docs.python.org/3/library/venv.html) (recommended)-->

## Development Roadmap

The project is actively in development, with planned enhancements that include:

- Improving data preprocessing steps and including into optimization process.
- Including sample_weights calculations for imbalanced regression datasets.
- Add some other optimizers.
- Improve feature selection.
- Speed-up and fix calibration.

## Acknowledgments

- [Optuna](https://optuna.org/): The hyperparameter optimization library used in this project.
- [SHAP](https://shap.readthedocs.io/en/stable/): The SHAP library for feature importance calculation.

## Contact

If you have any questions or suggestions regarding this project, please feel free to contact [me](mailto:avchauzov@gmail.com).

<!--### Installation

1. Clone the repository to your local machine:
   
   ```bash
   git clone https://github.com/your-username/your-project.git

Navigate to the project directory:

bash
Copy code
cd your-project
Create and activate a virtual environment (optional but recommended):

bash
Copy code
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
Install project dependencies from requirements.txt:

bash
Copy code
pip install -r requirements.txt
Usage
Explore and modify the provided scripts in the respective directories to suit your machine learning projects.
Consult the documentation and comments within each script for guidance on usage.
Customize and extend the scripts as necessary for your specific project requirements.
Stay updated with ongoing development and planned improvements.

Contributing
Contributions are welcome! If you would like to contribute to this project, please review the CONTRIBUTING.md file for guidelines on how to get started.

License
This project is licensed under the MIT License - see the LICENSE.txt file for details.-->
