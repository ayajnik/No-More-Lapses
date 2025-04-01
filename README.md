# No More Lapses

**No More Lapses** is an AI-driven application designed to predict and analyze policy lapse rates, providing valuable insights to insurance companies and policyholders. By leveraging advanced machine learning techniques, the application identifies high-risk policies and offers recommendations to mitigate potential lapses.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Policy Lapse Prediction**: Utilizes machine learning models to predict the likelihood of policy lapses based on historical data.
- **Data Analysis Tools**: Provides tools to read, process, and analyze CSV data related to policy predictions.
- **Web Research Integration**: Incorporates web search capabilities to gather external information relevant to policy analysis.
- **Codebase Analysis**: Includes utilities to inspect and list source code modules, aiding in better understanding and maintenance of the codebase.

## Installation

To set up the **No More Lapses** application, follow these steps:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/yourusername/no-more-lapses.git
   cd no-more-lapses

2. Install dependencies

  `pip install -r requirements.txt`

# Project Workflow

This is more of a technical section where we will describe how are we approaching the project at every stage of the workflow.

1. Update the params.yml file
2. Update the config.yml file
3. Update the entity folder which is defining the variables/ hyper-parameters defined for every stage of the pipeline
4. Update the configuration folder
5. Update the components folder which contains single script for each stage
6. Update the pipeline folder folder where we assemble all of the components of the pipeline.
7. Create the dvc pipeline
