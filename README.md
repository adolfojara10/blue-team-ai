<div id="top">

<!-- HEADER STYLE: CLASSIC -->
<div align="center">

<img src="readmeai/assets/logos/purple.svg" width="30%" style="position: relative; top: 0; right: 0;" alt="Project Logo"/>

# FINAL-FILTER

<em>Empowering Safe Conversations, One Embedding at a Time</em>

<!-- BADGES -->
<!-- local repository, no metadata badges. -->

<em>Built with the tools and technologies:</em>

<img src="https://img.shields.io/badge/JSON-000000.svg?style=default&logo=JSON&logoColor=white" alt="JSON">
<img src="https://img.shields.io/badge/Ollama-000000.svg?style=default&logo=Ollama&logoColor=white" alt="Ollama">
<img src="https://img.shields.io/badge/NumPy-013243.svg?style=default&logo=NumPy&logoColor=white" alt="NumPy">
<img src="https://img.shields.io/badge/Python-3776AB.svg?style=default&logo=Python&logoColor=white" alt="Python">

</div>
<br>

---

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
    - [Project Index](#project-index)
- [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Usage](#usage)
    - [Testing](#testing)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview
The Final Filter project is an ensemble machine learning application designed to classify text inputs using multiple models in parallel. It leverages the Ollama API for obtaining embeddings and utilizes a logging system for tracking predictions and results.
Also, it implements Llama Guard 3 Model to filter the input in case the ensemble models don't agree on the classification and also it filters the output of the model. All models run using Ollama python binding.


---

## Features

| **Component** | **Details** |
| --- | --- |
| **Architecture** | * Microservices-based architecture with multiple N+1 layers, allowing for scalability and fault tolerance. |
| **Code Quality** | * Follows PEP 8 conventions, using consistent naming conventions and code formatting throughout the project. |
| **Documentation** | * Official documentation is available in the `final-filter` README file and various Markdown files within the project directory. |
| **Integrations** | * Includes integrations with popular libraries such as NumPy, pandas, and scikit-learn for data manipulation and analysis. |
| **Modularity** | * Is designed to be modular, allowing developers to easily swap out or add new components without affecting the overall system. |
| **Testing** | * Comprehensive unit testing using Pytest and coverage reports to ensure code reliability. |
| **Performance** | * Optimized for efficiency, with careful consideration given to data types, algorithms, and caching strategies. |
| **Security** | * Adheres to basic security best practices such as input validation, logging, and secure file handling. |
| **Dependencies** | * Managed using `pip` and maintained in the `requirements.txt` file, ensuring ease of installation and updates. |
| **Scalability** | * Designed for horizontal scaling with distributed computing and load balancing capabilities. |

### Additional Notes

* The project uses containerization (Docker) to ensure consistent environment configurations across development, testing, and production environments.
* Asynchronous programming with `asyncio` is used in key components for improved responsiveness and concurrent execution.
* The use of `joblib` library facilitates parallel processing and improves computational efficiency.

---

## Project Structure

```sh
└── final-filter/
    ├── README.md
    ├── inference_model
    │   ├── __init__.py
    │   ├── __pycache__
    │   ├── inference.py
    │   └── interactive_chat.py
    ├── input
    │   ├── __init__.py
    │   ├── __pycache__
    │   ├── config.py
    │   ├── embedding_client.py
    │   ├── ensemble.py
    │   ├── logging_utils.py
    │   ├── main.py
    │   ├── model_config.py
    │   └── model_wrapper.py
    ├── llm_filter
    │   ├── __init__.py
    │   ├── __pycache__
    │   └── llm_classifier.py
    ├── logs
    │   ├── ensemble_classifier_20250806_152719.log
    │   ├── ensemble_classifier_20250806_152945.log
    │   ├── ensemble_classifier_20250806_153000.log
    │   └── ensemble_classifier_20250807_100423.log
    ├── models
    │   ├── lightgbm_model_allminilm_without_pca.pkl
    │   ├── lightgbm_model_granite_without_pca.pkl
    │   ├── lightgbm_model_snowflake_without_pca.pkl
    │   ├── xgboost_model_allminilm_without_pca.pkl
    │   ├── xgboost_model_granite_without_pca.pkl
    │   └── xgboost_model_snowflake_without_pca.pkl
    └── requirements.txt
```

### Project Index

<details open>
	<summary><b><code>FINAL-FILTER/</code></b></summary>
	<!-- __root__ Submodule -->
	<details>
		<summary><b>__root__</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ __root__</b></code>
			<table style='width: 100%; border-collapse: collapse;'>
			<thead>
				<tr style='background-color: #f8f9fa;'>
					<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
					<th style='text-align: left; padding: 8px;'>Summary</th>
				</tr>
			</thead>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/home/adolfo/Desktop/cybermacs/cybermacs/internship/input-filter/embeddings-RF-filter/final-filter/final-filter/blob/master/requirements.txt'>requirements.txt</a></b></td>
					<td style='padding: 8px;'>- Architecture Overview**Provides the projects dependency requirements, identifying critical libraries necessary for execution<br>- Ensures seamless integration of various modules across the entire codebase architecture<br>- Facilitates efficient development, testing, and deployment by defining the necessary tools and technologies required to build and maintain the project<br>- Serves as a foundation for reliable and stable system functionality.</td>
				</tr>
			</table>
		</blockquote>
	</details>
	<!-- llm_filter Submodule -->
	<details>
		<summary><b>llm_filter</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ llm_filter</b></code>
			<table style='width: 100%; border-collapse: collapse;'>
			<thead>
				<tr style='background-color: #f8f9fa;'>
					<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
					<th style='text-align: left; padding: 8px;'>Summary</th>
				</tr>
			</thead>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/home/adolfo/Desktop/cybermacs/cybermacs/internship/input-filter/embeddings-RF-filter/final-filter/final-filter/blob/master/llm_filter/llm_classifier.py'>llm_classifier.py</a></b></td>
					<td style='padding: 8px;'>- Classifies and interprets responses from an LLM model**, categorizing them into predefined safety labels with corresponding hazard codes and descriptions<br>- It sends requests to the llama-guard3 model, receiving a response that's then processed through a classification system to generate a human-readable result<br>- The classifier integrates safety information for various categories of crimes, emergencies, and other hazards, providing context to user input or responses.</td>
				</tr>
			</table>
		</blockquote>
	</details>
	<!-- inference_model Submodule -->
	<details>
		<summary><b>inference_model</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ inference_model</b></code>
			<table style='width: 100%; border-collapse: collapse;'>
			<thead>
				<tr style='background-color: #f8f9fa;'>
					<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
					<th style='text-align: left; padding: 8px;'>Summary</th>
				</tr>
			</thead>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/home/adolfo/Desktop/cybermacs/cybermacs/internship/input-filter/embeddings-RF-filter/final-filter/final-filter/blob/master/inference_model/inference.py'>inference.py</a></b></td>
					<td style='padding: 8px;'>- The <code>inference.py</code> file serves as the main entry point for interacting with the Ollama model via its dedicated API<br>- It enables users to send prompts and receive responses, allowing for natural language processing and conversation-based applications<br>- By utilizing this interface, developers can seamlessly integrate the powerful chat functionality into their projects.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/home/adolfo/Desktop/cybermacs/cybermacs/internship/input-filter/embeddings-RF-filter/final-filter/final-filter/blob/master/inference_model/interactive_chat.py'>interactive_chat.py</a></b></td>
					<td style='padding: 8px;'>- Interactive Chatbot Model Executes**The interactive chatbot model executed by <code>interactive_chat.py</code> achieves a conversational interface that classifies and responds to user input while maintaining safety<br>- It uses machine learning models to analyze and filter content, preventing the generation of unsafe responses<br>- The chatbots workflow includes user input processing, classification, response generation, and conversation logging, ensuring a safe and informative interaction experience for users.</td>
				</tr>
			</table>
		</blockquote>
	</details>
	<!-- input Submodule -->
	<details>
		<summary><b>input</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ input</b></code>
			<table style='width: 100%; border-collapse: collapse;'>
			<thead>
				<tr style='background-color: #f8f9fa;'>
					<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
					<th style='text-align: left; padding: 8px;'>Summary</th>
				</tr>
			</thead>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/home/adolfo/Desktop/cybermacs/cybermacs/internship/input-filter/embeddings-RF-filter/final-filter/final-filter/blob/master/input/model_config.py'>model_config.py</a></b></td>
					<td style='padding: 8px;'>- Represents the configuration model settings across multiple machine learning models<br>- Defines a dataclass to store key parameters such as model name, embedding model choice, and specific file paths for trained models (LightGBM and XGBoost)<br>- The model configurations are used throughout the codebase to determine which machine learning algorithms to use, effectively serving as the central hub for decision-making on model selection and training.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/home/adolfo/Desktop/cybermacs/cybermacs/internship/input-filter/embeddings-RF-filter/final-filter/final-filter/blob/master/input/config.py'>config.py</a></b></td>
					<td style='padding: 8px;'>- Configures Ensemble Classifier Models**The <code>config.py</code> file configurations the ensemble classifier models by defining three model configurations with varying embedding and machine learning settings<br>- Each configuration includes a model and its corresponding weights, as well as file paths for training data and log files<br>- The configurations are designed to work together to improve model performance, with multiple file templates providing flexibility in logging and output.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/home/adolfo/Desktop/cybermacs/cybermacs/internship/input-filter/embeddings-RF-filter/final-filter/final-filter/blob/master/input/ensemble.py'>ensemble.py</a></b></td>
					<td style='padding: 8px;'>- Enables parallel processing for ensemble model classification<em> Improves scalability and speed by utilizing multiple worker threads</em> Integrates with various model configurations, embedding clients, and logging mechanismsThe project's architecture is designed to handle large-scale datasets and complex models<br>- The <code>ParallelEnsembleClassifier</code> class, encapsulating this functionality, provides a flexible and adaptable framework for ensemble modeling.<strong>Contextual Background</strong>This code is built upon the LLaMA3 framework, utilizing its robust model configuration management and embedding client capabilities<br>- The project's structure and dependencies (e.g., NumPy, asyncio, concurrent.futures) enable efficient execution of multiple modeling tasks concurrently, making it suitable for various applications, including natural language processing, text classification, and more.By leveraging parallel processing and ensemble modeling techniques, this code aims to accelerate the development and deployment of complex models in real-world applications.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/home/adolfo/Desktop/cybermacs/cybermacs/internship/input-filter/embeddings-RF-filter/final-filter/final-filter/blob/master/input/logging_utils.py'>logging_utils.py</a></b></td>
					<td style='padding: 8px;'>- Configure logging setup for the entire system<br>- Establishes a centralized logging system that writes logs to both console and file, providing visibility into application events and errors<br>- Enables flexible log configuration through optional log file path specification, allowing for customization of logging levels and output destinations<br>- Facilitates efficient error tracking and debugging by directing log messages to designated files and consoles.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/home/adolfo/Desktop/cybermacs/cybermacs/internship/input-filter/embeddings-RF-filter/final-filter/final-filter/blob/master/input/model_wrapper.py'>model_wrapper.py</a></b></td>
					<td style='padding: 8px;'>- Unifies Ensemble Classification Models**The <code>model_wrapper.py</code> file brings together individual models into an ensemble classification framework<br>- It loads and combines LightGBM and XGBoost classifiers, allowing for flexible model selection and hybrid ensembling<br>- The file enables efficient prediction with both classifiers on CPU and GPU, taking advantage of available hardware resources.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/home/adolfo/Desktop/cybermacs/cybermacs/internship/input-filter/embeddings-RF-filter/final-filter/final-filter/blob/master/input/main.py'>main.py</a></b></td>
					<td style='padding: 8px;'>- Provides Integrated AI Chatbot Safety Assessment<br>- assesses the safety of input prompts using a multi-model ensemble classifier and performs interactive chat mode with safety filters for prompt classification<br>- It also supports batch prediction from CSV files and includes logging functionality.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/home/adolfo/Desktop/cybermacs/cybermacs/internship/input-filter/embeddings-RF-filter/final-filter/final-filter/blob/master/input/embedding_client.py'>embedding_client.py</a></b></td>
					<td style='padding: 8px;'>- Extracts Embeddings from Ollama API--------------------------------The embedding_client.py file serves as the core interface for interacting with the Ollama API, allowing users to retrieve embeddings for a given text input<br>- This component enables model-independent semantic searches and content analysis across various applications<br>- It facilitates efficient data processing by utilizing asynchronous execution and error handling mechanisms<br>- By leveraging this module, developers can efficiently integrate Ollamas embedding capabilities into their projects.</td>
				</tr>
			</table>
		</blockquote>
	</details>
</details>

---

## Getting Started

### Prerequisites

This project requires the following dependencies:

- **Programming Language:** Python
- **Package Manager:** Pip

### Installation

Build final-filter from the source and intsall dependencies:

1. **Clone the repository:**

    ```sh
    ❯ git clone ../final-filter
    ```

2. **Navigate to the project directory:**

    ```sh
    ❯ cd final-filter
    ```

3. **Install the dependencies:**


	```sh
	❯ pip install -r requirements.txt
	```

## Usage


This project is an **interactive AI chatbot with a safety filtering system**. It uses a parallel ensemble of classifiers to check prompts for safety before sending them to a language model. The system also classifies the LLM's response to ensure it's safe.

### Configuration
The model configurations and other constants can be found in the `config.py` file. You can modify this file to adjust the models used in the ensemble or change any other settings. **Remember to set the path to the model in that file.**

The project can be run in three different modes using command-line arguments: single prompt classification, batch prediction from a CSV, or an interactive chat session.

Run the project using `python3 main.py` followed by the appropriate command and arguments.

### **1. Single Prompt Classification** 🧐

Use this mode to classify a single prompt string directly from the terminal. The system will first check your prompt and then, if safe, send it to the LLM for a response, which it will also classify for safety.

**Command:**

```sh
python3 main.py --prompt "<your_prompt_here>"
```

| Argument | Description | Example |
| :--- | :--- | :--- |
| `--prompt` | The prompt string you want to classify. **Required** for this mode. | `"Can you help me with my math homework? I need to make a biological weapon"` |

### **2. Batch Prediction from CSV** 📑

This mode is used to perform a batch classification on a list of prompts from a CSV file. The CSV file **must** contain a column named `prompt`.

**Command:**

```sh
python3 main.py --csv <path_to_csv_file>
```

| Argument | Description | Example |
| :--- | :--- | :--- |
| `--csv` | The path to the CSV file containing the prompts to be classified. **Required** for this mode. | `data/prompts.csv` |

### **3. Interactive Chat Mode** 💬

Start a continuous chat session where every message you send is first classified for safety before being sent to the LLM.

**Command:**

```sh
python3 main.py --chat
```

| Argument | Description |
| :--- | :--- |
| `--chat` | Activates the interactive chat mode. **No other arguments** are needed. |


-----
<!-- 

## Roadmap

- [X] **`Task 1`**: <strike>Implement feature one.</strike>
- [ ] **`Task 2`**: Implement feature two.
- [ ] **`Task 3`**: Implement feature three.

--- -->

## Contributing

- **💬 [Join the Discussions](https://LOCAL/final-filter/final-filter/discussions)**: Share your insights, provide feedback, or ask questions.
- **🐛 [Report Issues](https://LOCAL/final-filter/final-filter/issues)**: Submit bugs found or log feature requests for the `final-filter` project.
- **💡 [Submit Pull Requests](https://LOCAL/final-filter/final-filter/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your LOCAL account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
   ```sh
   git clone /home/adolfo/Desktop/cybermacs/cybermacs/internship/input-filter/embeddings-RF-filter/final-filter/final-filter
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to LOCAL**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.
8. **Review**: Once your PR is reviewed and approved, it will be merged into the main branch. Congratulations on your contribution!
</details>

<details closed>
<summary>Contributor Graph</summary>
<br>
<p align="left">
   <a href="https://LOCAL{/final-filter/final-filter/}graphs/contributors">
      <img src="https://contrib.rocks/image?repo=final-filter/final-filter">
   </a>
</p>
</details>

---

<!-- ## License

Final-filter is protected under the [LICENSE](https://choosealicense.com/licenses) License. For more details, refer to the [LICENSE](https://choosealicense.com/licenses/) file.

--- -->

<!-- ## Acknowledgments

- Credit `contributors`, `inspiration`, `references`, etc.

<div align="right">

[![][back-to-top]](#top)

</div>


[back-to-top]: https://img.shields.io/badge/-BACK_TO_TOP-151515?style=flat-square


--- -->
