# Predicting Adverse Effects of Drug Interactions

We achieved 98% accuracy in predicting unsafe drug combinations by utilizing a stack ensemble model with Logistic Regression, Random Forest, and CatBoost on the 2014-2024 U.S. Food and Drug Administration (FDA) Adverse Event Reporting System (FAERS) dataset, all while demonstrating the model in a Gradio web application for more convenient usage. This project is developed by Class 13B of the 2025 AI4ALL Ignite Program.

## Problem Statement <!--- do not change this line -->

Adverse drug interactions (ADIs) are a leading cause of preventable harm in healthcare. According to the FDA, in 2022, there were over 1.25 million serious adverse events reported and nearly 175,000 deaths. Manually identifying unsafe drug combinations is impractical due to the vast number of possible interactions and hidden risk patterns. Hence, to address this problem, our team leverages machine learning to accurately predict unsafe drug combinations and thus improve prescription safety.

## Key Results <!--- do not change this line -->

1. Achieved 98% ROC AUC accuracy on predicting over 222,000 unique drug interactions.
2. Deployed the model onto an interactive web application for more convenient drug interaction prediction, with features
   - searching for and selecting drugs easily from a dropdown list
   - providing binary prediction with a confidence score from 0-100

## Methodologies <!--- do not change this line -->

To accomplish this, we cleaned the dataset, trained a stack ensemble model with Logistic Regression, Random Forest, and CatBoost, and applied 5-fold cross-validation to avoid overfitting. We then developed a web application using Gradio by passing in the dataset and making live predictions using the pre-trained model.

## Data Sources <!--- do not change this line -->

FDA FAERS dataset spanning 2014Q3-2024Q3: [High-Order Drug-Drug Interaction Dataset (HODDI)](https://github.com/TIML-Group/HODDI)

## Technologies Used <!--- do not change this line -->

- Python
- Git
- pandas
- gradio

## Usage Instruction <!--- do not change this line -->
1. Download or load the dataset from [HODDI](https://github.com/TIML-Group/HODDI).
2. Install the CatBoost Python package in your terminal or command prompt.
```
pip install catboost
```
3. Modify model.py with your specific data path and save results to your designated directory.
```
python /path_to_model/model.py
```
4. Install the gradio package in your terminal or command prompt.
```
pip install gradio
```
5. Run the web_app.py with the saved pre-trained model.
```
python /path_to_web_app/web_app.py
```

## Contribution Guidelines <!--- do not change this line -->

We welcome contributions to improve both the model and the web application!

To contribute:
- Fork this repository
- Make your changes
- Submit a pull request

Your contributions will be reviewed and may be merged into the main branch. Thank you for helping us make this project better!

## Authors <!--- do not change this line -->

This project was completed in collaboration with:
- Jiyuan Ji ([cji28@amherst.edu](mailto:cji28@amherst.edu))
- Emily Hsu ([eh119@wellesley.edu](mailto:eh119@wellesley.edu))
- Anusri Nagarajan ([anusri.nagarajan@sjsu.edu](mailto:anusri.nagarajan@sjsu.edu))
- Ahmed Mohammed ([ahmed.mohammed@bison.howard.edu](mailto:ahmed.mohammed@bison.howard.edu))

## Acknowledgements <!--- do not change this line -->

Thank you to:
- Marilu Duque, the AI4ALL Class 13 Instructor, for teaching us the necessary skills and ethical considerations in creating this project,
- Ankush Rastogi, our project mentor, for providing useful suggestions on developing and deploying our model,
- Sai Donepudi, the AI4ALL Class 13 Teaching Fellow, for offering helpful tips on developing our project, and
- AI4ALL Ignite program, for providing the opportunity for us to learn and apply AI to an impactful real-world problem, especially to be aware of AI ethics.
