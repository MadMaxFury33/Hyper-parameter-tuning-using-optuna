# 🎯 Hyperparameter Tuning with Optuna — Random Forest Example

This repository consists of the **fundamental concepts of using Optuna for hyperparameter tuning**, demonstrated with a **Random Forest model**.  
It showcases how Optuna can efficiently optimize model hyperparameters to achieve improved predictive performance with minimal manual effort.

---

## 📘 Overview

Hyperparameter tuning plays a vital role in maximizing a model’s accuracy and generalization.  
This notebook provides a **step-by-step guide** to applying **Optuna** for tuning the hyperparameters of a **Random Forest** model, including how to define an objective function, run multiple trials, and analyze optimization results.

---

## 🧩 Key Concepts Covered

- Basics of **Optuna** for automated hyperparameter search  
- Setting up the **objective function** for Random Forest  
- Creating and executing an **Optuna study**  
- Evaluating and visualizing **optimization history**  
- Extracting and applying **best hyperparameters** to the model  

---

## 📊 Example Workflow

1. **Data Preparation**
   - Load and explore dataset (`healthexp` from Seaborn)
   - Handle missing values and perform feature selection  

2. **Model Setup**
   - Use `RandomForestRegressor` or `RandomForestClassifier` from Scikit-Learn  
   - Define the hyperparameter search space (e.g., `n_estimators`, `max_depth`, `min_samples_split`)  

3. **Optimization with Optuna**
   ```python
   import optuna

   def objective(trial):
       n_estimators = trial.suggest_int("n_estimators", 100, 1000)
       max_depth = trial.suggest_int("max_depth", 3, 20)
       min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
       # Define and evaluate RandomForest model here

   study = optuna.create_study(direction="maximize")
   study.optimize(objective, n_trials=50)
   ```

4. **Visualization**
   - Optimization history  
   - Parameter importance plots  
   - Best parameters extraction  

---

## 🧠 Why Optuna?

- **Automates hyperparameter search** intelligently  
- **Prunes unpromising trials** early to save time  
- **Integrates seamlessly** with Scikit-Learn and major ML libraries  
- Offers **interactive visualization** for insights and transparency  

---

## ⚙️ Installation

```bash
pip install optuna scikit-learn seaborn matplotlib pandas
```

---

## 🚀 Usage

Run the notebook to explore and tune Random Forest hyperparameters:

```bash
jupyter notebook hyperparameter_tuning_optuna.ipynb
```

You can modify the objective function to experiment with other models or datasets.

---

## 📈 Example Visualizations

Optuna’s built-in visual tools help analyze the search space and optimization process:

```python
optuna.visualization.plot_optimization_history(study)
optuna.visualization.plot_param_importances(study)
```

---

## 📂 Repository Structure

```
├── hyperparameter_tuning_optuna.ipynb   # Main Jupyter Notebook
├── README.md                            # Documentation
└── requirements.txt (optional)          # Dependencies list
```

---

## 🔍 Example Hyperparameters Tuned

| Parameter            | Description                              |
|----------------------|------------------------------------------|
| `n_estimators`       | Number of trees in the forest            |
| `max_depth`          | Maximum tree depth                       |
| `min_samples_split`  | Minimum samples required to split a node |
| `min_samples_leaf`   | Minimum samples per leaf node            |

---

## 🧩 Future Enhancements

- Compare **Optuna vs GridSearchCV** performance  
- Add **cross-validation** inside objective function  
- Extend examples to **XGBoost**, **LightGBM**, and **CatBoost**  

---

## 🧑‍💻 Author

**Prajwal Mahanawar**  
MSc in Data Science & Statistical Learning  
📍 University of Limerick | 🔗 [GitHub](https://github.com/MadMaxFury33)
