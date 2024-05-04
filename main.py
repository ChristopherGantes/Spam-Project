# main.py
from data import preprocessing
from models import NaiveBayesModel, SVMModel, DecisionTreeModel, RandomForestModel, LogisticRegressionModel
from evaluation import evaluate_models, perform_cross_validation, perform_statistical_test

# Load and preprocess data
X_train, y_train, X_test, y_test = preprocessing.preprocess_text('data/spam.csv')

# Initialize models
models = {
    'Naive Bayes': NaiveBayesModel(),
    'SVM': SVMModel(),
    'Decision Tree': DecisionTreeModel(),
    'Random Forest': RandomForestModel(),
    'Logistic Regression': LogisticRegressionModel()
}

# Evaluate models
results = evaluate_models(models, X_train, y_train, X_test, y_test)

# Print results
for model, accuracy in results.items():
    print(f'{model}: Accuracy = {accuracy}')

# Perform cross-validation and statistical tests
for model_name, model in models.items():
    cv_score = perform_cross_validation(model, X_train, y_train)
    print(f'{model_name}: Cross-validation score = {cv_score}')

# Perform statistical test between models
baseline_model = models['Naive Bayes']
for model_name, model in models.items():
    if model_name != 'Naive Bayes':
        t_statistic, p_value = perform_statistical_test(results['Naive Bayes'], results[model_name])
        print(f'Statistical test between {model_name} and Naive Bayes: t-statistic = {t_statistic}, p-value = {p_value}')
