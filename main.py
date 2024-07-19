import pandas as pd
from scipy import stats

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split, cross_val_score


# Load data
df = pd.read_csv('data/spam.csv', encoding='windows-1252')
# Extract text data
text = df['text']
# Preprocess the text data using TF-IDF vectorization
vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(text)
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.7)

# Creating the models
DecisionTreeModel = DecisionTreeClassifier()
LogisticRegressionModel = LogisticRegression()
NaiveBayesModel = MultinomialNB()
RandomForestModel = RandomForestClassifier()
SVMModel = SVC()

models = {
    'Decision Tree': DecisionTreeModel,
    'Logistic Regression': LogisticRegressionModel,
    'Naive Bayes': NaiveBayesModel,
    'Random Forest': RandomForestModel,
    'SVM': SVMModel,
}

# Evaluate models
results = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    accuracy = accuracy_score(y_test, predicted)
    precision = precision_score(y_test, predicted, pos_label='spam')
    results[model_name] = accuracy
# Print results
    print(model_name)
    print("\tAccuracy = ", accuracy)
    print("\tPrecision = ", precision)

print()

# df = pd.read_csv('data/email_classification.csv', encoding='windows-1252')
# text = df['text']
# x = vectorizer.fit_transform(text)
# y = df['label']
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.7)
#
#
# results = {}
# for model_name, model in models.items():
#     model.fit(X_train, y_train)
#     predicted = model.predict(X_test)
#     accuracy = accuracy_score(y_test, predicted)
#     precision = precision_score(y_test, predicted, pos_label='spam')
#     results[model_name] = accuracy
#     print(model_name)
#     print("\tAccuracy = ", accuracy)
#     print("\tPrecision = ", precision)
#
# print()

# Perform cross-validation
cv_scores = {}
for model_name, model in models.items():
    cv_scores[model_name] = cross_val_score(model, X_train, y_train, cv=5)
    cv_scores[model_name] = cv_scores[model_name]
    print(f'{model_name} CV score = {cv_scores[model_name].mean()}')

print()

# Perform statistical test between models
for model_name, model in models.items():
    if model_name != 'Naive Bayes':
        t_stat, p_value = stats.ttest_rel(cv_scores['Naive Bayes'], cv_scores[model_name])
        print(f'Stat test between {model_name} and Naive Bayes')
        print(f'\tt-stat = {t_stat}')
        print(f'\tp-value = {p_value}')
