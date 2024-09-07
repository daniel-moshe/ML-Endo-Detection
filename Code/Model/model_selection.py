import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np

from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC

class ModelSelector:
    def __init__(self):
        self.models = {
                          'CatBoost': CatBoostClassifier(logging_level='Silent'),
                          'XGBoost': xgb.XGBClassifier(eval_metric='logloss'),
                          'Logistic Regression': LogisticRegression(max_iter=10000),
                          'SVM': SVC(),
                          'Neural Network': MLPClassifier(),
                          'Random Forest': RandomForestClassifier()
                      }
        self.results = {}
        self.best_model = None
        self.best_score = 0
        self.best_model_name = None

    def train_models(self, X_train, y_train, X_test, y_test):
        for name, model in self.models.items():
            print(f'Training {name} model')
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            self.save_results(name, y_test, y_pred)
            self.update_best_model(name, model)
        print(f'Finished training, best model is {self.best_model_name}')

    def save_results(self, model_name, y_test, y_pred):
        self.results[model_name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
        }
        
    def update_best_model(self, name, model):
        curr_accuracy = self.results[name]['accuracy']
        if curr_accuracy > self.best_score:
            self.best_score = curr_accuracy
            self.best_model = model
            self.best_model_name = name
            
    def print_results(self):
        for name, metrics in self.results.items():
            print(f'model: {name}')
            print('accuracy: ', metrics['accuracy'])
            print('f1 score: ', metrics['f1_score'])
            print()

    def plot_results(self):
        model_names = list(self.results.keys())
        accuracy_scores = [metrics['accuracy'] for metrics in self.results.values()]
        f1_scores = [metrics['f1_score'] for metrics in self.results.values()]

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(model_names))
        width = 0.35

        acc = ax.bar(x - width/2, accuracy_scores, width, label='Accuracy', color='skyblue')
        f1 = ax.bar(x + width/2, f1_scores, width, label='F1 score', color='#ffb3b7')

        ax.set_xlabel('Model')
        ax.set_ylabel('Scores')
        ax.set_title('Model Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.legend()

        def add_labels(bars):
            for item in bars:
                height = item.get_height()
                ax.annotate(f'{height:.3f}', 
                            xy=(item.get_x() + item.get_width()/2, height), 
                            xytext=(0, 3),
                            textcoords='offset points', ha='center', va='bottom',
                            )
        add_labels(acc)
        add_labels(f1)

        plt.show()