import pickle
import json

from sklearn.model_selection import train_test_split,  GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score
from catboost import CatBoostClassifier
from datetime import datetime

from create_cohort import Cohort
from utils import *

params = {
    'depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'iterations': [100, 200, 300],
    'l2_leaf_reg': [1, 3, 5, 7, 9]
}

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

def create_x_y_from_cohort():
    logging.info("Creating the cohort")
    cohort = Cohort()
    cohort.create_cohort()

    cohort.drop_cols([feature_to_code("Had menopause"), feature_to_code("Ever had hysterectomy"), feature_to_code("Age at hysterectomy"), feature_to_code("Age at menopause")])
    cohort.drop_cols([feature_to_code("Year of birth")]) 

    logging.info("Splitting to X and y")
    cohort.split_x_y()
    X_train, X_test, y_train, y_test = train_test_split(cohort.X, cohort.y, test_size=0.3, shuffle=True, random_state=42)

    return X_train, X_test, y_train, y_test

def create_catboost_model(params):
    logging.info("Creating the model")
    clf = CatBoostClassifier(verbose=False)
    scorer = make_scorer(accuracy_score)
    clf_grid = GridSearchCV(estimator=clf, param_grid=params, scoring=scorer, cv=5, verbose=1)

    return clf_grid

def get_results(clf_grid):
    logging.info("Returning the results")
    return {
        'best_params': clf_grid.best_params_,
        'best_score': clf_grid.best_score_,
    }

def predict_model(model, X_test, y_test):
    logging.info("Getting predictions")
    y_pred = model.predict(X_test)

    logging.info('\nCatBoost model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

def main():
    X_train, X_test, y_train, y_test = create_x_y_from_cohort()
    clf_grid = create_catboost_model(params)

    clf_grid.fit(X_train, y_train)

    results = get_results(clf_grid)

    predict_model(clf_grid, X_test, y_test)

    # Save a JSON with the key results
    with open(f'grid_search_summary_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=4, default=str)

    logging.info(f"Grid search completed. Results saved with timestamp {timestamp}")

    pickle.dump(clf_grid, open('model.pkl','wb'))

if __name__ == "__main__":
    main()