from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from src.utils import save_object, evaluate_model


class ModelTrainer:

    def initiate_model_trainer(self, X_train, X_test, y_train, y_test):

        models = {

            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest": RandomForestRegressor()

        }

        model_report = evaluate_model(X_train, y_train, X_test, y_test, models)

        best_model_score = max(sorted(model_report.values()))

        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]

        best_model = models[best_model_name]

        # SAVE MODEL
        save_object(
            file_path="artifacts/model.pkl",
            obj=best_model
        )