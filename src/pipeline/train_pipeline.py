from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainPipeline:

    def run_pipeline(self):

        # Data Ingestion
        data_ingestion = DataIngestion()
        train_data, test_data = data_ingestion.initiate_data_ingestion()

        # Data Transformation
        data_transformation = DataTransformation()
        X_train, X_test, y_train, y_test = data_transformation.initiate_data_transformation(
            train_data, test_data
        )

        # Model Training
        modeltrainer = ModelTrainer()
        modeltrainer.initiate_model_trainer(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    obj = TrainPipeline()
    obj.run_pipeline()