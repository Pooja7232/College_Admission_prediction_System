import pandas as pd

class DataTransformation:

    def initiate_data_transformation(self,train_path,test_path):

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        # remove extra spaces
        train_df.columns = train_df.columns.str.strip()
        test_df.columns = test_df.columns.str.strip()

        # drop Serial No column
        if "Serial No." in train_df.columns:
            train_df = train_df.drop(columns=["Serial No."])
            test_df = test_df.drop(columns=["Serial No."])

        target = "Chance of Admit"

        X_train = train_df.drop(columns=[target])
        y_train = train_df[target]

        X_test = test_df.drop(columns=[target])
        y_test = test_df[target]

        return X_train, X_test, y_train, y_test