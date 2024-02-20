import numpy as np
import os
from datetime import datetime
from typing import Tuple, Union, List

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import xgboost as xgb
import pandas as pd
import pickle
import json

CATEGORICAL_FEATURES = ["OPERA", "TIPOVUELO", "MES"]

TRAIN_COLUMNS = ["OPERA", "MES", "TIPOVUELO", "SIGLADES", "DIANOM"]

TRAIN_FEATURES = [
    "OPERA_Latin American Wings",
    "MES_7",
    "MES_10",
    "OPERA_Grupo LATAM",
    "MES_12",
    "TIPOVUELO_I",
    "MES_4",
    "MES_11",
    "OPERA_Sky Airline",
    "OPERA_Copa Air",
]

THRESHOLD_IN_MINUTES: float = 15


class DelayModel:

    def __init__(self):
        self._model = None  # Model should be saved in this attribute.
        self._model_default_params = None
        # rate to split data and save test set.
        self._test_set_rate = None
        self._model_path = None
        self._model_version = None

        # Load default config
        with open("challenge/default.json", "r") as f:
            config = json.load(f)
            self._model_version = (
                config["model_version"] if "model_version" in config else "v1"
            )
            self._model_path = (
                config["model_path"] if "model_path" in config else "challenge/models"
            )
            self._test_set_rate = (
                config["test_set_rate"] if "test_set_rate" in config else 0.33
            )
            self._model_default_params = (
                config["model_default_params"]
                if "model_default_params" in config
                else {"random_state": 1, "learning_rate": 0.01}
            )

    def __get_min_diff(self, data: pd.DataFrame):
        fecha_o = datetime.strptime(data["Fecha-O"], "%Y-%m-%d %H:%M:%S")
        fecha_i = datetime.strptime(data["Fecha-I"], "%Y-%m-%d %H:%M:%S")
        min_diff = ((fecha_o - fecha_i).total_seconds()) / 60
        return min_diff

    def __scale_labels_weights(self, labels: pd.DataFrame):

        target_column = labels.columns[0]
        n_y0 = len(labels[labels[target_column] == 0])
        n_y1 = len(labels[labels[target_column] == 1])
        return n_y0 / n_y1

    def __preprocess_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process categorical features by generating one hot vectors for each one.
        """
        features = pd.concat(
            [
                pd.get_dummies(data[feature], prefix=feature)
                for feature in CATEGORICAL_FEATURES
            ],
            axis=1,
        )
        return features

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None,
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        train_columns = TRAIN_COLUMNS.copy()

        data = shuffle(data, random_state=111)

        features = self.__preprocess_features(data)
        features = features[TRAIN_FEATURES]

        if target_column:
            train_columns += [target_column]

            data["min_diff"] = data.apply(self.__get_min_diff, axis=1)
            data[target_column] = np.where(
                data["min_diff"] > THRESHOLD_IN_MINUTES, 1, 0
            )

            target = data[[target_column]]
            return (features, target)

        return features

    def __split_data(self, features, target):

        x_train, x_test, y_train, y_test = train_test_split(
            features, target, test_size=self._test_set_rate, random_state=42
        )

        # x_train.to_csv(f"{self._model_path}/{self._model_version}/x_train.csv", index=False)
        # x_test.to_csv(f"{self._model_path}/{self._model_version}/x_test.csv", index=False)
        # y_train.to_csv(f"{self._model_path}/{self._model_version}/y_train.csv", index=False)
        # y_test.to_csv(f"{self._model_path}/{self._model_version}/y_test.csv", index=False)
        return x_train, x_test, y_train, y_test

    def __save(self, configs):
        """
        Save model with the path and version got from default.json file
        """

        if not os.path.exists(f"{self._model_path}/{self._model_version}"):
            os.makedirs(f"{self._model_path}/{self._model_version}", exist_ok=True)

        # Save model and configs
        with open(f"{self._model_path}/{self._model_version}/model.pkl", "wb") as f:
            pickle.dump(self._model, file=f)
        with open(f"{self._model_path}/{self._model_version}/config.pkl", "wb") as f:
            pickle.dump(configs, file=f)

    def __load_model(self):
        """
        Load a model and configs
        """
        with open(f"{self._model_path}/{self._model_version}/model.pkl", "rb") as f:
            self._model = pickle.load(f)

        with open(f"{self._model_path}/{self._model_version}/config.pkl", "rb") as f:
            self._model_config = pickle.load(f)

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """

        # Note: Since the unittest is passing all the data to the fit model, Im spliting data hgere
        x_train, _, y_train, _ = self.__split_data(features, target)

        scale = self.__scale_labels_weights(y_train)

        model_params = {**self._model_default_params, **dict(scale_pos_weight=scale)}
        self._model = xgb.XGBClassifier(**model_params)

        self._model.fit(x_train, y_train)

        # Saving the model and configs
        self.__save(model_params)

    def get_model(self):
        if not self._model:

            # If there is no model we need to try to load first
            self.__load_model()

        return self._model

    def predict(self, features: pd.DataFrame) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.

        Returns:
            (List[int]): predicted targets.
        """
        model = self.get_model()

        predictions = model.predict(features)
        return predictions.tolist()
