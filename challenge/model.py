import numpy as np
import os
from datetime import datetime
from typing import Tuple, Union, List

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import xgboost as xgb
import pandas as pd
import pickle

from .config_loader import ModelConfigLoader


class DelayModel:

    def __init__(self):
        self._model = None  # Model should be saved in this attribute.
        self._model_default_params = None
        # rate to split data and save test set.

        config_loader = ModelConfigLoader()

        self._test_set_rate = (
            config_loader.test_set_rate if config_loader.test_set_rate else 0.33
        )
        self._model_path = (
            config_loader.model_path if config_loader.model_path else "challenge/models"
        )
        self._model_version = (
            config_loader.model_version if config_loader.model_version else "v1"
        )
        self._threshold_in_minutes = (
            config_loader.threshold_in_minutes
            if config_loader.threshold_in_minutes
            else 15.0
        )
        # Set train categorical features from default.yml
        self.categorical_features = config_loader.categorical_features

        self.data_columns = config_loader.data_columns

        self.train_features = config_loader.train_features_name
        # Load default config
        self._model_default_params = config_loader.default_model_params

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

    def preprocess_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process categorical features by generating one hot vectors for each one.
        """
        # Encode categorical features
        features = pd.concat(
            [
                pd.get_dummies(data[feature], prefix=feature)
                for feature in self.categorical_features
            ],
            axis=1,
        )
        # Join other features
        features = pd.concat(
            [data.drop(columns=self.categorical_features), features], axis=1
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
        train_columns = self.data_columns.copy()
        data = shuffle(data, random_state=111)
        features = self.preprocess_features(data)
        features = features[self.train_features]

        if target_column:
            train_columns += [target_column]

            data["min_diff"] = data.apply(self.__get_min_diff, axis=1)
            data[target_column] = np.where(
                data["min_diff"] > self._threshold_in_minutes, 1, 0
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
