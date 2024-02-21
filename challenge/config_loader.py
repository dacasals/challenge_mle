import yaml


class ConfigLoader:
    """
    Base class for load an manage confuration from a yaml file.
    """

    def __init__(self, path) -> None:
        self.path = path
        # self.all_configs = self.load_config_from_yaml()

    def load_config_from_yaml(self):
        with open(self.path, "r") as f:
            return yaml.safe_load(f)


class ModelConfigLoader(ConfigLoader):
    """
    Load and define model configurations to be used in model and api
    """

    def __init__(self, path="challenge/default.yml") -> None:
        super().__init__(path)
        all_config = self.load_config_from_yaml()
        self.raw_model_config = all_config["ModelConfig"]
        self.model_version = None
        self.model_path = None
        self.test_set_rate = None

        self.set_allowed_features()
        self.set_default_model_config()
        self.set_default_model_params()
        self.set_categorical_features()
        self.set_data_columns()

    def set_allowed_features(self):
        """
        It set the features names allowed and used by the model during training and predictions.
        """
        train_features = self.raw_model_config["training_features"]
        all_features_name = []

        if "categorical" in train_features:
            for feature_name, values in train_features["categorical"].items():
                for val in values:
                    all_features_name.append(f"{feature_name}_{val}")

        if "default" in train_features and type(train_features["default"]) == list:
            all_features_name.extend(train_features["default"])

        # Always sort alphabetically the features names
        self.train_features_name = sorted(all_features_name)

    def set_default_model_config(self):
        """
        It set the configurations for the model save and load as well as other training details.
        """
        # model version used in the path to save each model, in separated folder by version.
        if "model_version" in self.raw_model_config:
            self.model_version = self.raw_model_config["model_version"]

        # Path where models will be saved
        if "model_path" in self.raw_model_config:
            self.model_path = self.raw_model_config["model_path"]

        # The rate to split the train, test data
        if "test_set_rate" in self.raw_model_config:
            self.test_set_rate = self.raw_model_config["test_set_rate"]

        # The threshold defined in the preprocessing target, to define if flight is delayed or not
        if "threshold_in_minutes" in self.raw_model_config:
            self.threshold_in_minutes = self.raw_model_config["threshold_in_minutes"]

    def set_default_model_params(self):
        """
        It set the default params used during the model instance creation.
        """
        self.default_model_params = {}
        if "default_model_params" in self.raw_model_config:
            self.default_model_params = self.raw_model_config["default_model_params"]

    def set_categorical_features(self):
        """
        Define categorical columns to be converted to one-hot vectors during preprocessing
        """
        train_features = self.raw_model_config["training_features"]
        features_name = []

        if "categorical" in train_features:
            for feature_name, _ in train_features["categorical"].items():
                features_name.append(feature_name)

        self.categorical_features = sorted(features_name)

    def set_data_columns(self):
        """
        Define data columns to be extracted from csv during preprocessing
        """
        train_features = self.raw_model_config["training_features"]

        columns = []
        if "categorical" in train_features:
            for feature_name, _ in train_features["categorical"].items():
                columns.append(feature_name)

        if "default" in train_features and train_features["default"]:
            columns.extend(train_features["default"])
        self.data_columns = sorted(columns)


class APIConfigLoader(ConfigLoader):
    """
    Define and load configurations for the API
    """

    def __init__(self, path="challenge/default.yml") -> None:
        super().__init__(path)
        all_config = self.load_config_from_yaml()
        self.raw_config = all_config["ApiConfig"]
        self.set_models_attributes()

    def set_models_attributes(self):
        self.models = {}

        if "models" in self.raw_config and self.raw_config["models"]:
            self.models = self.raw_config["models"]
