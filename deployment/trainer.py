import sys
import os
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.append(f'{os.getcwd()}/')

from challenge.model import DelayModel
import pandas as pd
def train():
    model = DelayModel()
    data = pd.read_csv(filepath_or_buffer="data/data.csv")
    features, target = model.preprocess(
        data=data,
        target_column="delay"
    )

    model.fit(
        features=features,
        target=target
    )
    print("version created", os.listdir(f"challenge/models/"))

if __name__ == '__main__':
    train()