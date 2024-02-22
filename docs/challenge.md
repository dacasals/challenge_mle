### Author
-  Name: Daniel Arturo Casals Amat
- Email: @dcasalsamat@gmail.com

### About notebook and models evaluation
I fond some bugs in the notebook shared with the DS analysis and train:

#### barplot calls
Seems like the way used to call the barplot function has changed. Now it is mandatory define the x, y parameters to use **sns.barplot**:
```python
#Org
sns.barplot(flight_type_rate_values, flight_type_rate['Tasa (%)'])
#Fixed to
sns.barplot(x=flight_type_rate_values, y=flight_type_rate['Tasa (%)'])
```

#### About the get_rate_from_column function

The **get_rate_from_column** function defined at the begining of section 3 has an issue that affect the posterior analysis.

It is defining **delay rate** as total[city] / count_rows_with_delay[city]. So it is not getting the rate correctly.

Example: is Houston city which has 17 delays over 317, so delay rate should be ~ 0.05 (5%), instead of the current (317/17).

Going to fix it to: [city] / total[city].

```python
# Original definition
def get_rate_from_column(data, column):
    delays = {}
    for _, row in data.iterrows():
        if row['delay'] == 1:
            if row[column] not in delays:
                delays[row[column]] = 1
            else:
                delays[row[column]] += 1
    total = data[column].value_counts().to_dict()
    
    rates = {}
    for name, total in total.items():
        if name in delays:
            rates[name] = round(total / delays[name], 2)
        else:
            rates[name] = 0
            
    return pd.DataFrame.from_dict(data = rates, orient = 'index', columns = ['Tasa (%)'])
```
Changed to
```python
def get_rate_from_column(data, column):
    delays = {}
    for _, row in data.iterrows():
        if row['delay'] == 1:
            if row[column] not in delays:
                delays[row[column]] = 1
            else:
                delays[row[column]] += 1
    total = data[column].value_counts().to_dict()
    
    rates = {}
    for name, total in total.items():
        if name in delays:
            rates[name] = round((delays[name] / float(total)*100) , 2)
        else:
            rates[name] = 0
            
    return pd.DataFrame.from_dict(data = rates, orient = 'index', columns = ['Tasa (%)'])
# Here we are defining to the Tasa (%) by multiplying the rate to 100, even though I think it is confusing talk to rates (usually [0:1]) expressed in percentages ([0:100]).
```
Given these changes I made other changes in the plotting calls like removing ylim in some cases since the ranges changed.

### About features

I notice that in the notebook, the DS finally did not used two of the features recommended: **high_season** and **period_day**. Given that the challence specifically says we should not improve the model I **did not** explore the impact of these features in the results.


### About Models selection

It seems the two models with top 10 featurs and balancing are performing almost exactly equal for metrics precision, recall, f1-score.
Given this, it is hard to pick one with no other information.

So, I check if this behaivor corresponds is for the exact data split that was done or it change between data splits.

I applied a ```cross validation``` tecnic in which I pick 3 different splits changing train and test data so I can check if the model results are consitent. with the results the DS got. The following are the results after aggregate with a mean the results for the 3 folds by model.

Details:
- Folds used: 3
- Data was split using same rate was used by the  DS (33% test)
- Results were aggregated using average.
- Features are the same 10 proposed by the DS.

| model              |   CLASS_ON_TIME |   CLASS DELAY |   accuracy | metric    |
|:-------------------|----------------:|--------------:|-----------:|:----------|
| LogisticRegression |        0.839341 |      0.294322 |   0.65755  | precision |
| XGBClassifier      |        0.838027 |      0.289478 |   0.641657 | precision |
| LogisticRegression |        0.722854 |      0.369764 |   0.65755  | recall    |
| XGBClassifier      |        0.700296 |      0.38324  |   0.641657 | recall    |
| LogisticRegression |        0.762182 |      0.230143 |   0.65755  | f1-score  |
| XGBClassifier      |        0.745741 |      0.227621 |   0.641657 | f1-score  |

#### Looking at Presition and Recall Metrics:

- Presition: TP/ (TP+ FP): % of correctly labeled positive (flights with delay) out of all rows **labeled** as positive (it can include flights On time that were labeled as delayed by the model).
- Recall: TP/ (TP+ FN): % of correclty labeled positive (flights with delay) out of all positives.

In this case, after checking some research [[1]](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-020-00380-z), it seems better way to determine best model is to look at Recall for the class 1 (delay).

This is because we want pick the models with the best performance on tagging flights as delayed only in the cases of actually it is true.
So we prefer Recall over Precision in this case, since we want to  avoid get too much False Negatives (flights delayed that was predicted as ON TIME for the model)
Given that Best model in the cross_validation for Recall in class 1 is the ```XGBClassifier``` with a mean of **0.383240** in the 3 folds tested.

Model selected: ```XGBClassifier```


### Part I. Model class implementation details

#### Configuration managment

Given the nature of the challenge I realized that a centralized way to manage configurations its a nice to have feature.
I added a  file the [[challenge/default.yml]](../challenge/default.yml) to declare configurations needed for both model and api.

Configurations related with model are declared under ```ModelConfig``` key in the yaml. Example of config and comments:

```yaml
ModelConfig:

  model_version: "v1" # Model version to used to save and load for diff train versions.
  model_path: "challenge/models"  # Default path to get the models
  test_set_rate: 0.33 # Test rate used in the data split
  threshold_in_minutes: 15.0 # other params required
  ...
  default_model_params: #  Here I can add default params to instantiate the model.
    random_state: 1
    ...
  training_features: 
    
    categorical: # Features to encode as one-hot vectors, the inside values are the ones allowed for training and prediction, used also in the api. for preprocesing the request inputs so we ca have all in same place keep the consistency of model changes.
      OPERA:
        - Latin American Wings
        ...
    default: # Other features that no need a preprocessing step.
        - feat_a
        - feat_b
        ...
...
```

I pick ```yaml``` instead of ```json``` u other format because its better to read it. To load and use this configurations in python code, some ConfigLoader classes were defined inside [[challenge/config_loader.py]](../challenge/config_loader.py)
These classes are used to load and expose as objects the configs.

Check ```.ModelConfigLoader``` for the model config and ```APIConfigLoader``` for the api config.


#### Other Model details

- Inside the model class I only added the notebook code actually used for training and predicting, other unused functions like ```is_high_season``` and ```get_period_day``` are not included since the DS did not used.
- There was a typo for the output type defined for ```preprocess``` function because the use or parenthesis. It was changed from ```-> Union(...)``` to ``` -> Union[...]```
- I added some auxiliary functions, check the code documentation for the details:
  - preprocess_features
  - preprocess
  - __scale_labels_weights
  - __split_data 
  - __save
  - __load_model
  - get_model
- As mentioned above, all configs are contained in the yaml file and is accessed with ModelConfigLoader.


### Part II. API implementation

#### Configuration managment

To manage the api configuration I follow the same approach used for the model.
I added inside the file the [[challenge/default.yml]](../challenge/default.yml), a key ```ApiConfig``` that include the api configuration. 

```yaml
###

...
      
ApiConfig:
  models: # Inside this key I defined relevant information to create the models
    FlightModel: # This is de metadata to build the model **Flight**.
      OPERA:  # It is used to define the values of an Enum called OperatorEnum
        - American Airlines
        - Air Canada
        - Air France
        ...
        
      MES:  # It is used to define the values of an Enum called MESEnum
        - 1
        - 2
        ...
        - 12
      
      TIPOVUELO: # It is used to define the values of an Enum called TIPOVUELOENUM
        - 'I'
        - 'N'
```

#### Data models definition

As described in the FastAPI doc, it is a good practice to define the inputs and 
outputs of each api route using ```pydantic models```. It has a several advantages for this usecase:
- Delegate to pydantic the input/output data validations.
- Also help with the return of an appropiate status code in case the validation fails.
- It helps to generate automatically an OpenAPI specification for the api that is very usefull for many uses cases including keep updated the documentation of the API.

I instantiate ```dynamically``` the Enums with the yaml config data and set `OPERA`, `MES` and `TIPOVUELO` as attributes of model `Flight`. So, in case its needed, for example add a new ```airline``` we just need edit the yaml file instead of rewrite the api or model code.

Models define inside the same api.py file:
- Flight
- Flights
- ResponseModel

If the api gain in complexity we could move these model to a separated layer.

#### Other api details

-  I notice model pydantic data validators return an status code `422`, instead of the code `400` required for the unit tests.
Thats the reason for the inclusion of a middleware (`exception_handler`) inside the same api.py file, so I can change the code from 422 -> 400
and pass the tests.

- Also added a function (`preprocess_model_request`) for the preprocessing step of the input model data, that reuse same function of the `DelayModel` class.
- I made some small fixes, related with project, dependecies:
    - add package anyio~=3.4 because there was a bug with the version 4 that affect the api-test.
    - add change package version of of locust to `locust~=2.23.0` because stress-test fails because an issue with the `jinja2` package installed by a locust dependency.


### Part III

To deploy the model in Gcloud I completed the Dockerfile with a python standard image using version 3.10.9.

- Model is available at [[https://challenge-mle-dap2zffasa-uc.a.run.app]](https://challenge-mle-dap2zffasa-uc.a.run.app). 

- After build the image it was pushed to Docker Hub [[https://hub.docker.com/repository/docker/dcasalsamat/challenge-mle/general]](https://hub.docker.com/repository/docker/dcasalsamat/challenge-mle/general).

- To serve the app i used `Google Cloud Run` service, just set low resources for the instance. 
- I ran the make stress-test for a few seconds and it worked ok withoud request fails. But I haven't tune tune too much the instance resources, only 512MB & 1CPU.

#### Part IV

The CI/CD implementation was added in the following way:

- CI:
  - I've created 3 steps, mainly using same make commands configured in the Makefile:
    - Checkout code
    - Install dependencies
    - Run model tests
    - Run api-test
  - Additionally, 3 python versions were tested 3.8(my local env), 3.10, 3.11. All ran ok, but the 3.11 took a lot to run, so I desided to choose 3.10 as the Docker image for the deployment.
  - The CI run for all most of the branches.

- CD:
  - For the Continuous Delivery process I added the following: 
  
    - Install dependencies
    - Run model train: In this step I also added to the Makefile a command to run a python script implemented (added in `deployment/trainer.py`) to train the model

    - Build: Build the docker image.
    - Publish: Publish the image in [[https://hub.docker.com/repository/docker/dcasalsamat/challenge-mle/general]](https://hub.docker.com/repository/docker/dcasalsamat/challenge-mle/general)

    - Authenticate to Google Cloud
    - Deploy to Cloud Run by updating the created instance with the docker image I built in preview step.

  - The CD run only for `main` branch so the app only us updated after receive a stable version.


