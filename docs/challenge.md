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


