###########################
# Load the data
###########################

import pandas as pd
heart = pd.read_csv("heart_failure.csv")

##########################
# Preliminary analyses
##########################

### Overview of the dataset  

print(heart.head())
print(heart.dtypes)
print(f"Number of missing values in the dataset : {heart.isna().sum().sum()}")
print(heart.describe().round(2))

### Mortality rate 

mortality_rate = sum(heart["DEATH_EVENT"])/len(heart)
print(f"The mortality rate among the patients of the study is {mortality_rate :.0%}")

### Distributions of the variables

import seaborn as sns
import matplotlib.pyplot as plt

## Binary variables (countplots)

binary_variables = ["anaemia", "diabetes", "high_blood_pressure", "smoking", "sex"]

plt.figure(figsize=(15, 8))

for i, var in enumerate(binary_variables, 1):
    plt.subplot(2, 3, i)  # 2 lines and 3 columns
    sns.countplot(data=heart, x=var, color="steelblue")
    
plt.savefig("distributions_binary_variables.png")

## Other variables (histograms)

other_variables = ["creatinine_phosphokinase", "ejection_fraction", "platelets", "serum_creatinine", "serum_sodium", "time", "age"] 

plt.figure(figsize=(15, 8))

for i, var in enumerate(other_variables, 1):
    plt.subplot(2, 4, i)  
    sns.histplot(data=heart, x=var, color="steelblue")
    plt.ylabel("")
    
plt.savefig("distributions_other_variables.png")

### Conditional distributions and how variables relate to each others

## Smoking conditional on gender
 
fig, ax = plt.subplots(figsize=[7, 5])

sns.countplot(
    data=heart,
    x="sex",
    hue="smoking",      
    ax=ax
    )

ax.set(
    xticks=[0, 1],
    xticklabels=["female", "male"],
    xlabel=""
)

plt.savefig("smoking_conditional_gender.png")

## Distribution of patients among the possible conditions

conditions = (
    heart[["anaemia", "high_blood_pressure", "diabetes"]]
        .value_counts()
        .reset_index(name="count")
)

# create a function that makes a label for each possible combination (for the graph)
def make_label(r): 
    labels = []
    if r["anaemia"] == 1:
        labels.append("Anaemia")
    if r["high_blood_pressure"] == 1:
        labels.append("High blood pressure")
    if r["diabetes"] == 1:
        labels.append("Diabetes")
        
    return " + ".join(labels) if labels else "No condition"

conditions["label"] = conditions.apply(make_label, axis=1)

# Don't include no condition in the graph
conditions = conditions[conditions["label"] != "No condition"]


fig, ax = plt.subplots(figsize=[15, 8])

sns.barplot(
    data=conditions,
    x="label",
    y="count",
    color="steelblue",
)

ax.set(
    xlabel="",
    ylabel="Number of patients"
)

plt.xticks(rotation=45, ha="right") 
plt.tight_layout()
plt.savefig("conditions.png")

## Distributions of the biological markers conditional on death event 

markers = ["creatinine_phosphokinase", "ejection_fraction", "platelets", "serum_creatinine", "serum_sodium"]

# unpivot the data for the plots (wide to long)
heart_long = heart.melt( 
    id_vars="DEATH_EVENT", 
    value_vars=markers,
    var_name="marker",
    value_name="value"
)

plt.figure(figsize=(15, 8))

for i, marker in enumerate(markers, 1): 
    
    ax = plt.subplot(2, 3, i)
    
    sns.boxplot(
        data=heart,
        x="DEATH_EVENT",
        y=marker,
        ax=ax
    )

    # log scale for creatinine markers as they can take large values
    if marker in ["creatinine_phosphokinase", "serum_creatinine"]:
        ax.set_yscale("log")
        
    ax.set(
        xlabel="",
        ylabel="",
        title=marker
    )

plt.savefig("markers_conditional_death_rate.png")
plt.close()

## Factors related to heart failure (test statistics)

from scipy.stats import ttest_ind, chi2_contingency, fisher_exact

cols = [col for col in heart.columns if col != "DEATH_EVENT"] # all variables except the target
outcome = "DEATH_EVENT"

results = []

for col in cols:
    
    # t test for non binary variables 
    if heart[col].dtype in ["int64", "float64"] and heart[col].nunique() > 2:
        
        group1 = heart.loc[heart[outcome]==1, col]
        group0 = heart.loc[heart[outcome]==0, col]
        stat, p = ttest_ind(group1, group0)
        test = "t-test"
    
    else:
        # Chi-square for binary variables
        
        # contigency table
        contingency = pd.crosstab(heart[col], heart[outcome])
        if contingency.shape == (2,2):
            # 2x2 table -> Fisher exact
            stat, p = fisher_exact(contingency)
            test = "Fisher exact"
        else:
            stat, p, _, _ = chi2_contingency(contingency)
            test = "Chi-square"
    
    # store the results
    results.append({
        "variable": col,
        "test": test,
        "statistic": stat,
        "p_value": p
    })


results = pd.DataFrame(results)
results['statistic'] = results['statistic'].round(3)
results['p_value'] = results['p_value'].round(3)
results = results.sort_values("p_value")
print(results)

#######################
# Machine learning bloc
#######################

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix


heart = pd.read_csv("heart_failure.csv")

### Basic training/validation/test approach

# use all the variables to predict the target DEATH_EVENT, except for time (see the report)
prediction_variables = [col for col in heart.columns if col != "DEATH_EVENT" and col != "time"] 

# separate the explanatory variables and the target 
heart_X = heart[prediction_variables]
heart_Y = heart["DEATH_EVENT"]

# split the data in training and test and stratify 
X_train, X_test, Y_train, Y_test = train_test_split(heart_X, heart_Y, 
                                                    test_size=0.2, 
                                                    random_state=0,
                                                    stratify=heart_Y)

# create a vsplit object to create train and validation (from previous train) 
v_split = StratifiedShuffleSplit(n_splits=1, train_size=0.75, random_state=0)

# create the classifier object
dt = DecisionTreeClassifier(random_state=42)

# pecify manually the set of possible model classes (defined with stopping parameters)
meta_param = {'max_depth': range(1, 11),
              'min_samples_split': [2, 10, 25, 50, 75, 85, 100, 200]}


# use a grid search to find the best class and the best model
dt_search = GridSearchCV(dt, meta_param, cv=v_split, n_jobs = -1)
# select the best model
dt_res = dt_search.fit(X_train, Y_train)

# visualize the grid with a heat map
df = pd.DataFrame(dt_res.cv_results_['params'])
df['mean_test_score'] = dt_res.cv_results_['mean_test_score']
sns.set_theme()
sns.set(rc={'figure.figsize': (8, 6)})
sns.heatmap(df.pivot_table(index='max_depth', columns='min_samples_split', 
                           values='mean_test_score'), 
            cmap='viridis', annot=True)
plt.xlabel('min samples split')
plt.ylabel('max depth')
plt.savefig('grid_search.png')

# Evaluate on the test
Y_test_pred = dt_res.predict(X_test)

print(f"The root mean squared error of the selected model on the test is {np.sqrt(mean_squared_error(Y_test, Y_test_pred)).round(2)}")

# Accuracy
test_acc = accuracy_score(Y_test, Y_test_pred)
print(f"The accuracy of the selected model on the test is {test_acc:.0%}")


# Visualize the selected model/tree 
from sklearn.tree import export_text
best_tree = dt_res.best_estimator_
tree_rules = export_text(best_tree, feature_names=list(X_train.columns))
print(tree_rules)

### Replication to see if performances stay stabe with different splits 

from sklearn.metrics import accuracy_score

heart = pd.read_csv("heart_failure.csv")
heart["age"] = heart["age"].astype(int)

prediction_variables = [col for col in heart.columns if col != "DEATH_EVENT" and col != "time"]
heart_X = heart[prediction_variables]
heart_Y = heart["DEATH_EVENT"]

meta_param = {'max_depth': range(1, 11),
              'min_samples_split': [2, 10, 25, 50, 75, 85, 100, 200]}

# strore the performances of the models 
accuracies = []

for seed in range(10):
    
    # splits
    X_train, X_test, Y_train, Y_test = train_test_split(heart_X, heart_Y,
                                                        test_size=0.2, 
                                                        random_state=seed,
                                                        stratify=heart_Y)
    v_split = StratifiedShuffleSplit(n_splits=1, train_size=0.75, random_state=seed)
    
    # model selection 
    dt = DecisionTreeClassifier(random_state=42)
    dt_search = GridSearchCV(dt, meta_param, cv=v_split, n_jobs = -1)
    
    # results
    dt_res = dt_search.fit(X_train, Y_train)
    Y_test_pred = dt_res.predict(X_test)
    
    # performance
    acc = accuracy_score(Y_test, Y_test_pred)
    
    accuracies.append(acc)

print(accuracies)

# --> performances are not very good and not stable (the data set is small)
