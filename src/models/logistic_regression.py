#%%
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_and_explore_csv(file_path):

    if os.path.exists(file_path):
        print(f" File found: {file_path}\n")

        # Load the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # Display first 5 rows
        print(" First 5 rows of the dataset:\n", df.head())

        # Display dataset info
        print("\n Dataset Information:")
        print(df.info())

        # Display basic statistics
        print("\n Summary Statistics:")
        print(df.describe())

        # Check for missing values
        print("\n Missing Values in Each Column:")
        print(df.isnull().sum())

        return df
    else:
        print(" File not found.")
        return None

# Example usage
file_path = "/Users/pratigyajamakatel/Downloads/data.csv"
df = load_and_explore_csv(file_path)

#  show a visualization
if df is not None:
    plt.figure(figsize=(8, 5))
    sns.heatmap(df.isnull(), cmap="viridis", cbar=False, yticklabels=False)
    plt.title("Missing Values Heatmap")
    plt.show()

#%%
import pandas as pd
import numpy as np

# Load the correct training and test datasets
training = pd.read_csv('/Users/pratigyajamakatel/Downloads/data.csv')  # Train dataset
test = pd.read_csv('/Users/pratigyajamakatel/Downloads/data.csv')  # Test dataset

# Mark train and test data
training['train_test'] = 1
test['train_test'] = 0
test['Survived'] = np.nan  # Placeholder for test labels

# Combine for preprocessing
all_data = pd.concat([training, test], axis=0, ignore_index=True)

# Display column names
print(all_data.columns)

#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_all_data(training, test, scale_features=True):


    # Combine training and test data
    training['train_test'] = 1
    test['train_test'] = 0
    test['Survived'] = np.nan  # Placeholder for test labels
    all_data = pd.concat([training, test], axis=0)

    # Feature Engineering
    all_data['cabin_multiple'] = all_data['Cabin'].apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))
    all_data['cabin_adv'] = all_data['Cabin'].apply(lambda x: str(x)[0])
    all_data['numeric_ticket'] = all_data['Ticket'].apply(lambda x: 1 if x.isnumeric() else 0)
    all_data['ticket_letters'] = all_data['Ticket'].apply(lambda x: ''.join(x.split(' ')[:-1]).replace('.','').replace('/','').lower()
                                                          if len(x.split(' ')[:-1]) > 0 else 0)
    all_data['name_title'] = all_data['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())

    # Handling missing values
    all_data['Age'] = all_data['Age'].fillna(training['Age'].median())
    all_data['Fare'] = all_data['Fare'].fillna(training['Fare'].median())

    # Drop rows where 'Embarked' is missing
    all_data.dropna(subset=['Embarked'], inplace=True)

    # Feature transformation
    all_data['norm_sibsp'] = np.log(all_data['SibSp'] + 1)
    all_data['norm_fare'] = np.log(all_data['Fare'] + 1)

    # Convert categorical column 'Pclass' to string
    all_data['Pclass'] = all_data['Pclass'].astype(str)

    # One-hot encoding categorical variables
    all_dummies = pd.get_dummies(all_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'norm_fare',
                                           'Embarked', 'cabin_adv', 'cabin_multiple', 'numeric_ticket',
                                           'name_title', 'train_test']])

    # Scale numerical features if enabled
    if scale_features:
        scale = StandardScaler()
        num_features = ['Age', 'SibSp', 'Parch', 'norm_fare']
        all_dummies[num_features] = scale.fit_transform(all_dummies[num_features])

    # Split train and test sets
    X_train = all_dummies[all_dummies.train_test == 1].drop(['train_test'], axis=1)
    X_test = all_dummies[all_dummies.train_test == 0].drop(['train_test'], axis=1)
    y_train = all_data[all_data.train_test == 1]['Survived']

    return X_train, X_test, y_train

# Example usage
X_train, X_test, y_train = preprocess_all_data(training, test, scale_features=True)
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}, y_train shape: {y_train.shape}")

#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_all_data(training, test, scale_features=True):


    # Combine training and test data
    training['train_test'] = 1
    test['train_test'] = 0
    test['Survived'] = np.nan  # Placeholder for test labels
    all_data = pd.concat([training, test], axis=0)

    # Feature Engineering
    all_data['cabin_multiple'] = all_data['Cabin'].apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))
    all_data['cabin_adv'] = all_data['Cabin'].apply(lambda x: str(x)[0])
    all_data['numeric_ticket'] = all_data['Ticket'].apply(lambda x: 1 if x.isnumeric() else 0)
    all_data['ticket_letters'] = all_data['Ticket'].apply(lambda x: ''.join(x.split(' ')[:-1]).replace('.','').replace('/','').lower()
                                                          if len(x.split(' ')[:-1]) > 0 else 0)
    all_data['name_title'] = all_data['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())

    # Handling missing values
    all_data['Age'] = all_data['Age'].fillna(training['Age'].median())
    all_data['Fare'] = all_data['Fare'].fillna(training['Fare'].median())

    # Drop rows where 'Embarked' is missing
    all_data.dropna(subset=['Embarked'], inplace=True)

    # Feature transformation
    all_data['norm_sibsp'] = np.log(all_data['SibSp'] + 1)
    all_data['norm_fare'] = np.log(all_data['Fare'] + 1)

    # Convert categorical column 'Pclass' to string
    all_data['Pclass'] = all_data['Pclass'].astype(str)

    # One-hot encoding categorical variables
    all_dummies = pd.get_dummies(all_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'norm_fare',
                                           'Embarked', 'cabin_adv', 'cabin_multiple', 'numeric_ticket',
                                           'name_title', 'train_test']])

    # Scale numerical features if enabled
    if scale_features:
        scale = StandardScaler()
        num_features = ['Age', 'SibSp', 'Parch', 'norm_fare']
        all_dummies[num_features] = scale.fit_transform(all_dummies[num_features])

    # Split train and test sets
    X_train = all_dummies[all_dummies.train_test == 1].drop(['train_test'], axis=1)
    X_test = all_dummies[all_dummies.train_test == 0].drop(['train_test'], axis=1)
    y_train = all_data[all_data.train_test == 1]['Survived']

    return X_train, X_test, y_train

# Example usage
X_train, X_test, y_train = preprocess_all_data(training, test, scale_features=True)
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}, y_train shape: {y_train.shape}")

#%%
#create all categorical variables that we did above for both training and test sets
all_data['cabin_multiple'] = all_data.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))
all_data['cabin_adv'] = all_data.Cabin.apply(lambda x: str(x)[0])
all_data['numeric_ticket'] = all_data.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)
all_data['ticket_letters'] = all_data.Ticket.apply(lambda x: ''.join(x.split(' ')[:-1]).replace('.','').replace('/','').lower() if len(x.split(' ')[:-1]) >0 else 0)
all_data['name_title'] = all_data.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())

#impute nulls for continuous data
#all_data.Age = all_data.Age.fillna(training.Age.mean())
all_data.Age = all_data.Age.fillna(training.Age.median())
#all_data.Fare = all_data.Fare.fillna(training.Fare.mean())
all_data.Fare = all_data.Fare.fillna(training.Fare.median())

#drop null 'embarked' rows. Only 2 instances of this in training and 0 in test
all_data.dropna(subset=['Embarked'],inplace = True)

#tried log norm of sibsp (not used)
all_data['norm_sibsp'] = np.log(all_data.SibSp+1)
all_data['norm_sibsp'].hist()
# log norm of fare (used)
all_data['norm_fare'] = np.log(all_data.Fare+1)
all_data['norm_fare'].hist()

# converted fare to category for pd.get_dummies()
all_data.Pclass = all_data.Pclass.astype(str)

#created dummy variables from categories (also can use OneHotEncoder)
all_dummies = pd.get_dummies(all_data[['Pclass','Sex','Age','SibSp','Parch','norm_fare','Embarked','cabin_adv','cabin_multiple','numeric_ticket','name_title','train_test']])

#Split to train test again
X_train = all_dummies[all_dummies.train_test == 1].drop(['train_test'], axis =1)
X_test = all_dummies[all_dummies.train_test == 0].drop(['train_test'], axis =1)


y_train = all_data[all_data.train_test==1].Survived
y_train.shape
#%%
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
#%%
# Scale data
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
all_dummies_scaled = all_dummies.copy()
all_dummies_scaled[['Age','SibSp','Parch','norm_fare']]= scale.fit_transform(all_dummies_scaled[['Age','SibSp','Parch','norm_fare']])
all_dummies_scaled

X_train_scaled = all_dummies_scaled[all_dummies_scaled.train_test == 1].drop(['train_test'], axis =1)
X_test_scaled = all_dummies_scaled[all_dummies_scaled.train_test == 0].drop(['train_test'], axis =1)

y_train = all_data[all_data.train_test==1].Survived
#%%
# Scale data
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
all_dummies_scaled = all_dummies.copy()
all_dummies_scaled[['Age','SibSp','Parch','norm_fare']]= scale.fit_transform(all_dummies_scaled[['Age','SibSp','Parch','norm_fare']])
all_dummies_scaled

X_train_scaled = all_dummies_scaled[all_dummies_scaled.train_test == 1].drop(['train_test'], axis =1)
X_test_scaled = all_dummies_scaled[all_dummies_scaled.train_test == 0].drop(['train_test'], axis =1)

y_train = all_data[all_data.train_test==1].Survived
#%%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

def evaluate_logistic_regression(X_train, y_train, cv_folds=5):
   
    lr = LogisticRegression(max_iter=2000)
    cv_scores = cross_val_score(lr, X_train, y_train, cv=cv_folds)

    print("Cross-validation scores:", cv_scores)
    print("Mean accuracy:", cv_scores.mean())

    return cv_scores, cv_scores.mean()

# Example usage
cv_scores, mean_score = evaluate_logistic_regression(X_train, y_train)

#%%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

def evaluate_logistic_regression_scaled(X_train_scaled, y_train, cv_folds=5):
   
    lr = LogisticRegression(max_iter=2000)
    cv_scores = cross_val_score(lr, X_train_scaled, y_train, cv=cv_folds)

    print("Cross-validation scores:", cv_scores)
    print("Mean accuracy:", cv_scores.mean())

    return cv_scores, cv_scores.mean()

# Example usage
cv_scores, mean_score = evaluate_logistic_regression_scaled(X_train_scaled, y_train)
