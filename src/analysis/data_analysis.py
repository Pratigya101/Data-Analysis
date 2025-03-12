#%%
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_csv(file_path):

    if os.path.exists(file_path):
        print(f"File found: {file_path}")
        df = pd.read_csv(file_path)
        print("First 5 rows of the dataset:\n", df.head())  # Display first few rows
        return df
    else:
        print("File not found.")
        return None

# Example usage
file_path = "/Users/pratigyajamakatel/Downloads/data.csv"
df = load_csv(file_path)

# Additional analysis (if data is loaded successfully)
if df is not None:
    print("\nDataset Information:")
    print(df.info())  # Show dataset info

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

# If the data is loaded successfully, show a visualization
if df is not None:
    plt.figure(figsize=(8, 5))
    sns.heatmap(df.isnull(), cmap="viridis", cbar=False, yticklabels=False)
    plt.title("Missing Values Heatmap")
    plt.show()

#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_and_prepare_data(train_path, test_path):

    # Check if files exist
    if not os.path.exists(train_path):
        print(f" Training file not found: {train_path}")
        return None
    if not os.path.exists(test_path):
        print(f" Test file not found: {test_path}")
        return None

    # Load CSV files
    training = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # Add 'train_test' indicator columns
    training['train_test'] = 1
    test['train_test'] = 0

    # Add 'Survived' column to test set as NaN
    test['Survived'] = np.nan

    # Combine both datasets
    all_data = pd.concat([training, test], ignore_index=True)

    print(" Data successfully loaded and combined!\n")

    # Return the combined dataset
    return all_data

# Example usage
train_file_path = "/Users/pratigyajamakatel/Downloads/data.csv"
test_file_path = "/Users/pratigyajamakatel/Downloads/data.csv"

all_data = load_and_prepare_data(train_file_path, test_file_path)

# If data is loaded successfully, display column names
if all_data is not None:
    print(" Columns in the combined dataset:\n", all_data.columns)

#%%
def quick_summary(df, name="Dataset"):

    print(f"\n Summary Statistics for {name}")
    print("-" * 40)
    print(df.describe())

# Example usage
quick_summary(training, "Training Data")

#%%
def numeric_summary(df, name="Dataset"):

    numeric_df = df.select_dtypes(include=['number'])  # Select only numeric columns
    print(f"\n Summary Statistics for Numeric Columns in {name}")
    print("-" * 50)
    print(numeric_df.describe())

    return numeric_df.describe()

# Example usage
numeric_summary(training, "Training Data")

#%%
def split_numeric_categorical(df):

    numeric_df = df.select_dtypes(include=['number'])  # Numeric columns
    categorical_df = df.select_dtypes(exclude=['number'])  # Categorical columns

    return numeric_df, categorical_df

# Example usage
df_num, df_cat = split_numeric_categorical(training)

# Display results
print(" Numeric Columns:\n", df_num.head())
print("\n Categorical Columns:\n", df_cat.head())

#%%
import matplotlib.pyplot as plt

def plot_numeric_distributions(df_num):

    for col in df_num.columns:
        plt.figure(figsize=(6, 4))
        plt.hist(df_num[col], bins=30, edgecolor='black', alpha=0.7)
        plt.title(col)
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.show()

# Example usage
plot_numeric_distributions(df_num)

#%%
import seaborn as sns
import matplotlib.pyplot as plt

def plot_correlation_heatmap(df_num):

    corr_matrix = df_num.corr()  # Compute correlation
    print(" Correlation Matrix:\n", corr_matrix)

    # Plot heatmap
    plt.figure(figsize=(6, 4))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.show()

# Example usage
plot_correlation_heatmap(df_num)

#%%
# compare survival rate across Age, SibSp, Parch, and Fare
pd.pivot_table(training, index = 'Survived', values = ['Age','SibSp','Parch','Fare'])
#%%
import matplotlib.pyplot as plt
import seaborn as sns

def plot_categorical_distributions(df_cat):

    for col in df_cat.columns:
        plt.figure(figsize=(8, 4))  # Adjust figure size

        sns.countplot(data=df_cat, x=col, hue=col, palette="viridis")  # Assign hue

        plt.title(col)  # Set title
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.legend(title=col)  # Add legend with column name
        plt.show()

# Example usage
plot_categorical_distributions(df_cat)

#%%
import pandas as pd

def compare_survival_by_category(training):

    # Survival by Passenger Class (Pclass)
    survival_pclass = pd.pivot_table(training, index='Survived', columns='Pclass', values='Ticket', aggfunc='count')

    # Survival by Gender (Sex)
    survival_sex = pd.pivot_table(training, index='Survived', columns='Sex', values='Ticket', aggfunc='count')

    # Survival by Embarkation Port (Embarked)
    survival_embarked = pd.pivot_table(training, index='Survived', columns='Embarked', values='Ticket', aggfunc='count')

    return survival_pclass, survival_sex, survival_embarked


# Example Usage:
pclass_result, sex_result, embarked_result = compare_survival_by_category(training)

# Print results
print("Survival by Pclass:\n", pclass_result)
print("\nSurvival by Sex:\n", sex_result)
print("\nSurvival by Embarked:\n", embarked_result)

#%%
import pandas as pd

def create_cabin_multiple(df):

    df['cabin_multiple'] = df['Cabin'].apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))
    return df['cabin_multiple'].value_counts()

# Example usage
cabin_counts = create_cabin_multiple(training)
print(cabin_counts)

#%%
import pandas as pd

def survival_by_cabin_multiple(training):

    pivot_table = pd.pivot_table(training, index='Survived', columns='cabin_multiple', values='Ticket', aggfunc='count')
    return pivot_table

# Example Usage:
cabin_multiple_result = survival_by_cabin_multiple(training)

# Print result
print("Survival by Cabin Multiple:\n", cabin_multiple_result)

#%%
import pandas as pd

def create_cabin_category(df, cabin_col='Cabin', new_col='cabin_adv'):

    df[new_col] = df[cabin_col].apply(lambda x: str(x)[0] if pd.notna(x) else 'N')
    return df

# Example Usage:
training = create_cabin_category(training)

# Display the first few rows to verify
print(training[['Cabin', 'cabin_adv']].head())

#%%
import pandas as pd

def analyze_survival_by_cabin(training):

    # Display value counts of cabin categories
    print("Cabin Category Counts:\n", training.cabin_adv.value_counts())

    # Create pivot table to compare survival rates by cabin category
    pivot_table = pd.pivot_table(training, index='Survived', columns='cabin_adv', values='Name', aggfunc='count')

    return pivot_table


# Example Usage:
pivot_result = analyze_survival_by_cabin(training)
print(pivot_result)

#%%
import pandas as pd

def process_ticket_features(df):

    df['numeric_ticket'] = df['Ticket'].apply(lambda x: 1 if x.isnumeric() else 0)
    df['ticket_letters'] = df['Ticket'].apply(lambda x:
        ''.join(x.split(' ')[:-1]).replace('.', '').replace('/', '').lower()
        if len(x.split(' ')[:-1]) > 0 else '0'
    )
    return df

# Example usage:
training = process_ticket_features(training)

# Display first few rows to verify
print(training[['Ticket', 'numeric_ticket', 'ticket_letters']].head())

#%%
import pandas as pd

def count_numeric_tickets(data, column='Ticket'):

    data['numeric_ticket'] = data[column].apply(lambda x: 1 if x.isnumeric() else 0)
    return data['numeric_ticket'].value_counts()

# Example usage:
ticket_counts = count_numeric_tickets(training)
print(ticket_counts)

#%%
import pandas as pd

# Ensure the correct option name is used
pd.set_option("display.max_rows", None)  #  Corrected option name

# Check if 'ticket_letters' column exists before calling value_counts()
if 'ticket_letters' in training.columns:
    print(training['ticket_letters'].value_counts())  #  Fix applied
else:
    print("Column 'ticket_letters' does not exist in the DataFrame.")

#%%
import pandas as pd

def survival_rate_by_ticket_type(df, ticket_col='Ticket', numeric_col='numeric_ticket'):

    return pd.pivot_table(df, index='Survived', columns=numeric_col, values=ticket_col, aggfunc='count')

# Example Usage:
survival_ticket_pivot = survival_rate_by_ticket_type(training)
print(survival_ticket_pivot)

#%%
#survival rate across different tyicket types
pd.pivot_table(training,index='Survived',columns='ticket_letters', values = 'Ticket', aggfunc='count')
#%%
import pandas as pd

def extract_name_title(df):

    if 'Name' in df.columns:
        df['name_title'] = df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
        return df['name_title'].value_counts()  # Return frequency of each title
    else:
        print(" Column 'Name' does not exist in the DataFrame.")
        return None

# Example usage
title_counts = extract_name_title(training)
print(title_counts)

#%%
training['name_title'].value_counts()
#%% md
# Data Preprocessing
# 
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
    all_data['ticket_letters'] = all_data['Ticket'].apply(
        lambda x: ''.join(x.split(' ')[:-1]).replace('.', '').replace('/', '').lower()
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

#create all categorical variables that we did above for both training and test sets
all_data['cabin_multiple'] = all_data.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))
all_data['cabin_adv'] = all_data.Cabin.apply(lambda x: str(x)[0])
all_data['numeric_ticket'] = all_data.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)
all_data['ticket_letters'] = all_data.Ticket.apply(
    lambda x: ''.join(x.split(' ')[:-1]).replace('.', '').replace('/', '').lower() if len(x.split(' ')[:-1]) > 0 else 0)
all_data['name_title'] = all_data.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())

#impute nulls for continuous data
#all_data.Age = all_data.Age.fillna(training.Age.mean())
all_data.Age = all_data.Age.fillna(training.Age.median())
#all_data.Fare = all_data.Fare.fillna(training.Fare.mean())
all_data.Fare = all_data.Fare.fillna(training.Fare.median())

#drop null 'embarked' rows. Only 2 instances of this in training and 0 in test
all_data.dropna(subset=['Embarked'], inplace=True)

#tried log norm of sibsp (not used)
all_data['norm_sibsp'] = np.log(all_data.SibSp + 1)
all_data['norm_sibsp'].hist()
# log norm of fare (used)
all_data['norm_fare'] = np.log(all_data.Fare + 1)
all_data['norm_fare'].hist()

# converted fare to category for pd.get_dummies()
all_data.Pclass = all_data.Pclass.astype(str)

#created dummy variables from categories (also can use OneHotEncoder)
all_dummies = pd.get_dummies(all_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'norm_fare', 'Embarked', 'cabin_adv',
                                       'cabin_multiple', 'numeric_ticket', 'name_title', 'train_test']])

#Split to train test again
X_train = all_dummies[all_dummies.train_test == 1].drop(['train_test'], axis=1)
X_test = all_dummies[all_dummies.train_test == 0].drop(['train_test'], axis=1)

y_train = all_data[all_data.train_test == 1].Survived
y_train.shape
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
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
#%%
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

def evaluate_naive_bayes(X_train, y_train, cv_folds=5):

    gnb = GaussianNB()
    cv_scores = cross_val_score(gnb, X_train, y_train, cv=cv_folds)

    print("Cross-validation scores:", cv_scores)
    print("Mean accuracy:", cv_scores.mean())

    return cv_scores, cv_scores.mean()

# Example usage
cv_scores, mean_score = evaluate_naive_bayes(X_train, y_train)

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
#%%
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

def evaluate_model(model, X_train, y_train, cv_folds=5):

    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds)

    print(f"Model: {model.__class__.__name__}")
    print("Cross-validation scores:", cv_scores)
    print("Mean accuracy:", cv_scores.mean())

    return cv_scores, cv_scores.mean()

# Example usage with Logistic Regression and Naïve Bayes
lr_model = LogisticRegression(max_iter=2000)
nb_model = GaussianNB()

print("Evaluating Logistic Regression:")
lr_scores, lr_mean = evaluate_model(lr_model, X_train, y_train)

print("\nEvaluating Naïve Bayes:")
nb_scores, nb_mean = evaluate_model(nb_model, X_train, y_train)

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

#%%
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

def evaluate_decision_tree(X_train, y_train, cv_folds=5, random_state=1):

    dt = DecisionTreeClassifier(random_state=random_state)
    cv_scores = cross_val_score(dt, X_train, y_train, cv=cv_folds)

    print("Cross-validation scores:", cv_scores)
    print("Mean accuracy:", cv_scores.mean())

    return cv_scores, cv_scores.mean()

# Example usage
cv_scores, mean_score = evaluate_decision_tree(X_train, y_train)
#%%
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

def evaluate_decision_tree_scaled(X_train_scaled, y_train, cv_folds=5, random_state=1):

    dt = DecisionTreeClassifier(random_state=random_state)
    cv_scores = cross_val_score(dt, X_train_scaled, y_train, cv=cv_folds)

    print("Cross-validation scores:", cv_scores)
    print("Mean accuracy:", cv_scores.mean())

    return cv_scores, cv_scores.mean()

# Example usage
cv_scores, mean_score = evaluate_decision_tree_scaled(X_train_scaled, y_train)
#%%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

def evaluate_knn(X_train, y_train, cv_folds=5, n_neighbors=5):

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    cv_scores = cross_val_score(knn, X_train, y_train, cv=cv_folds)

    print(f"KNN (n_neighbors={n_neighbors}) - Cross-validation scores:", cv_scores)
    print("Mean accuracy:", cv_scores.mean())

    return cv_scores, cv_scores.mean()

# Example usage
cv_scores, mean_score = evaluate_knn(X_train, y_train)
#%%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

def evaluate_knn_scaled(X_train_scaled, y_train, cv_folds=5, n_neighbors=5):

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    cv_scores = cross_val_score(knn, X_train_scaled, y_train, cv=cv_folds)

    print(f"KNN (n_neighbors={n_neighbors}) - Cross-validation scores:", cv_scores)
    print("Mean accuracy:", cv_scores.mean())

    return cv_scores, cv_scores.mean()

# Example usage
cv_scores, mean_score = evaluate_knn_scaled(X_train_scaled, y_train)
#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def evaluate_random_forest(X_train, y_train, cv_folds=5, random_state=1):

    # Initialize the Random Forest model
    rf = RandomForestClassifier(random_state=random_state)

    # Perform cross-validation
    cv_scores = cross_val_score(rf, X_train, y_train, cv=cv_folds)

    # Print results
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean cross-validation score: {cv_scores.mean():.4f}")

    return cv_scores, cv_scores.mean()

# Example usage
cv_results, mean_cv_score = evaluate_random_forest(X_train, y_train)

#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def evaluate_random_forest_scaled(X_train_scaled, y_train, cv_folds=5, random_state=1):

    # Initialize the Random Forest model
    rf = RandomForestClassifier(random_state=random_state)

    # Perform cross-validation
    cv_scores = cross_val_score(rf, X_train_scaled, y_train, cv=cv_folds)

    # Print results
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean cross-validation score: {cv_scores.mean():.4f}")

    return cv_scores, cv_scores.mean()

# Example usage
cv_results, mean_cv_score = evaluate_random_forest_scaled(X_train_scaled, y_train)

#%%
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

def evaluate_svc_scaled(X_train_scaled, y_train, cv_folds=5):

    # Initialize the Support Vector Classifier
    svc = SVC(probability=True)

    # Perform cross-validation
    cv_scores = cross_val_score(svc, X_train_scaled, y_train, cv=cv_folds)

    # Print results
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean cross-validation score: {cv_scores.mean():.4f}")

    return cv_scores, cv_scores.mean()

# Example usage
cv_results_svc, mean_cv_score_svc = evaluate_svc_scaled(X_train_scaled, y_train)

#%%
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

def evaluate_xgb_scaled(X_train_scaled, y_train, cv_folds=5):

    # Initialize the XGBoost classifier
    xgb = XGBClassifier(random_state=1, use_label_encoder=False, eval_metric='logloss')

    # Perform cross-validation
    cv_scores = cross_val_score(xgb, X_train_scaled, y_train, cv=cv_folds)

    # Print results
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean cross-validation score: {cv_scores.mean():.4f}")

    return cv_scores, cv_scores.mean()

# Example usage
cv_results_xgb, mean_cv_score_xgb = evaluate_xgb_scaled(X_train_scaled, y_train)

#%%
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Define models
lr = LogisticRegression(max_iter=2000, penalty='l2', C=1.0, solver='liblinear', random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)
rf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
gnb = GaussianNB()
svc = SVC(probability=True, kernel='rbf', C=1.0, random_state=42)
xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

#%%
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score

def evaluate_voting_classifier(models, X_train_scaled, y_train, voting_type='soft', cv_folds=5):

    # Initialize Voting Classifier
    voting_clf = VotingClassifier(estimators=models, voting=voting_type)

    # Perform cross-validation
    cv_scores = cross_val_score(voting_clf, X_train_scaled, y_train, cv=cv_folds)

    # Print results
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean cross-validation score: {cv_scores.mean():.4f}")

    return cv_scores, cv_scores.mean()

# Define models
models_list = [
    ('lr', lr),
    ('knn', knn),
    ('rf', rf),
    ('gnb', gnb),
    ('svc', svc),
    ('xgb', xgb)
]

# Example usage
cv_results_voting, mean_cv_score_voting = evaluate_voting_classifier(models_list, X_train_scaled, y_train)

#%%
#Voting classifier takes all of the inputs and averages the results. For a "hard" voting classifier each classifier gets 1 vote "yes" or "no" and the result is just a popular vote. For this, you generally want odd numbers
#A "soft" classifier averages the confidence of each of the models. If a the average confidence is > 50% that it is a 1 it will be counted as such
from sklearn.ensemble import VotingClassifier
voting_clf = VotingClassifier(estimators = [('lr',lr),('knn',knn),('rf',rf),('gnb',gnb),('svc',svc),('xgb',xgb)], voting = 'soft')
#%%
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score

def train_voting_classifier(models, X_train_scaled, y_train, voting_type='soft', cv_folds=5):

    # Initialize Voting Classifier
    voting_clf = VotingClassifier(estimators=models, voting=voting_type)

    # Perform cross-validation
    cv_scores = cross_val_score(voting_clf, X_train_scaled, y_train, cv=cv_folds)

    # Train the classifier on the full training data
    voting_clf.fit(X_train_scaled, y_train)

    # Print results
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean cross-validation score: {cv_scores.mean():.4f}")

    return voting_clf, cv_scores.mean()

# Define models
models_list = [
    ('lr', lr),
    ('knn', knn),
    ('rf', rf),
    ('gnb', gnb),
    ('svc', svc),
    ('xgb', xgb)
]

# Train the Voting Classifier
voting_clf_trained, mean_cv_score = train_voting_classifier(models_list, X_train_scaled, y_train)

#%%
cv = cross_val_score(voting_clf,X_train_scaled,y_train,cv=5)
print(cv)
print(cv.mean())
#%%
from sklearn.model_selection import cross_val_score

def evaluate_model(model, X_train_scaled, y_train, cv_folds=5):

    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_folds)

    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean cross-validation score: {cv_scores.mean():.4f}")

    return cv_scores, cv_scores.mean()

# Example usage: Evaluating Voting Classifier
cv_results_voting, mean_cv_score_voting = evaluate_model(voting_clf, X_train_scaled, y_train)

#%%
# Ensure the lengths match
if len(X_test_scaled) != len(test):
    print(f"Mismatch: X_test_scaled has {len(X_test_scaled)} rows, but test has {len(test)} rows.")

# Generate predictions
y_hat_base_vc = voting_clf.predict(X_test_scaled).astype(int)

# Ensure lengths match before creating DataFrame
if len(y_hat_base_vc) == len(test.PassengerId):
    base_submission = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': y_hat_base_vc})
    base_submission.to_csv('base_submission.csv', index=False)
    print("Submission file saved successfully!")
else:
    print("Error: Prediction array length does not match PassengerId length.")

#%%
import pandas as pd

def generate_submission(model, X_test_scaled, test, filename='submission.csv'):

    # Ensure the lengths match
    if len(X_test_scaled) != len(test):
        print(f"Warning: X_test_scaled has {len(X_test_scaled)} rows, but test has {len(test)} rows.")
        return

    # Generate predictions
    y_predictions = model.predict(X_test_scaled).astype(int)

    # Ensure predictions length matches the number of PassengerIds
    if len(y_predictions) == len(test.PassengerId):
        submission_df = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': y_predictions})
        submission_df.to_csv(filename, index=False)
        print(f"Submission file '{filename}' saved successfully!")
    else:
        print("Error: Prediction array length does not match PassengerId length.")

# Example usage:
generate_submission(voting_clf, X_test_scaled, test, filename='base_submission.csv')

#%%
import pandas as pd

# Generate predictions
y_hat_base_vc = voting_clf.predict(X_test_scaled).astype(int)

# Ensure lengths match
if len(y_hat_base_vc) == len(test):
    base_submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': y_hat_base_vc})
    base_submission.to_csv('base_submission.csv', index=False)
    print("Submission file saved successfully!")
else:
    print(f"Error: Prediction array length ({len(y_hat_base_vc)}) does not match PassengerId length ({len(test)})")

#%%
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
#%%
#simple performance reporting function
def clf_performance(classifier, model_name):
    print(model_name)
    print('Best Score: ' + str(classifier.best_score_))
    print('Best Parameters: ' + str(classifier.best_params_))
#%%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import numpy as np

def tune_logistic_regression(X_train_scaled, y_train):

    lr = LogisticRegression()

    param_grid = {
        'max_iter': [2000],
        'penalty': ['l1', 'l2'],
        'C': np.logspace(-4, 4, 20),
        'solver': ['liblinear']
    }

    clf_lr = GridSearchCV(lr, param_grid=param_grid, cv=5, verbose=True, n_jobs=-1)
    best_clf_lr = clf_lr.fit(X_train_scaled, y_train)

    print("Best Logistic Regression Parameters:", best_clf_lr.best_params_)
    print(f"Best Score: {best_clf_lr.best_score_:.4f}")

    return best_clf_lr.best_estimator_, best_clf_lr.best_params_

# Example usage
best_lr_model, best_lr_params = tune_logistic_regression(X_train_scaled, y_train)

#%%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

def tune_knn(X_train_scaled, y_train):

    knn = KNeighborsClassifier()

    param_grid = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree'],
        'p': [1, 2]  # Minkowski distance metric (1: Manhattan, 2: Euclidean)
    }

    clf_knn = GridSearchCV(knn, param_grid=param_grid, cv=5, verbose=True, n_jobs=-1)
    best_clf_knn = clf_knn.fit(X_train_scaled, y_train)

    print("Best KNN Parameters:", best_clf_knn.best_params_)
    print(f"Best Score: {best_clf_knn.best_score_:.4f}")

    return best_clf_knn.best_estimator_, best_clf_knn.best_params_

# Example usage
best_knn_model, best_knn_params = tune_knn(X_train_scaled, y_train)

#%%
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV

def tune_svm(X_train_scaled, y_train, cv_folds=3, n_iter_search=10):

    svc = SVC(probability=True)

    param_grid = [
        {'kernel': ['rbf'], 'gamma': [0.1, 1, 10], 'C': [0.1, 1, 10, 100]},
        {'kernel': ['linear'], 'C': [0.1, 1, 10]},
        {'kernel': ['poly'], 'degree': [2, 3], 'C': [0.1, 1, 10]}
    ]

    clf_svc = RandomizedSearchCV(svc, param_distributions=param_grid, cv=cv_folds,
                                 verbose=2, n_jobs=-1, n_iter=n_iter_search, random_state=42)

    best_clf_svc = clf_svc.fit(X_train_scaled, y_train)

    print("Best SVM Parameters:", best_clf_svc.best_params_)
    print(f"Best Score: {best_clf_svc.best_score_:.4f}")

    return best_clf_svc.best_estimator_, best_clf_svc.best_params_

# Example usage
best_svm_model, best_svm_params = tune_svm(X_train_scaled, y_train)

#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def tune_random_forest(X_train_scaled, y_train, cv_folds=5):

    rf = RandomForestClassifier(random_state=1)

    param_grid = {
        'n_estimators': [400, 450, 500, 550],
        'criterion': ['gini', 'entropy'],
        'bootstrap': [True],
        'max_depth': [15, 20, 25],
        'max_features': ['auto', 'sqrt', 10],
        'min_samples_leaf': [2, 3],
        'min_samples_split': [2, 3]
    }

    clf_rf = GridSearchCV(rf, param_grid=param_grid, cv=cv_folds, verbose=True, n_jobs=-1)

    best_clf_rf = clf_rf.fit(X_train_scaled, y_train)

    print("Best Random Forest Parameters:", best_clf_rf.best_params_)
    print(f"Best Score: {best_clf_rf.best_score_:.4f}")

    return best_clf_rf.best_estimator_, best_clf_rf.best_params_

# Example usage
best_rf_model, best_rf_params = tune_random_forest(X_train_scaled, y_train)

#%%
import pandas as pd
import matplotlib.pyplot as plt

def plot_feature_importance(model, X_train_scaled, top_n=20):

    # Extract feature importances
    feat_importances = pd.Series(model.feature_importances_, index=X_train_scaled.columns)

    # Plot the top N important features
    plt.figure(figsize=(10, 6))
    feat_importances.nlargest(top_n).plot(kind='barh', color='steelblue')
    plt.title(f"Top {top_n} Feature Importances")
    plt.xlabel("Feature Importance Score")
    plt.ylabel("Features")
    plt.gca().invert_yaxis()  # Invert to show the highest importance on top
    plt.show()

# Train the best Random Forest model
best_rf = best_clf_rf.best_estimator_.fit(X_train_scaled, y_train)

# Plot feature importance
plot_feature_importance(best_rf, X_train_scaled)

#%%
xgb = XGBClassifier(random_state = 1)

param_grid = {
    'n_estimators': [450,500,550],
    'colsample_bytree': [0.75,0.8,0.85],
    'max_depth': [None],
    'reg_alpha': [1],
    'reg_lambda': [2, 5, 10],
    'subsample': [0.55, 0.6, .65],
    'learning_rate':[0.5],
    'gamma':[.5,1,2],
    'min_child_weight':[0.01],
    'sampling_method': ['uniform']
}

clf_xgb = GridSearchCV(xgb, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
best_clf_xgb = clf_xgb.fit(X_train_scaled,y_train)
clf_performance(best_clf_xgb,'XGB')
#%%
print(best_clf_lr if 'best_clf_lr' in globals() else "best_clf_lr is NOT defined")
print(best_clf_knn if 'best_clf_knn' in globals() else "best_clf_knn is NOT defined")
print(best_clf_svc if 'best_clf_svc' in globals() else "best_clf_svc is NOT defined")
print(best_clf_rf if 'best_clf_rf' in globals() else "best_clf_rf is NOT defined")
print(best_clf_xgb if 'best_clf_xgb' in globals() else "best_clf_xgb is NOT defined")

#%%
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score

def evaluate_voting_classifiers(classifiers, X_train, y_train, cv_folds=5):

    results = {}

    for name, clf in classifiers.items():
        print(f"Evaluating {name}...")
        scores = cross_val_score(clf, X_train, y_train, cv=cv_folds)
        mean_score = scores.mean()
        results[name] = mean_score
        print(f"{name} - Mean CV Score: {mean_score:.4f}\n")

    return results


# Define optimized models
best_lr = best_clf_lr.best_estimator_
best_knn = best_clf_knn.best_estimator_
best_svc = best_clf_svc.best_estimator_
best_rf = best_clf_rf.best_estimator_
best_xgb = best_clf_xgb.best_estimator_

# Define Voting Classifiers
voting_classifiers = {
    "voting_clf_hard": VotingClassifier(estimators=[('knn', best_knn), ('rf', best_rf), ('svc', best_svc)], voting='hard'),
    "voting_clf_soft": VotingClassifier(estimators=[('knn', best_knn), ('rf', best_rf), ('svc', best_svc)], voting='soft'),
    "voting_clf_all": VotingClassifier(estimators=[('knn', best_knn), ('rf', best_rf), ('svc', best_svc), ('lr', best_lr)], voting='soft'),
    "voting_clf_xgb": VotingClassifier(estimators=[('knn', best_knn), ('rf', best_rf), ('svc', best_svc), ('xgb', best_xgb), ('lr', best_lr)], voting='soft'),
}

# Evaluate all Voting Classifiers
cv_scores = evaluate_voting_classifiers(voting_classifiers, X_train, y_train)

# Print summary
print("Final Voting Classifier Scores:", cv_scores)

#%%


best_lr = best_clf_lr.best_estimator_
best_knn = best_clf_knn.best_estimator_
best_svc = best_clf_svc.best_estimator_
best_rf = best_clf_rf.best_estimator_
best_xgb = best_clf_xgb.best_estimator_

voting_clf_hard = VotingClassifier(estimators = [('knn',best_knn),('rf',best_rf),('svc',best_svc)], voting = 'hard')
voting_clf_soft = VotingClassifier(estimators = [('knn',best_knn),('rf',best_rf),('svc',best_svc)], voting = 'soft')
voting_clf_all = VotingClassifier(estimators = [('knn',best_knn),('rf',best_rf),('svc',best_svc), ('lr', best_lr)], voting = 'soft')
voting_clf_xgb = VotingClassifier(estimators = [('knn',best_knn),('rf',best_rf),('svc',best_svc), ('xgb', best_xgb),('lr', best_lr)], voting = 'soft')

print('voting_clf_hard :',cross_val_score(voting_clf_hard,X_train,y_train,cv=5))
print('voting_clf_hard mean :',cross_val_score(voting_clf_hard,X_train,y_train,cv=5).mean())

print('voting_clf_soft :',cross_val_score(voting_clf_soft,X_train,y_train,cv=5))
print('voting_clf_soft mean :',cross_val_score(voting_clf_soft,X_train,y_train,cv=5).mean())

print('voting_clf_all :',cross_val_score(voting_clf_all,X_train,y_train,cv=5))
print('voting_clf_all mean :',cross_val_score(voting_clf_all,X_train,y_train,cv=5).mean())

print('voting_clf_xgb :',cross_val_score(voting_clf_xgb,X_train,y_train,cv=5))
print('voting_clf_xgb mean :',cross_val_score(voting_clf_xgb,X_train,y_train,cv=5).mean())
#%%
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score

def evaluate_voting_classifiers(models, X_train_scaled, y_train, cv_folds=5):

    results = {}

    for name, clf in models.items():
        scores = cross_val_score(clf, X_train_scaled, y_train, cv=cv_folds)
        mean_score = scores.mean()
        std_score = scores.std()

        print(f"{name} Accuracy: {mean_score:.4f} (+/- {std_score:.4f})")
        results[name] = mean_score

    return results


# Define best models from GridSearchCV
best_lr = best_clf_lr.best_estimator_
best_knn = best_clf_knn.best_estimator_
best_svc = best_clf_svc.best_estimator_
best_rf = best_clf_rf.best_estimator_
best_xgb = best_clf_xgb.best_estimator_

# Define Voting Classifiers
voting_classifiers = {
    "Hard Voting": VotingClassifier(estimators=[('knn', best_knn), ('rf', best_rf), ('svc', best_svc)], voting='hard'),
    "Soft Voting": VotingClassifier(estimators=[('knn', best_knn), ('rf', best_rf), ('svc', best_svc)], voting='soft'),
    "Soft Voting + LR": VotingClassifier(estimators=[('knn', best_knn), ('rf', best_rf), ('svc', best_svc), ('lr', best_lr)], voting='soft'),
    "Soft Voting + XGB": VotingClassifier(estimators=[('knn', best_knn), ('rf', best_rf), ('svc', best_svc), ('xgb', best_xgb), ('lr', best_lr)], voting='soft')
}

# Run the evaluation
voting_results = evaluate_voting_classifiers(voting_classifiers, X_train_scaled, y_train)

#%%
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV

def optimize_voting_weights(voting_clf, X_train_scaled, y_train, weight_options=None, cv_folds=5):

    if weight_options is None:
        weight_options = [[1,1,1], [1,2,1], [1,1,2], [2,1,1], [2,2,1], [1,2,2], [2,1,2]]

    # Define parameter grid
    params = {'weights': weight_options}

    # Perform GridSearch for weight optimization
    vote_weight_search = GridSearchCV(voting_clf, param_grid=params, cv=cv_folds, verbose=True, n_jobs=-1)
    best_clf_weight = vote_weight_search.fit(X_train_scaled, y_train)

    # Evaluate performance
    clf_performance(best_clf_weight, 'Optimized Soft VC Weights')

    return best_clf_weight.best_estimator_


# Define Soft Voting Classifier with initial equal weights
voting_clf_soft = VotingClassifier(estimators=[('knn', best_knn), ('rf', best_rf), ('svc', best_svc)], voting='soft')

# Run weight optimization
best_voting_clf_weighted = optimize_voting_weights(voting_clf_soft, X_train_scaled, y_train)

# Generate final predictions
voting_clf_sub = best_voting_clf_weighted.predict(X_test_scaled)

#%%
def train_and_predict(models, X_train_scaled, y_train, X_test_scaled):

    predictions = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_scaled, y_train)  # Train model
        predictions[name] = model.predict(X_test_scaled).astype(int)  # Generate predictions
        print(f"{name} training completed!\n")

    return predictions


# Define models
models_dict = {
    "voting_clf_hard": voting_clf_hard,
    "voting_clf_soft": voting_clf_soft,
    "voting_clf_all": voting_clf_all,
    "voting_clf_xgb": voting_clf_xgb,
    "random_forest": best_rf
}

# Run function to train models and get predictions
predictions = train_and_predict(models_dict, X_train_scaled, y_train, X_test_scaled)

# Access individual model predictions
y_hat_vc_hard = predictions["voting_clf_hard"]
y_hat_vc_soft = predictions["voting_clf_soft"]
y_hat_vc_all = predictions["voting_clf_all"]
y_hat_vc_xgb = predictions["voting_clf_xgb"]
y_hat_rf = predictions["random_forest"]

#%%
from sklearn.metrics import accuracy_score

# Check accuracy if y_test is available
print("Voting Classifier (Hard) Accuracy:", accuracy_score(y_train, y_hat_vc_hard))
print("Random Forest Accuracy:", accuracy_score(y_train, y_hat_rf))
print("Voting Classifier (Soft) Accuracy:", accuracy_score(y_train, y_hat_vc_soft))
print("Voting Classifier (All) Accuracy:", accuracy_score(y_train, y_hat_vc_all))
print("Voting Classifier (XGB) Accuracy:", accuracy_score(y_train, y_hat_vc_xgb))

#%%
from sklearn.ensemble import VotingClassifier

def train_voting_classifier(best_clf_lr, best_clf_knn, best_clf_svc, best_clf_rf, best_clf_xgb, X_train_scaled, y_train):

    best_lr = best_clf_lr.best_estimator_
    best_knn = best_clf_knn.best_estimator_
    best_svc = best_clf_svc.best_estimator_
    best_rf = best_clf_rf.best_estimator_
    best_xgb = best_clf_xgb.best_estimator_

    voting_clf = VotingClassifier(
        estimators=[
            ('lr', best_lr),
            ('knn', best_knn),
            ('svc', best_svc),
            ('rf', best_rf),
            ('xgb', best_xgb)
        ],
        voting='soft'
    )

    voting_clf.fit(X_train_scaled, y_train)
    return voting_clf

#%%
import pandas as pd
from sklearn.ensemble import VotingClassifier

def train_voting_classifier(X_train_scaled, y_train, best_clf_lr, best_clf_knn, best_clf_svc, best_clf_rf, best_clf_xgb):

    # Retrieve best estimators
    best_lr = best_clf_lr.best_estimator_
    best_knn = best_clf_knn.best_estimator_
    best_svc = best_clf_svc.best_estimator_
    best_rf = best_clf_rf.best_estimator_
    best_xgb = best_clf_xgb.best_estimator_

    # Define Voting Classifier
    voting_clf = VotingClassifier(
        estimators=[
            ('lr', best_lr),
            ('knn', best_knn),
            ('svc', best_svc),
            ('rf', best_rf),
            ('xgb', best_xgb)
        ],
        voting='soft'
    )

    # Train the classifier
    voting_clf.fit(X_train_scaled, y_train)
    print("Voting Classifier trained successfully!")

    return voting_clf


def predict_and_save(voting_clf, X_test_scaled, test, filename='base_submission.csv'):

    # Make predictions
    y_hat_base_vc = voting_clf.predict(X_test_scaled).astype(int)

    # Print first 10 predictions
    print("First 10 predictions:", y_hat_base_vc[:10])

    # Ensure predictions match test set length before saving
    if len(y_hat_base_vc) == len(test):
        base_submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': y_hat_base_vc})
        base_submission.to_csv(filename, index=False)
        print(f"Submission file '{filename}' saved successfully!")
    else:
        print(f"Error: Prediction length {len(y_hat_base_vc)} does not match test set length {len(test)}.")

    return y

#%%
print(y_hat_base_vc[:10])  # Print first 10 predictions

#%%
