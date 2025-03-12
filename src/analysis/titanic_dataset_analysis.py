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