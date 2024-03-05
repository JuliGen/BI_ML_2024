import os
import warnings

import pandas as pd
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

def run_eda(df: pd.DataFrame) -> None:
    
    """
    Create EDA
    
    params:
    df - Dataframe for conducting EDA
    """

    # Greeting
    print(f'\033[1m Hello, I\'m an assistant, my name is Rex \U0001F996. Today I will be your guide to the world of your dataframe.\033[0m\n')

    # Counting rows and columns
    space = ' '
    number_column, number_row = len(df.columns), len(df)
    print(f'\033[1;32m1) Number of columns:\033[0;0m {number_column},\n{space * 2}\033[1;32m Number of row:\033[0;0m {number_row}\n')

    # Identify dtype
    column_list_num = []
    column_list_category = []
    column_list_str = []

    for column in df.columns:
        if df[column].dtype == float:
            if df[column].nunique() >= 12:
                column_list_num.append(column)
            else:
                column_list_category.append(column)
        elif df[column].dtype == int:
            if df[column].nunique() >= 12:
                column_list_num.append(column)
            else:
                column_list_category.append(column)
        elif df[column].dtype == str:
            if df[column].nunique() >= 12:
                column_list_str.append(column)
            else:
                column_list_category(column)
        else:
            if df[column].nunique() >= 13:
                column_list_str.append(column)
            else:
                column_list_category.append(column)

    print(f'\033[1;31m2) Numerical columns:\033[0;0m {column_list_num},')
    print(f"{space * 2}\033[1;31m String columns:\033[0;0m {column_list_str},")
    print(f"{space * 2}\033[1;31m Categorical columns:\033[0;0m  {column_list_category}\n")

    # Counts and frequences for categorical column
    freq = []
    data_for_table = []
    print(f'\033[1;36m3) Number of values and their frequencies:\033[1;0m\n')

    for column in column_list_category:
        value_list = df[column].value_counts().index
        value_count_list = df[column].value_counts().values
        freq_list = list(map(lambda x: x / value_count_list.sum(), value_count_list))

        for value, value_count, freq in zip(value_list, value_count_list, freq_list):
            data_for_table.append([value, value_count, round(freq, 3)])
        print(f"\033[1;36m{column}\033[0;0m")
        print(tabulate(data_for_table, headers=['Name', 'Count', 'Frequences'], tablefmt='rounded_outline') + '\n')
        data_for_table.clear()

    # Calculate basic statistic for numeric column
    print(f'\033[1;35m4) Basic statistics for numerical columns:\033[1;0m\n')

    for column in column_list_num:
        max_ = df[column].max()
        min_ = df[column].min()
        mean_ = df[column].mean()
        std_ = df[column].std()
        q0_25 = df[column].quantile(q=0.25)
        median_ = df[column].median()
        q0_75 = df[column].quantile(q=0.75)

        data_for_table.append([max_, min_, mean_, std_, q0_25, median_, q0_75])
        print(f"\033[1;35m{column}\033[0;0m")
        print(tabulate(data_for_table, headers=['Max', 'Min', 'Mean', 'Std', 'Q_0.25', 'Median', 'Q_0.75'],
                       tablefmt='rounded_outline') + '\n')
        data_for_table.clear()
        
    # Finding outliers
    count_outliers = []
    print(f"\033[1;32m5) Number of outliers:\033[0m\n")

    for column in column_list_num[:5]:
        q0_25 = df[column].quantile(q=0.25)
        q0_75 = df[column].quantile(q=0.75)
        IQR = q0_75 - q0_25
        upper_limit = q0_75 + 1.5 * IQR
        lower_limit = q0_25 - 1.5 * IQR
        count_outliers.append(
            df[column][df[column] <= lower_limit].size + df[column][df[column] >= upper_limit].size)

    print(tabulate([count_outliers], headers=column_list_num[:5], tablefmt='rounded_outline') + '\n')

    count_outliers = []
    for column in column_list_num[5:]:
        q0_25 = df[column].quantile(q=0.25)
        q0_75 = df[column].quantile(q=0.75)
        IQR = q0_75 - q0_25
        upper_limit = q0_75 + 1.5 * IQR
        lower_limit = q0_25 - 1.5 * IQR
        count_outliers.append(
            df[column][df[column] <= lower_limit].size + df[column][df[column] >= upper_limit].size)

    print(tabulate([count_outliers], headers=column_list_num[5:], tablefmt='rounded_outline') + '\n')

    # Find NA
    na_in_columns = df.isna().any()
    print(f"\033[1;34m6) Number of NA:\033[0m\n")

    print(f"\033[1m Total:\033[0m {df.isna().sum().sum()}")
    print(f"\033[1m Number of rows with NA:\033[0m {df.isna().any(axis=1).sum()}")
    print(f"\033[1m Columns containing NA:\033[0m {list(na_in_columns[na_in_columns].index)}\n")

    # Plot for NA
    sns.displot(data=df.isna().melt(value_name="Missing values (Na)", var_name='Column'), y="Column",
                hue="Missing values (Na)", multiple="fill", palette='Set2')
    plt.title("Percentage of missing values (Na)", weight="bold")
    plt.xlabel("Proportion")
    plt.ylabel("Column")

    # Duplicate rows
    print(f"\033[1;36m7) Number of duplicate rows:\033[0m {df.duplicated().sum()}\n")


