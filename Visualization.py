import pandas as pd
import glob 
import os
# This is the directory path where the CSV files are located
directory_path = "/Users/sophiawang/科研/史料数据集/报刊文章名/1904-1929/embedding regression/all/结果/政治诉求"

# This is the pattern to match all CSV files in the directory
pattern = os.path.join(directory_path, "*.csv")

# This is the directory path where the new CSV files will be saved
output_directory_path = "/Users/sophiawang/科研/史料数据集/报刊文章名/1904-1929/embedding regression/all/结果/政治诉求/绘图结果"

# Iterate over all CSV files in the directory
for file_path in glob.glob(pattern):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Split the DataFrame based on the 'variable' column and save each split as a new CSV file
    for variable, group_df in df.groupby('Variable'):
        # Construct the new file name by replacing '民主' with the value of 'variable'
        base_name = os.path.basename(file_path)
        new_file_name = base_name.replace('民主', variable)
        
        # Construct the new file path
        file_path_new = os.path.join(output_directory_path, new_file_name)
        
        # Save the split DataFrame to a new CSV file
        group_df.to_csv(file_path_new, index=False)

# Since this code involves file system operations, it's provided as a demonstration
# In an actual environment, the code can be run to perform the operations

# Let's print out a statement indicating the operation is designed to complete
print("CSV files are designed to be processed and saved to the output directory.")


import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from collections import OrderedDict
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import resample
import matplotlib.pyplot as plt
import datetime
from datetime import datetime
from numpy.linalg import inv, norm
from scipy.stats import t
import seaborn as sns
import re

CREATE_NEW = True  # 是否为原始嵌入中已有的词创建新嵌入
FLOAT = np.float32
INT = np.uint64
SPACE = ' '
# 绘图设置
import platform
import matplotlib
system = platform.system()  # 获取操作系统类型

if system == 'Windows':
    font = {'family': 'SimHei'}
elif system == 'Darwin':
    font = {'family': 'Arial Unicode MS'}
else:
    # 如果是其他系统，可以使用系统默认字体
    font = {'family': 'sans-serif'}
matplotlib.rc('font', **font)  # 设置全局字体




result=pd.read_csv("/Users/sophiawang/科研/史料数据集/报刊文章名/1904-1929/embedding regression/all/结果/政治诉求/绘图结果/独立~官员+1912-1-1+1912-1-1*官员+1913-6-1+1913-6-1*官员+1915-12-31+1915-12-31*官员+1925-3-12+1925-3-12*官员+1927-2-1+1927-2-1*官员.csv")


interaction_terms = result[result['coefficient'].str.contains('\*')]
variable_name = interaction_terms['Variable'].iloc[0]

# Assuming the dataframe 'df' is already defined and contains the necessary data
df = interaction_terms.copy()  # Create a copy to avoid SettingWithCopyWarning

# Define colors for significant and non-significant points
color_significant = 'navy'  # dark blue
color_nonsignificant = 'lightblue'  # light blue

# Define the significance levels
def get_significance(p):
    if p < 0.05:
        return '*', color_significant
    else:
        return '', color_nonsignificant

def significance_annotation(p):
    if p < 0.01:
        return '***'
    elif p < 0.05:
        return '**'
    elif p < 0.1:
        return '*'
    else:
        return ''
def significance_color(p):
    if p < 0.05:
        return 'navy'
    else:
        return 'lightblue'

# Extract the 'Date' from the 'coefficient' column and verify if the extraction is correct
df['Date'] = df['coefficient'].str.extract(r'(\d{4}-\d{1,2}-\d{1,2})')
assert df['Date'].isnull().sum() == 0, "Some dates could not be extracted properly."

# Apply the function to determine the significance and color
df['Significance'] = df['p_value'].apply(significance_annotation)
df['Color'] = df['p_value'].apply(significance_color)
# Start plotting
fig, ax = plt.subplots(figsize=(12, 6))

# Plot the normed estimates and confidence intervals, and annotate significance
for _, row in df.iterrows():
    yerr = [[row['normed_estimate'] - row['lower_ci']], [row['upper_ci'] - row['normed_estimate']]]
    ax.errorbar(row['Date'], row['normed_estimate'], yerr=yerr, fmt='o', color=row['Color'], ecolor=row['Color'], capsize=5, capthick=2)
    ax.text(row['Date'], row['normed_estimate'], ' ' + row['Significance'], 
            verticalalignment='center', color='black')

# Add custom legend for color significance
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_significant, markersize=10, label='Significant (p < 0.05)'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_nonsignificant, markersize=10, label='Not significant (p ≥ 0.05)')
]
ax.legend(handles=legend_elements, loc='upper right')

# Adjustments for better aesthetics
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Normed Estimate', fontsize=12)
# 使用suptitle为整个图设置标题，并留出空间给discussion_group_label
plt.suptitle('Interaction Terms Estimates with Confidence Intervals', fontsize=16, y=1.05)

# 将discussion_group_label设置为轴标题
discussion_group_label = f'Discussion Group: {variable_name}'
ax.set_title(discussion_group_label, fontsize=14, pad=20)

plt.xticks(rotation=45)
plt.tight_layout(pad=0.5)

# 显示图表
plt.show()

import os
import pandas as pd
import matplotlib.pyplot as plt

def result_pic(filepath):
    result = pd.read_csv(filepath)
    interaction_terms = result[result['coefficient'].str.contains('\*')]
    variable_name = interaction_terms['Variable'].iloc[0]
    # Extract the discussion group from the variable name
    if not interaction_terms.empty:
        # 假设coefficient列的格式为 "term*discussion_group"
        discussion_group = interaction_terms['coefficient'].str.split('*').str[1].iloc[0]
    else:
        discussion_group = 'N/A'

    df = interaction_terms.copy()  # Create a copy to avoid SettingWithCopyWarning

    df['Date'] = df['coefficient'].str.extract(r'(\d{4}-\d{1,2}-\d{1,2})')
    assert df['Date'].isnull().sum() == 0, "Some dates could not be extracted properly."

    def significance_annotation(p):
        if p < 0.01:
            return '***'
        elif p < 0.05:
            return '**'
        elif p < 0.1:
            return '*'
        else:
            return ''

    def significance_color(p):
        if p < 0.05:
            return 'navy'
        else:
            return 'lightblue'

    df['Significance'] = df['p_value'].apply(significance_annotation)
    df['Color'] = df['p_value'].apply(significance_color)

    fig, ax = plt.subplots(figsize=(12, 6))

    for _, row in df.iterrows():
        yerr = [[row['normed_estimate'] - row['lower_ci']], [row['upper_ci'] - row['normed_estimate']]]
        ax.errorbar(row['Date'], row['normed_estimate'], yerr=yerr, fmt='o', color=row['Color'], ecolor=row['Color'], capsize=5, capthick=2)
        ax.text(row['Date'], row['normed_estimate'], ' ' + row['Significance'], verticalalignment='center', color='black')

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='navy', markersize=10, label='Significant (p < 0.05)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=10, label='Not significant (p ≥ 0.05)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Normed Estimate', fontsize=12)
    plt.suptitle('Interaction Terms Estimates with Confidence Intervals', fontsize=16, y=1.05)

    discussion_group_label = f'Discussion Topic: {variable_name} - Discussion Group: {discussion_group}'
    ax.set_title(discussion_group_label, fontsize=14, pad=20)

    plt.xticks(rotation=45)
    plt.tight_layout(pad=0.5)  # Adjust padding for tight layout

    return fig, ax  # Return the figure and axis objects for further manipulation if needed

input_folder_path = '/Users/sophiawang/科研/史料数据集/报刊文章名/1904-1929/embedding regression/all/结果/政治诉求/绘图结果/'
output_folder_path = '/Users/sophiawang/科研/史料数据集/报刊文章名/1904-1929/embedding regression/all/结果/政治诉求/绘图结果/绘图/'

for filename in os.listdir(input_folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(input_folder_path, filename)
        fig, ax = result_pic(file_path)  # Call the plotting function
        output_file_path = os.path.join(output_folder_path, filename.replace('.csv', '.png'))
        fig.savefig(output_file_path,dpi=600)  # Save the figure using the figure object
        plt.close(fig)  # Close the figure after saving

print("所有图表已生成并保存。")
