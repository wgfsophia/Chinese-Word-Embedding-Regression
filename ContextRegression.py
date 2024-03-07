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

def load_vectors(vectorfile):
    words = set()
    with open(vectorfile, 'r') as f:
        for line in f:
            parts = line.strip().split(SPACE)
            word = parts[0]
            vector_parts = parts[1:]

            # 检查是否有足够的部分构成一个向量
            if len(vector_parts) != 300:
                print(f"Warning: Skipping word '{word}' due to incorrect number of vector parts: {len(vector_parts)}")
                continue

            try:
                vector = np.array([float(p) for p in vector_parts], dtype=FLOAT)
            except ValueError as e:
                print(f"Error converting vector parts to floats for word '{word}': {e}")
                continue

            if word not in words:
                words.add(word)
                yield word, vector

def create_context_corpus_with_date(df, target_word, window):
    context_corpus = []
    corpus_with_date = []

    for index, row in df.iterrows():
        words = row['Tokenized_Title'].split()
        time_stamp = row['Date']

        for i in range(len(words)):
            if words[i] == target_word:
                start = max(0, i - window)
                end = min(len(words), i + window + 1)
                context = ' '.join(words[start:end])
                context_corpus.append(context)
                corpus_with_date.append((context, time_stamp))
            
 
    return context_corpus, corpus_with_date









def extract_and_prepare_covariates(formula, corpus_with_date):
    # 分割公式以获取因变量和独立变量
    dependent_var, independent_vars_str = formula.split('~')
    dependent_var = dependent_var.strip()
    independent_vars = independent_vars_str.strip().split('+')

    # 从corpus_with_date中提取文本数据和时间数据
    texts = [item[0] for item in corpus_with_date]
    time_data = [item[1] for item in corpus_with_date]

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    feature_matrix = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

    # 创建一个空的DataFrame用于存储协变量
    cov_vars = pd.DataFrame(index=feature_matrix.index)

    # 处理每个独立变量，包括交互项
    for var in independent_vars:
        if '*' in var:  # 检查并处理交互项
            factors = var.split('*')
            interaction_term = pd.Series([1]*len(texts), index=feature_matrix.index)  # 初始化为全1
            for factor in factors:
                if re.match(r'\d{4}-\d{1,2}-\d{1,2}', factor):  # 日期格式
                    cutoff_datetime = datetime.strptime(factor, "%Y-%m-%d")
                    temp_series = pd.Series([1 if datetime.strptime(time, "%Y-%m-%d") > cutoff_datetime else 0 for time in time_data], index=feature_matrix.index)
                elif factor in feature_matrix.columns:  # 文本特征
                    temp_series = feature_matrix[factor]
                else:
                    raise ValueError(f"The covariate '{factor}' could not be found in the data.")
                interaction_term *= temp_series  # 生成交互项
            cov_vars[var] = interaction_term
        elif var == '.':  # 全部变量
            cov_vars = feature_matrix.copy()
            break
        elif re.match(r'\d{4}-\d{1,2}-\d{1,2}', var):  # 日期格式
            cutoff_datetime = datetime.strptime(var, "%Y-%m-%d")
            cov_vars[var] = [1 if datetime.strptime(time, "%Y-%m-%d") > cutoff_datetime else 0 for time in time_data]
        elif var in feature_matrix.columns:  # 文本特征
            cov_vars[var] = feature_matrix[var]
        else:
            raise ValueError(f"The covariate '{var}' could not be found in the data.")

    cov_vars_with_intercept = cov_vars.copy()
    cov_vars_with_intercept['Intercept'] = 1

    return cov_vars_with_intercept



def dem(x, w2v, transform=True, transform_matrix=None, verbose=True):
    pre_trained = np.array(list(w2v.values()))

    if transform and transform_matrix is not None:
        if pre_trained.shape[1] != transform_matrix.shape[0]:
            raise ValueError("Dimensions of pre-trained embeddings and transform matrix do not match.")

    dfm_features = x.shape[1]
    pre_trained_features = pre_trained.shape[0]
    overlapping_indices = [i for i in range(dfm_features) if i < pre_trained_features]
    if len(overlapping_indices) == 0:
        raise ValueError("No overlapping features with the pre-trained embeddings.")

    pre_trained_filtered = pre_trained[overlapping_indices, :]
    context_embeddings = x.dot(pre_trained_filtered)

    N = np.array(x.sum(axis=1)).flatten()
    N[N == 0] = 1  
    context_embeddings = context_embeddings / N[:, np.newaxis]

    if transform and transform_matrix is not None:
        context_embeddings = np.dot(context_embeddings, transform_matrix)

    if verbose:
        print("Total observations included in regression:", context_embeddings.shape[0])
    observation=context_embeddings.shape[0]

    return context_embeddings,observation

def run_jack_ols(X, Y, confidence_level=0.95):
    n, p = X.shape
    model = LinearRegression()
    model.fit(X, Y)
    beta = model.coef_.T  # 转置以匹配维度

    # 初始化用于存储自举系数范数的数组
    bootstrap_norms = np.zeros((1000, p))

    # 自举重采样
    for i in range(1000):
        X_resampled, Y_resampled = resample(X, Y)
        model.fit(X_resampled, Y_resampled)
        bootstrap_coeffs = model.coef_.T
        for j in range(p):
            bootstrap_norms[i, j] = norm(bootstrap_coeffs[j, :])

    # 计算每个自变量的欧几里得范数、标准误差和置信区间
    original_norms = np.array([norm(beta[j, :]) for j in range(p)])
    stderrs = np.std(bootstrap_norms, axis=0)
    ci_multipliers = t.ppf((1 + confidence_level) / 2, n - 1)
    lower_cis = original_norms - ci_multipliers * stderrs
    upper_cis = original_norms + ci_multipliers * stderrs
    # 计算 t 值和 p 值
    t_values = original_norms / stderrs
    p_values = [2 * t.sf(np.abs(t_val), df=n - 1) for t_val in t_values]  # 双尾检验

    # 创建 DataFrame
    result_df = pd.DataFrame({
        'coefficient': X.columns,
        'normed_estimate': original_norms,
        'std_error': stderrs,
        'lower_ci': lower_cis,
        'upper_ci': upper_cis,
        't_value': t_values,
        'p_value': p_values
    })

    return result_df


def analysis_for_dependent_vars(dependent_vars, corpus, w2v, transform_matrix, formula_template):
    """
    运行对多个因变量的分析，每个因变量使用相同的formula模板。
    
    Parameters:
    - dependent_vars: 因变量名称的列表。
    - corpus: 分析用的语料库。
    - w2v: word2vec模型实例。
    - transform_matrix: 转换矩阵。
    - formula_template: 公式模板，其中应包含一个用于替换的{dep_var}标记。
    
    Returns:
    - final_results_df: 包含所有结果的DataFrame。
    """
    final_results_df = pd.DataFrame()
    results = []  # 确保这个列表在循环开始之前是空的

    for dep_var_name in dependent_vars:
        try:
            context_corpus, corpus_with_date = create_context_corpus_with_date(corpus, dep_var_name, 10)
            vectorizer = CountVectorizer()
            target_matrix = vectorizer.fit_transform(context_corpus)
            dep_var,observation = dem(target_matrix, w2v, transform=True, transform_matrix=transform_matrix, verbose=True)
            
            # 使用公式模板并替换依赖变量名称
            formula = formula_template.format(dep_var=dep_var_name)
            
            cov_vars = extract_and_prepare_covariates(formula, corpus_with_date)

            # 调用 run_jack_ols 函数
            result_df = run_jack_ols(cov_vars, dep_var)

            # 添加因变量名到结果 DataFrame
            result_df['Variable'] = dep_var_name
            result_df['Observations']=observation

            # 将结果 DataFrame 添加到总结果列表中
            results.append(result_df)
        except ValueError as e:
            print(f"Warning: {e} - Skipping {dep_var_name}")
            continue  # 跳过当前的迭代

    # 合并所有结果 DataFrame
    final_results_df = pd.concat(results, ignore_index=True)

    return final_results_df

#使用实例
#输入公式格式：formula='target_word=independent_var1(string)+independent_var2(string)|cutoff_time1,cutoff_time2'
if __name__ == "__main__":
    GLOVEFILE = 'sgns.merge.bigram.txt'  
    w2v = OrderedDict(load_vectors(GLOVEFILE))

    matrix_file='all_matrix' #使用alcembedding中训练好的变换矩阵
    transform_matrix= np.fromfile(matrix_file, dtype=np.float32) 
    transform_matrix= transform_matrix.reshape((300, 300))

    corpus = pd.read_csv('/Users/sophiawang/科研/史料数据集/报刊文章名/1904-1929/embedding regression/all/all.csv')  
    
    window = 10
    tokenized_corpus = corpus[corpus['Tokenized_Title'].notna() & (corpus['Tokenized_Title'] != '')]
    corpus=tokenized_corpus
    # 调用函数并接收返回值
    
    #使用实例
    #谁在呼吁民主
    dependent_vars=['民主','独立','自治','共和','立宪','自由','平等','选举'] 
    #权力的党化
    dependent_vars=['国民党','党派','政党'] 
    #公民的组成
    dependent_vars=['市民'] 
    formula_templates=['{dep_var}~学生+1912-1-1+1912-1-1*学生+1913-6-1+1913-6-1*学生+1915-12-31+1915-12-31*学生+1925-3-12+1925-3-12*学生+1927-2-1+1927-2-1*学生',
                       '{dep_var}~官员+1912-1-1+1912-1-1*官员+1913-6-1+1913-6-1*官员+1915-12-31+1915-12-31*官员+1925-3-12+1925-3-12*官员+1927-2-1+1927-2-1*官员',
                       '{dep_var}~士绅+1912-1-1+1912-1-1*士绅+1913-6-1+1913-6-1*士绅+1915-12-31+1915-12-31*士绅+1925-3-12+1925-3-12*士绅+1927-2-1+1927-2-1*士绅',
                       '{dep_var}~军队+1912-1-1+1912-1-1*军队+1913-6-1+1913-6-1*军队+1915-12-31+1915-12-31*军队+1925-3-12+1925-3-12*军队+1927-2-1+1927-2-1*军队',
                       '{dep_var}~工人+1912-1-1+1912-1-1*工人+1913-6-1+1913-6-1*工人+1915-12-31+1915-12-31*工人+1925-3-12+1925-3-12*工人+1927-2-1+1927-2-1*工人',
                       '{dep_var}~商人+1912-1-1+1912-1-1*商人+1913-6-1+1913-6-1*商人+1915-12-31+1915-12-31*商人+1925-3-12+1925-3-12*商人+1927-2-1+1927-2-1*商人',
                       '{dep_var}~女+1912-1-1+1912-1-1*女+1913-6-1+1913-6-1*女+1915-12-31+1915-12-31*女+1925-3-12+1925-3-12*女+1927-2-1+1927-2-1*女',
                       '{dep_var}~农民+1912-1-1+1912-1-1*农民+1913-6-1+1913-6-1*农民+1915-12-31+1915-12-31*农民+1925-3-12+1925-3-12*商人+1927-2-1+1927-2-1*商人',]
    
    for formula_template in formula_templates:
        try:
            final_results_df =analysis_for_dependent_vars(dependent_vars, corpus, w2v, transform_matrix, formula_template)
            formula_without_dep_var = formula_template.replace('{dep_var}~', '')
            savetitle=f'{dependent_vars[0]}~{formula_without_dep_var}'
            final_results_df.to_csv(f'{savetitle}.csv')
            print(final_results_df)
        except Exception as e:
            print(f'处理回归 "{formula_template}" 时发生错误：{e}。跳过。')
