# -*- coding: utf-8 -*-
import numpy as np
from collections import Counter
import time

def euclidean_distance(x1, x2):
    #计算欧氏距离：√(∑(xi - xj)²)
    return np.sqrt(np.sum((x1 - x2) ** 2))

def knn_predict(X_train, y_train, x_test, k):
    # k近邻分类器预测

    # 计算测试样本与所有训练样本的距离
    distances = []
    for i in range(len(X_train)):
        dist = euclidean_distance(x_test, X_train[i])
        distances.append((dist, y_train[i]))
    
    # 按距离排序并选择k个最近邻
    distances.sort(key=lambda x: x[0])
    k_nearest = distances[:k]
    
    # 提取k个最近邻的标签
    k_labels = [label for _, label in k_nearest]
    
    # 多数投票决定预测结果
    vote_result = Counter(k_labels)
    return vote_result.most_common(1)[0][0]

def loo_eval(X, y, k):
    #留一法交叉验证

    n_samples = len(X)
    correct_count = 0
    
    print(f"开始k={k}的留一法交叉验证，共{n_samples}轮...")
    
    for i in range(n_samples):
        # 显示进度
        if (i + 1) % 100 == 0:
            print(f"进度: {i+1}/{n_samples}")
        
        # 留出第i个样本作为测试
        X_test = X[i]
        y_test = y[i]
        
        # 其余样本作为训练集
        X_train = np.vstack([X[:i], X[i+1:]])
        y_train = np.hstack([y[:i], y[i+1:]])
        
        # 预测
        y_pred = knn_predict(X_train, y_train, X_test, k)
        
        # 统计正确预测
        if y_pred == y_test:
            correct_count += 1
    
    accuracy = correct_count / n_samples
    return accuracy

def load_arff_data(filename):
    #加载ARFF格式数据文件 

    print(f"加载数据文件: {filename}")
    
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 找到数据开始行
    data_start = -1
    for i, line in enumerate(lines):
        if line.strip() == '@DATA':
            data_start = i + 1
            break
    
    if data_start == -1:
        raise ValueError("未找到@DATA标记")
    
    # 读取数据
    data_lines = lines[data_start:]
    data = []
    
    for line in data_lines:
        line = line.strip()
        if line and not line.startswith('%'):  # 忽略空行和注释行
            values = [float(x) for x in line.split(',')]
            data.append(values)
    
    data_array = np.array(data)
    
    # 分离特征和标签
    X = data_array[:, :-1]  # 前256列是特征
    y = data_array[:, -1].astype(int)  # 最后一列是标签
    
    return X, y

def main():
    #主函数：执行完整的kNN手写数字识别实验

    print("=" * 50)
    print("kNN手写数字识别实验")
    print("=" * 50)
    
    # 1. 加载数据
    try:
        X, y = load_arff_data('semeion_tenclass.arff')
        print(f"数据加载成功！")
        print(f"样本数: {len(X)}, 像素数: {X.shape[1]}")
        print(f"标签范围: {np.min(y)} - {np.max(y)}")
    except Exception as e:
        print(f"数据加载失败: {e}")
        return
    
    # 2. 执行留一法交叉验证
    print("\n开始留一法交叉验证...")
    
    k_values = [1, 3, 5]
    results = []
    
    for k in k_values:
        print(f"\n--- 测试 k={k} ---")
        start_time = time.time()
        
        acc = loo_eval(X, y, k)
        
        end_time = time.time()
        print(f"k={k} LOO 准确率 = {acc:.4f}")
        print(f"用时: {end_time - start_time:.1f}秒")
        
        results.append(acc)
    
    # 3. 输出最终结果
    print("\n" + "=" * 50)
    print("实验结果汇总")
    print("=" * 50)
    for i, k in enumerate(k_values):
        print(f"k={k} LOO 准确率 = {results[i]:.4f}")

if __name__ == "__main__":
    main()