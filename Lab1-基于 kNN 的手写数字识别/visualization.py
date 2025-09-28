# -*- coding: utf-8 -*-
"""
kNN分类快速可视化脚本
直接生成综合的九宫格热力图
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import pandas as pd
from collections import Counter
import time

# 导入主实验模块的函数
from exp_1 import euclidean_distance, knn_predict, load_arff_data

# 设置中文字体支持
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 9
plt.rcParams['font.family'] = 'sans-serif'
sns.set_style("whitegrid")

def generate_results_with_loo(X, y, k_values=[1, 3, 5], sample_size=300):
    """
    使用exp_1.py中的留一法交叉验证逻辑生成结果
    """
    print("使用留一法交叉验证生成kNN分类结果...")
    all_results = {}
    
    # 如果样本数太多，进行采样以提高速度
    if len(X) > sample_size:
        print(f"采样 {sample_size} 个样本进行快速分析...")
        # 确保每个类别都有代表性
        indices_per_class = []
        samples_per_class = sample_size // 10  # 每类相同数量
        
        for digit in range(10):
            digit_indices = np.where(y == digit)[0]
            if len(digit_indices) >= samples_per_class:
                selected = np.random.choice(digit_indices, samples_per_class, replace=False)
                indices_per_class.extend(selected)
        
        # 如果还不够，随机补充
        if len(indices_per_class) < sample_size:
            remaining = sample_size - len(indices_per_class)
            all_indices = set(range(len(X)))
            available = list(all_indices - set(indices_per_class))
            if len(available) >= remaining:
                additional = np.random.choice(available, remaining, replace=False)
                indices_per_class.extend(additional)
        
        # 使用采样数据
        indices = indices_per_class[:sample_size]
        X_sample = X[indices]
        y_sample = y[indices]
    else:
        X_sample = X
        y_sample = y
    
    for k in k_values:
        print(f"\n处理 k={k}...")
        
        # 使用exp_1.py中的loo_eval逻辑
        from exp_1 import loo_eval
        accuracy = loo_eval(X_sample, y_sample, k)
        
        # 为了生成混淆矩阵，我们需要预测结果
        print(f"生成预测结果用于可视化...")
        y_true = []
        y_pred = []
        
        n_samples = len(X_sample)
        for i in range(n_samples):
            if (i + 1) % 50 == 0:
                print(f"  生成预测: {i+1}/{n_samples}")
            
            # 留出第i个样本作为测试
            X_test = X_sample[i]
            y_test = y_sample[i]
            
            # 其余样本作为训练集
            X_train = np.vstack([X_sample[:i], X_sample[i+1:]])
            y_train = np.hstack([y_sample[:i], y_sample[i+1:]])
            
            # 预测
            prediction = knn_predict(X_train, y_train, X_test, k)
            
            y_true.append(y_test)
            y_pred.append(prediction)
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # 验证准确率一致性
        calc_accuracy = np.sum(y_true == y_pred) / len(y_true)
        print(f"  验证: loo_eval准确率={accuracy:.4f}, 计算准确率={calc_accuracy:.4f}")
        
        # 计算各类别准确率
        class_accuracies = []
        for digit in range(10):
            mask = (y_true == digit)
            if np.sum(mask) > 0:
                class_acc = np.sum((y_true == y_pred) & mask) / np.sum(mask)
                class_accuracies.append(class_acc)
            else:
                class_accuracies.append(0)
        
        # 计算详细指标
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0)
        
        performance = {
            'accuracy': accuracy,  # 使用loo_eval的结果
            'class_accuracies': np.array(class_accuracies),
            'avg_precision': np.mean(precision),
            'avg_recall': np.mean(recall),
            'f1_score': np.mean(f1)
        }
        
        all_results[k] = {
            'predictions': (y_true, y_pred),
            'performance': performance,
            'sample_data': (X_sample, y_sample)  # 保存采样的数据
        }
        
        print(f"  k={k} 完成，准确率: {accuracy:.4f}")
    
    return all_results

def create_comprehensive_visualization(all_results, save_path='knn_comprehensive_analysis.png'):
    """
    创建综合的九宫格可视化
    """
    # 创建大图
    fig = plt.figure(figsize=(18, 14))
    
    # 调整布局参数
    plt.subplots_adjust(hspace=0.35, wspace=0.25, top=0.94, bottom=0.06, left=0.06, right=0.96)
    
    k_values = sorted(all_results.keys())
    
    # 第一行：混淆矩阵 (3个)
    for i, k in enumerate(k_values):
        ax = plt.subplot(3, 3, i+1)
        y_true, y_pred = all_results[k]['predictions']
        cm = confusion_matrix(y_true, y_pred, labels=range(10))
        
        # 绘制混淆矩阵
        im = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=range(10), yticklabels=range(10),
                        cbar_kws={'shrink': 0.7}, square=True)
        
        accuracy = all_results[k]['performance']['accuracy']
        ax.set_title(f'k={k} Confusion Matrix\nAccuracy: {accuracy:.3f}', 
                    fontsize=11, fontweight='bold', pad=12)
        ax.set_xlabel('Predicted Label', fontsize=9)
        ax.set_ylabel('True Label', fontsize=9)
        ax.tick_params(labelsize=8)
    
    # 第二行：各类别准确率 (3个)
    for i, k in enumerate(k_values):
        ax = plt.subplot(3, 3, i+4)
        class_accuracies = all_results[k]['performance']['class_accuracies']
        
        # 重塑为2x5网格
        acc_matrix = class_accuracies.reshape(2, 5)
        labels = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        
        # 绘制热力图
        sns.heatmap(acc_matrix, annot=labels, fmt='d', cmap='RdYlGn', 
                    vmin=0, vmax=1, ax=ax, cbar_kws={'shrink': 0.7},
                    square=True)
        
        # 添加准确率数值
        for row in range(2):
            for col in range(5):
                if acc_matrix[row, col] > 0:  # 只显示有数据的
                    ax.text(col+0.5, row+0.8, f'{acc_matrix[row,col]:.2f}', 
                           ha='center', va='center', fontsize=8, 
                           color='white', fontweight='bold')
        
        ax.set_title(f'k={k} Class Accuracy', fontsize=11, fontweight='bold', pad=12)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # 第三行左侧：性能对比 (占2个位置)
    ax_perf = plt.subplot(3, 3, (7, 8))
    
    # 准备性能对比数据
    metrics = ['Accuracy', 'Recall', 'Precision', 'F1-Score']
    data_matrix = []
    for k in k_values:
        perf = all_results[k]['performance']
        data_matrix.append([
            perf['accuracy'],
            perf['avg_recall'],
            perf['avg_precision'],
            perf['f1_score']
        ])
    
    data_matrix = np.array(data_matrix)
    df_perf = pd.DataFrame(data_matrix, 
                          index=[f'k={k}' for k in k_values],
                          columns=metrics)
    
    sns.heatmap(df_perf, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax_perf,
                cbar_kws={'shrink': 0.7})
    ax_perf.set_title('k-Value Performance Comparison', fontsize=11, fontweight='bold', pad=12)
    ax_perf.set_xlabel('Performance Metrics', fontsize=9)
    ax_perf.set_ylabel('k-Value', fontsize=9)
    ax_perf.tick_params(labelsize=8)
    
    # 第三行右侧：实验总结
    ax_summary = plt.subplot(3, 3, 9)
    ax_summary.axis('off')
    
    # 找出最佳k值
    best_k = max(k_values, key=lambda k: all_results[k]['performance']['accuracy'])
    best_acc = all_results[best_k]['performance']['accuracy']
    
    # 创建总结文本
    summary_text = "Experiment Summary\n" + "="*18 + "\n\n"
    summary_text += f"Best k-value: {best_k}\n"
    summary_text += f"Highest Accuracy: {best_acc:.4f}\n\n"
    summary_text += "k-value Accuracies:\n"
    
    for k in k_values:
        acc = all_results[k]['performance']['accuracy']
        summary_text += f"  k={k}: {acc:.4f}\n"
    
    sample_size = len(all_results[k_values[0]]['predictions'][0])
    summary_text += f"\nExperiment Setup:\n"
    summary_text += f"  Samples: {sample_size}\n"
    summary_text += f"  Features: 256\n"
    summary_text += f"  Classes: 10\n"
    summary_text += f"  Validation: LOO\n"
    
    # 添加建议
    if best_k == 1:
        suggestion = "Suggestion: k=1 performs\nbest, good data quality"
    elif best_k == max(k_values):
        suggestion = f"Suggestion: Try larger\nk values"
    else:
        suggestion = f"Suggestion: k={best_k} balances\nbias and variance"
    
    summary_text += f"\n{suggestion}"
    
    ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes,
                   fontsize=9, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.4", facecolor="lightblue", alpha=0.8))
    

    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"综合分析图已保存到: {save_path}")
    
    plt.show()
    return fig

def visualize_misclassified_samples(X, y, all_results, save_dir='misclassified_samples'):
    """
    可视化并保存错误分类的样本
    """
    import os
    
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"创建目录: {save_dir}")
    
    for k in sorted(all_results.keys()):
        print(f"\n分析k={k}的错误分类样本...")
        
        y_true, y_pred = all_results[k]['predictions']
        
        # 找出所有错误分类的样本
        misclassified_indices = []
        misclassified_info = []
        
        for i in range(len(y_true)):
            if y_true[i] != y_pred[i]:
                misclassified_indices.append(i)
                misclassified_info.append({
                    'index': i,
                    'true_label': y_true[i],
                    'pred_label': y_pred[i],
                    'image': X[i]
                })
        
        print(f"k={k}: 发现 {len(misclassified_indices)} 个错误分类样本")
        
        if len(misclassified_indices) == 0:
            print(f"k={k}: 没有错误分类样本！")
            continue
        
        # 按每页16个样本进行分组
        samples_per_page = 16
        num_pages = (len(misclassified_info) + samples_per_page - 1) // samples_per_page
        
        for page in range(num_pages):
            start_idx = page * samples_per_page
            end_idx = min(start_idx + samples_per_page, len(misclassified_info))
            page_samples = misclassified_info[start_idx:end_idx]
            
            # 创建4x4网格
            fig, axes = plt.subplots(4, 4, figsize=(12, 12))
            axes = axes.flatten()
            
            for i, sample in enumerate(page_samples):
                # 显示图像
                axes[i].imshow(sample['image'].reshape(16, 16), cmap='gray')
                
                # 设置标题 - 红色表示错误
                title = f"True: {sample['true_label']}\nPred: {sample['pred_label']}"
                axes[i].set_title(title, color='red', fontsize=10, fontweight='bold')
                axes[i].axis('off')
            
            # 隐藏多余的子图
            for i in range(len(page_samples), 16):
                axes[i].axis('off')
            
            # 设置总标题
            total_errors = len(misclassified_info)
            page_title = f'k={k} Misclassified Samples (Page {page+1}/{num_pages})\n'
            page_title += f'Showing {len(page_samples)} of {total_errors} total errors'
            fig.suptitle(page_title, fontsize=14, fontweight='bold', y=0.98)
            
            # 调整布局
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            # 保存图片
            if num_pages == 1:
                filename = f'{save_dir}/k{k}_misclassified_samples.png'
            else:
                filename = f'{save_dir}/k{k}_misclassified_samples_page{page+1}.png'
            
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"保存: {filename}")
            
            plt.show()
        
        # 生成错误分类统计报告
        error_stats = {}
        for sample in misclassified_info:
            true_label = sample['true_label']
            pred_label = sample['pred_label']
            key = f"{true_label}→{pred_label}"
            error_stats[key] = error_stats.get(key, 0) + 1
        
        # 保存统计报告
        report_filename = f'{save_dir}/k{k}_error_statistics.txt'
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(f"k={k} 错误分类统计报告\n")
            f.write("="*40 + "\n\n")
            f.write(f"总错误数量: {len(misclassified_info)}\n")
            f.write(f"总样本数量: {len(y_true)}\n")
            f.write(f"错误率: {len(misclassified_info)/len(y_true)*100:.2f}%\n\n")
            f.write("具体错误类型统计:\n")
            f.write("-"*30 + "\n")
            
            # 按错误数量排序
            sorted_errors = sorted(error_stats.items(), key=lambda x: x[1], reverse=True)
            for error_type, count in sorted_errors:
                f.write(f"{error_type}: {count} 次\n")
        
        print(f"保存统计报告: {report_filename}")

def analyze_error_patterns(all_results, save_dir='misclassified_samples'):
    """
    分析错误模式并生成可视化
    """
    import os
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 创建错误模式对比图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, k in enumerate(sorted(all_results.keys())):
        y_true, y_pred = all_results[k]['predictions']
        
        # 创建错误矩阵（只显示错误的部分）
        error_matrix = np.zeros((10, 10))
        
        for true_label, pred_label in zip(y_true, y_pred):
            if true_label != pred_label:
                error_matrix[true_label, pred_label] += 1
        
        # 绘制热力图
        im = sns.heatmap(error_matrix, annot=True, fmt='g', cmap='Reds', ax=axes[i],
                        xticklabels=range(10), yticklabels=range(10),
                        cbar_kws={'shrink': 0.8})
        
        axes[i].set_title(f'k={k} Error Pattern\n(True → Predicted)', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Predicted Label')
        axes[i].set_ylabel('True Label')
    
    plt.tight_layout()
    
    # 保存错误模式图
    error_pattern_file = f'{save_dir}/error_patterns_comparison.png'
    plt.savefig(error_pattern_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"保存错误模式对比图: {error_pattern_file}")
    
    plt.show()

def main():
    """
    主函数
    """
    print("=" * 50)
    print("kNN Quick Visualization Analysis")
    print("=" * 50)
    
    # 加载数据
    print("加载数据...")
    try:
        X, y = load_arff_data('semeion_tenclass.arff')
        print(f"数据加载成功！总样本数: {len(X)}")
    except Exception as e:
        print(f"数据加载失败: {e}")
        print("请确保semeion_tenclass.arff文件在当前目录中")
        return
    
    # 设置参数
    k_values = [1, 3, 5]
    sample_size = 1593  # 增加样本数以获得更准确的结果
    
    print(f"实验参数: k值={k_values}, 最大样本数={sample_size}")
    
    # 生成结果
    start_time = time.time()
    all_results = generate_results_with_loo(X, y, k_values, sample_size)
    end_time = time.time()
    
    print(f"\n计算完成，用时: {end_time - start_time:.1f}秒")
    
    # 创建可视化
    print("生成综合可视化...")
    create_comprehensive_visualization(all_results, 'knn_comprehensive_analysis.png')
    
    # 分析错误分类样本
    print("\n" + "=" * 30)
    print("分析错误分类样本...")
    print("=" * 30)
    
    # 可视化错误分类的样本 - 使用保存的采样数据
    # 从第一个k值结果中获取采样数据（所有k值使用相同的采样）
    first_k = sorted(all_results.keys())[0]
    X_sample, y_sample = all_results[first_k]['sample_data']
    
    visualize_misclassified_samples(X_sample, y_sample, all_results, 'misclassified_samples')
    
    # 分析错误模式
    analyze_error_patterns(all_results, 'misclassified_samples')
    
    print("\n" + "=" * 50)
    print("完整分析完成！")
    print("生成的文件:")
    print("  - knn_comprehensive_analysis.png (综合分析)")
    print("  - misclassified_samples/ (错误分类分析目录)")
    print("    - k1_misclassified_samples.png")
    print("    - k3_misclassified_samples.png") 
    print("    - k5_misclassified_samples.png")
    print("    - k*_error_statistics.txt (错误统计报告)")
    print("    - error_patterns_comparison.png (错误模式对比)")
    print("=" * 50)

if __name__ == "__main__":
    main()
