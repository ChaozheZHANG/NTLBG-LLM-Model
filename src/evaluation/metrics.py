"""
NTLBG-LLM评估指标模块
包含各种视频问答任务的评估指标
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict
import json
import re
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from rouge_score import rouge_scorer
from bert_score import BERTScorer
import sacrebleu
from torchmetrics.text import BLEUScore
from torchmetrics.functional import bleu_score as torch_bleu_score


class VideoQAMetrics:
    """视频问答评估指标"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化各种评估器
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.bert_scorer = BERTScorer(model_type='bert-base-uncased', device=self.device)
        self.bleu_scorer = BLEUScore()
        
        # 初始化NLTK组件
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet', quiet=True)
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # 累积统计
        self.reset_metrics()
    
    def reset_metrics(self):
        """重置所有累积指标"""
        self.predictions = []
        self.references = []
        self.question_types = []
        self.video_ids = []
        self.qa_pairs = []
        
        # 分类任务指标
        self.correct_predictions = 0
        self.total_predictions = 0
        
        # 文本生成指标
        self.rouge_scores = []
        self.bleu_scores = []
        self.bert_scores = []
        
        # 详细统计
        self.type_wise_metrics = defaultdict(lambda: defaultdict(list))
        self.detailed_results = []
    
    def add_batch(self, 
                  predictions: List[str], 
                  references: List[str], 
                  question_types: Optional[List[str]] = None,
                  video_ids: Optional[List[str]] = None,
                  questions: Optional[List[str]] = None):
        """添加一个批次的预测结果"""
        
        batch_size = len(predictions)
        assert len(references) == batch_size, "预测和参考答案数量不匹配"
        
        # 添加到累积列表
        self.predictions.extend(predictions)
        self.references.extend(references)
        
        if question_types:
            self.question_types.extend(question_types)
        else:
            self.question_types.extend(['unknown'] * batch_size)
        
        if video_ids:
            self.video_ids.extend(video_ids)
        else:
            self.video_ids.extend([f'video_{i}' for i in range(len(self.video_ids), len(self.video_ids) + batch_size)])
        
        if questions:
            self.qa_pairs.extend(list(zip(questions, predictions, references)))
        else:
            self.qa_pairs.extend([('', pred, ref) for pred, ref in zip(predictions, references)])
        
        # 计算批次指标
        self._compute_batch_metrics(predictions, references, question_types)
    
    def _compute_batch_metrics(self, 
                              predictions: List[str], 
                              references: List[str],
                              question_types: Optional[List[str]] = None):
        """计算批次指标"""
        
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            q_type = question_types[i] if question_types else 'unknown'
            
            # 文本预处理
            pred_clean = self._clean_text(pred)
            ref_clean = self._clean_text(ref)
            
            # 精确匹配
            exact_match = (pred_clean.lower() == ref_clean.lower())
            self.type_wise_metrics[q_type]['exact_match'].append(exact_match)
            
            # ROUGE分数
            rouge_scores = self.rouge_scorer.score(pred_clean, ref_clean)
            for metric, score in rouge_scores.items():
                self.type_wise_metrics[q_type][f'rouge_{metric}'].append(score.fmeasure)
            
            # BLEU分数
            try:
                bleu_score = self._compute_bleu_score(pred_clean, ref_clean)
                self.type_wise_metrics[q_type]['bleu'].append(bleu_score)
            except:
                self.type_wise_metrics[q_type]['bleu'].append(0.0)
            
            # 词级别指标
            pred_words = self._tokenize_text(pred_clean)
            ref_words = self._tokenize_text(ref_clean)
            
            # F1分数
            f1_score = self._compute_f1_score(pred_words, ref_words)
            self.type_wise_metrics[q_type]['f1'].append(f1_score)
            
            # 记录详细结果
            self.detailed_results.append({
                'video_id': self.video_ids[len(self.detailed_results)] if len(self.video_ids) > len(self.detailed_results) else f'video_{len(self.detailed_results)}',
                'question_type': q_type,
                'prediction': pred,
                'reference': ref,
                'exact_match': exact_match,
                'rouge_l': rouge_scores['rougeL'].fmeasure,
                'bleu': bleu_score if 'bleu_score' in locals() else 0.0,
                'f1': f1_score
            })
    
    def compute_metrics(self) -> Dict[str, Any]:
        """计算最终指标"""
        if not self.predictions:
            return {}
        
        metrics = {}
        
        # 总体指标
        metrics['total_samples'] = len(self.predictions)
        
        # 精确匹配
        all_exact_matches = []
        for q_type in self.type_wise_metrics:
            all_exact_matches.extend(self.type_wise_metrics[q_type]['exact_match'])
        metrics['exact_match_accuracy'] = np.mean(all_exact_matches) if all_exact_matches else 0.0
        
        # ROUGE分数
        rouge_metrics = ['rouge_rouge1', 'rouge_rouge2', 'rouge_rougeL']
        for metric in rouge_metrics:
            all_scores = []
            for q_type in self.type_wise_metrics:
                all_scores.extend(self.type_wise_metrics[q_type].get(metric, []))
            metrics[metric.replace('rouge_', '')] = np.mean(all_scores) if all_scores else 0.0
        
        # BLEU分数
        all_bleu = []
        for q_type in self.type_wise_metrics:
            all_bleu.extend(self.type_wise_metrics[q_type].get('bleu', []))
        metrics['bleu'] = np.mean(all_bleu) if all_bleu else 0.0
        
        # F1分数
        all_f1 = []
        for q_type in self.type_wise_metrics:
            all_f1.extend(self.type_wise_metrics[q_type].get('f1', []))
        metrics['f1'] = np.mean(all_f1) if all_f1 else 0.0
        
        # 分类型指标
        metrics['by_question_type'] = {}
        for q_type in self.type_wise_metrics:
            type_metrics = {}
            for metric_name, scores in self.type_wise_metrics[q_type].items():
                type_metrics[metric_name] = {
                    'mean': np.mean(scores) if scores else 0.0,
                    'std': np.std(scores) if scores else 0.0,
                    'count': len(scores)
                }
            metrics['by_question_type'][q_type] = type_metrics
        
        # BERT分数（如果有足够样本）
        if len(self.predictions) <= 1000:  # 避免内存问题
            try:
                bert_scores = self._compute_bert_scores(self.predictions, self.references)
                metrics['bert_score'] = {
                    'precision': np.mean(bert_scores['precision']),
                    'recall': np.mean(bert_scores['recall']),
                    'f1': np.mean(bert_scores['f1'])
                }
            except Exception as e:
                print(f"BERT分数计算失败: {e}")
                metrics['bert_score'] = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        return metrics
    
    def _clean_text(self, text: str) -> str:
        """清理文本"""
        if not text:
            return ""
        
        # 转换为小写
        text = text.lower()
        
        # 移除标点符号
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # 移除多余空格
        text = ' '.join(text.split())
        
        return text
    
    def _tokenize_text(self, text: str) -> List[str]:
        """分词"""
        try:
            tokens = word_tokenize(text)
            # 移除停用词并进行词形还原
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
            return tokens
        except:
            return text.split()
    
    def _compute_bleu_score(self, prediction: str, reference: str) -> float:
        """计算BLEU分数"""
        try:
            # 使用sacrebleu计算
            bleu = sacrebleu.sentence_bleu(prediction, [reference])
            return bleu.score / 100.0  # 转换为0-1范围
        except:
            # 回退到简单实现
            pred_tokens = prediction.split()
            ref_tokens = reference.split()
            
            if not pred_tokens or not ref_tokens:
                return 0.0
            
            # 计算1-gram精确度
            pred_1gram = set(pred_tokens)
            ref_1gram = set(ref_tokens)
            
            if not ref_1gram:
                return 0.0
            
            precision = len(pred_1gram & ref_1gram) / len(pred_1gram)
            recall = len(pred_1gram & ref_1gram) / len(ref_1gram)
            
            if precision + recall == 0:
                return 0.0
            
            return 2 * precision * recall / (precision + recall)
    
    def _compute_f1_score(self, pred_words: List[str], ref_words: List[str]) -> float:
        """计算F1分数"""
        if not pred_words or not ref_words:
            return 0.0
        
        pred_set = set(pred_words)
        ref_set = set(ref_words)
        
        if not ref_set:
            return 0.0
        
        intersection = pred_set & ref_set
        
        if not intersection:
            return 0.0
        
        precision = len(intersection) / len(pred_set)
        recall = len(intersection) / len(ref_set)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)
    
    def _compute_bert_scores(self, predictions: List[str], references: List[str]) -> Dict[str, List[float]]:
        """计算BERT分数"""
        P, R, F1 = self.bert_scorer.score(predictions, references)
        
        return {
            'precision': P.tolist(),
            'recall': R.tolist(),
            'f1': F1.tolist()
        }
    
    def get_detailed_results(self) -> List[Dict]:
        """获取详细结果"""
        return self.detailed_results
    
    def save_results(self, filepath: str):
        """保存结果到文件"""
        results = {
            'metrics': self.compute_metrics(),
            'detailed_results': self.get_detailed_results(),
            'summary': {
                'total_samples': len(self.predictions),
                'question_types': list(set(self.question_types)),
                'unique_videos': len(set(self.video_ids))
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    def print_summary(self):
        """打印指标摘要"""
        metrics = self.compute_metrics()
        
        print("=== NTLBG-LLM评估结果摘要 ===")
        print(f"总样本数: {metrics.get('total_samples', 0)}")
        print(f"精确匹配准确率: {metrics.get('exact_match_accuracy', 0.0):.4f}")
        print(f"ROUGE-L: {metrics.get('rougeL', 0.0):.4f}")
        print(f"BLEU: {metrics.get('bleu', 0.0):.4f}")
        print(f"F1: {metrics.get('f1', 0.0):.4f}")
        
        if 'bert_score' in metrics:
            print(f"BERT-F1: {metrics['bert_score']['f1']:.4f}")
        
        print("\n=== 分问题类型结果 ===")
        for q_type, type_metrics in metrics.get('by_question_type', {}).items():
            print(f"\n{q_type}:")
            if 'exact_match' in type_metrics:
                print(f"  精确匹配: {type_metrics['exact_match']['mean']:.4f}")
            if 'rouge_rougeL' in type_metrics:
                print(f"  ROUGE-L: {type_metrics['rouge_rougeL']['mean']:.4f}")
            if 'bleu' in type_metrics:
                print(f"  BLEU: {type_metrics['bleu']['mean']:.4f}")
            if 'f1' in type_metrics:
                print(f"  F1: {type_metrics['f1']['mean']:.4f}")
            print(f"  样本数: {type_metrics.get('exact_match', {}).get('count', 0)}")


class NTLBGSpecificMetrics:
    """NTLBG特定指标"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 代表点相关指标
        self.representative_points_metrics = []
        self.coverage_scores = []
        self.diversity_scores = []
        self.efficiency_scores = []
        
        # 统计学指标
        self.mahalanobis_distances = []
        self.distribution_consistency = []
        self.query_relevance_scores = []
    
    def add_ntlbg_outputs(self, outputs: Dict[str, torch.Tensor]):
        """添加NTLBG模型输出"""
        # 提取NTLBG相关信息
        ntlbg_components = outputs.get('ntlbg_components', {})
        
        # 代表点覆盖度
        if 'coverage_score' in ntlbg_components:
            coverage = ntlbg_components['coverage_score']
            self.coverage_scores.append(coverage.mean().item())
        
        # 代表点多样性
        if 'diversity_score' in ntlbg_components:
            diversity = ntlbg_components['diversity_score']
            self.diversity_scores.append(diversity.mean().item())
        
        # 查询相关性
        if 'query_relevance' in ntlbg_components:
            relevance = ntlbg_components['query_relevance']
            self.query_relevance_scores.append(relevance.mean().item())
        
        # 马氏距离一致性
        if 'representative_points' in ntlbg_components:
            rep_points = ntlbg_components['representative_points']
            mean = ntlbg_components.get('distribution_mean')
            cov = ntlbg_components.get('distribution_cov')
            
            if mean is not None and cov is not None:
                mahal_dist = self._compute_mahalanobis_consistency(rep_points, mean, cov)
                self.mahalanobis_distances.append(mahal_dist)
        
        # 计算效率指标
        if 'attention_weights' in ntlbg_components:
            attention = ntlbg_components['attention_weights']
            efficiency = self._compute_efficiency_score(attention)
            self.efficiency_scores.append(efficiency)
    
    def _compute_mahalanobis_consistency(self, points: torch.Tensor, mean: torch.Tensor, cov: torch.Tensor) -> float:
        """计算马氏距离一致性"""
        batch_size = points.size(0)
        n_points = points.size(1)
        
        total_consistency = 0.0
        
        for i in range(batch_size):
            # 计算每个点的马氏距离
            diff = points[i] - mean[i].unsqueeze(0)
            
            try:
                cov_inv = torch.inverse(cov[i] + 1e-6 * torch.eye(cov.size(-1), device=cov.device))
                mahal_dist = torch.sqrt(torch.sum(diff * torch.matmul(diff, cov_inv), dim=-1))
                
                # 计算距离的标准差（一致性）
                consistency = 1.0 / (1.0 + torch.std(mahal_dist).item())
                total_consistency += consistency
            except:
                total_consistency += 0.0
        
        return total_consistency / batch_size
    
    def _compute_efficiency_score(self, attention_weights: torch.Tensor) -> float:
        """计算注意力效率分数"""
        # 计算注意力熵
        entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=-1)
        
        # 归一化熵
        max_entropy = torch.log(torch.tensor(attention_weights.size(-1), dtype=torch.float32))
        normalized_entropy = entropy / max_entropy
        
        # 效率分数：1 - 归一化熵（越集中越高效）
        efficiency = 1.0 - normalized_entropy.mean().item()
        
        return efficiency
    
    def compute_ntlbg_metrics(self) -> Dict[str, float]:
        """计算NTLBG特定指标"""
        metrics = {}
        
        if self.coverage_scores:
            metrics['coverage_score'] = {
                'mean': np.mean(self.coverage_scores),
                'std': np.std(self.coverage_scores),
                'min': np.min(self.coverage_scores),
                'max': np.max(self.coverage_scores)
            }
        
        if self.diversity_scores:
            metrics['diversity_score'] = {
                'mean': np.mean(self.diversity_scores),
                'std': np.std(self.diversity_scores),
                'min': np.min(self.diversity_scores),
                'max': np.max(self.diversity_scores)
            }
        
        if self.query_relevance_scores:
            metrics['query_relevance'] = {
                'mean': np.mean(self.query_relevance_scores),
                'std': np.std(self.query_relevance_scores),
                'min': np.min(self.query_relevance_scores),
                'max': np.max(self.query_relevance_scores)
            }
        
        if self.mahalanobis_distances:
            metrics['mahalanobis_consistency'] = {
                'mean': np.mean(self.mahalanobis_distances),
                'std': np.std(self.mahalanobis_distances),
                'min': np.min(self.mahalanobis_distances),
                'max': np.max(self.mahalanobis_distances)
            }
        
        if self.efficiency_scores:
            metrics['attention_efficiency'] = {
                'mean': np.mean(self.efficiency_scores),
                'std': np.std(self.efficiency_scores),
                'min': np.min(self.efficiency_scores),
                'max': np.max(self.efficiency_scores)
            }
        
        return metrics


class EvaluationRunner:
    """评估运行器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.video_qa_metrics = VideoQAMetrics(config)
        self.ntlbg_metrics = NTLBGSpecificMetrics(config)
    
    def evaluate_batch(self, 
                      model_outputs: Dict[str, torch.Tensor],
                      batch: Dict[str, torch.Tensor],
                      predictions: List[str],
                      references: List[str],
                      question_types: Optional[List[str]] = None,
                      video_ids: Optional[List[str]] = None,
                      questions: Optional[List[str]] = None):
        """评估一个批次"""
        
        # 添加到VideoQA指标
        self.video_qa_metrics.add_batch(
            predictions=predictions,
            references=references,
            question_types=question_types,
            video_ids=video_ids,
            questions=questions
        )
        
        # 添加到NTLBG指标
        self.ntlbg_metrics.add_ntlbg_outputs(model_outputs)
    
    def compute_all_metrics(self) -> Dict[str, Any]:
        """计算所有指标"""
        results = {}
        
        # VideoQA指标
        results['video_qa'] = self.video_qa_metrics.compute_metrics()
        
        # NTLBG指标
        results['ntlbg'] = self.ntlbg_metrics.compute_ntlbg_metrics()
        
        # 综合指标
        results['summary'] = {
            'overall_performance': results['video_qa'].get('exact_match_accuracy', 0.0),
            'generation_quality': results['video_qa'].get('rougeL', 0.0),
            'ntlbg_effectiveness': results['ntlbg'].get('coverage_score', {}).get('mean', 0.0),
            'efficiency': results['ntlbg'].get('attention_efficiency', {}).get('mean', 0.0)
        }
        
        return results
    
    def save_results(self, filepath: str):
        """保存评估结果"""
        results = self.compute_all_metrics()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    def print_summary(self):
        """打印评估摘要"""
        self.video_qa_metrics.print_summary()
        
        ntlbg_metrics = self.ntlbg_metrics.compute_ntlbg_metrics()
        if ntlbg_metrics:
            print("\n=== NTLBG特定指标 ===")
            for metric_name, metric_stats in ntlbg_metrics.items():
                print(f"{metric_name}: {metric_stats.get('mean', 0.0):.4f} ± {metric_stats.get('std', 0.0):.4f}")
    
    def reset_metrics(self):
        """重置所有指标"""
        self.video_qa_metrics.reset_metrics()
        self.ntlbg_metrics = NTLBGSpecificMetrics(self.config) 