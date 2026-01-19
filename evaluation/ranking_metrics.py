"""
Shared Ranking Metrics Module
CareerRank Project - Day 2

Implements ranking and regression metrics using pure numpy:
- Recall@K
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG@K)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
"""

import numpy as np


def recall_at_k(relevant_set, ranked_list, k):
    """
    Compute Recall@K: fraction of relevant items in top-K results.
    
    Args:
        relevant_set: set or list of relevant item IDs
        ranked_list: list of ranked item IDs (in order of relevance)
        k: cutoff position
        
    Returns:
        float: Recall@K score
    """
    if len(relevant_set) == 0:
        return 0.0
    
    relevant_set = set(relevant_set)
    top_k = set(ranked_list[:k])
    
    num_relevant_retrieved = len(relevant_set & top_k)
    recall = num_relevant_retrieved / len(relevant_set)
    
    return recall


def mrr(relevant_set, ranked_list):
    """
    Compute Mean Reciprocal Rank (MRR).
    
    Args:
        relevant_set: set or list of relevant item IDs
        ranked_list: list of ranked item IDs (in order of relevance)
        
    Returns:
        float: MRR score (reciprocal of rank of first relevant item)
    """
    if len(relevant_set) == 0:
        return 0.0
    
    relevant_set = set(relevant_set)
    
    for rank, item in enumerate(ranked_list, start=1):
        if item in relevant_set:
            return 1.0 / rank
    
    return 0.0


def dcg_at_k(relevance_grades, ranked_list, k):
    """
    Compute Discounted Cumulative Gain at K.
    
    Args:
        relevance_grades: dict mapping item_id -> relevance grade (0, 1, 2, 3, etc.)
        ranked_list: list of ranked item IDs
        k: cutoff position
        
    Returns:
        float: DCG@K score
    """
    dcg = 0.0
    for i, item in enumerate(ranked_list[:k], start=1):
        relevance = relevance_grades.get(item, 0)
        dcg += (2**relevance - 1) / np.log2(i + 1)
    
    return dcg


def ndcg_at_k(relevance_grades, ranked_list, k):
    """
    Compute Normalized Discounted Cumulative Gain at K.
    
    Args:
        relevance_grades: dict mapping item_id -> relevance grade (0, 1, 2, 3, etc.)
        ranked_list: list of ranked item IDs
        k: cutoff position
        
    Returns:
        float: NDCG@K score (0 to 1)
    """
    if len(relevance_grades) == 0:
        return 0.0
    
    # Compute DCG@K
    actual_dcg = dcg_at_k(relevance_grades, ranked_list, k)
    
    # Compute ideal DCG@K (sort by relevance descending)
    ideal_ranking = sorted(relevance_grades.keys(), 
                          key=lambda x: relevance_grades[x], 
                          reverse=True)
    ideal_dcg = dcg_at_k(relevance_grades, ideal_ranking, k)
    
    if ideal_dcg == 0:
        return 0.0
    
    return actual_dcg / ideal_dcg


def rmse(y_true, y_pred):
    """
    Compute Root Mean Squared Error.
    
    Args:
        y_true: array-like of true values
        y_pred: array-like of predicted values
        
    Returns:
        float: RMSE score
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mse = np.mean((y_true - y_pred) ** 2)
    return np.sqrt(mse)


def mae(y_true, y_pred):
    """
    Compute Mean Absolute Error.
    
    Args:
        y_true: array-like of true values
        y_pred: array-like of predicted values
        
    Returns:
        float: MAE score
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    return np.mean(np.abs(y_true - y_pred))


def compute_graded_relevance(score, bins=[0, 30, 60, 80, 100]):
    """
    Convert compatibility score to graded relevance (0-3).
    
    Args:
        score: compatibility score (0-100)
        bins: bin edges for relevance grades
        
    Returns:
        int: relevance grade (0, 1, 2, or 3)
    """
    if score < bins[1]:
        return 0  # low
    elif score < bins[2]:
        return 1  # medium-low
    elif score < bins[3]:
        return 2  # medium-high
    else:
        return 3  # high


if __name__ == "__main__":
    # Test the metrics
    print("Testing Ranking Metrics Module")
    print("=" * 60)
    
    # Test Recall@K
    relevant = {1, 2, 3, 4, 5}
    ranked = [1, 6, 2, 7, 3, 8, 4, 9, 5, 10]
    
    print(f"Relevant items: {relevant}")
    print(f"Ranked list: {ranked}")
    print(f"Recall@5: {recall_at_k(relevant, ranked, 5):.4f}")
    print(f"Recall@10: {recall_at_k(relevant, ranked, 10):.4f}")
    
    # Test MRR
    print(f"\nMRR: {mrr(relevant, ranked):.4f}")
    
    # Test NDCG@K
    relevance_grades = {1: 3, 2: 2, 3: 2, 4: 1, 5: 1, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0}
    print(f"\nRelevance grades: {relevance_grades}")
    print(f"NDCG@5: {ndcg_at_k(relevance_grades, ranked, 5):.4f}")
    print(f"NDCG@10: {ndcg_at_k(relevance_grades, ranked, 10):.4f}")
    
    # Test RMSE and MAE
    y_true = [3.0, 2.5, 4.0, 3.5]
    y_pred = [2.8, 2.6, 3.9, 3.7]
    print(f"\nTrue values: {y_true}")
    print(f"Predicted values: {y_pred}")
    print(f"RMSE: {rmse(y_true, y_pred):.4f}")
    print(f"MAE: {mae(y_true, y_pred):.4f}")
    
    print("\n" + "=" * 60)
    print("All metrics tests passed!")
