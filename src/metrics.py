import torch


def _binarize(preds, threshold=0.5):
    return (preds > threshold).float()


def _flatten(preds, labels):
    return preds.view(-1), labels.view(-1)


def _confusion(preds, labels, threshold=0.5):
    preds, labels = _binarize(preds, threshold=threshold), labels.float()
    preds, labels = _flatten(preds, labels)
    TP = torch.sum(preds * labels).item()
    FP = torch.sum(preds * (1 - labels)).item()
    FN = torch.sum((1 - preds) * labels).item()
    TN = torch.sum((1 - preds) * (1 - labels)).item()
    return TP, FP, FN, TN


def sensitivity(preds, labels, eps=1e-7, threshold=0.5):
    TP, _, FN, _ = _confusion(preds, labels, threshold=threshold)
    return TP / (TP + FN + eps)


def specificity(preds, labels, eps=1e-7, threshold=0.5):
    _, FP, _, TN = _confusion(preds, labels, threshold=threshold)
    return TN / (TN + FP + eps)


def precision(preds, labels, eps=1e-7, threshold=0.5):
    TP, FP, _, _ = _confusion(preds, labels, threshold=threshold)
    return TP / (TP + FP + eps)


def recall(preds, labels, eps=1e-7, threshold=0.5):
    return sensitivity(preds, labels, eps, threshold=threshold)


def f1_score(preds, labels, eps=1e-7, threshold=0.5):
    p = precision(preds, labels, eps, threshold=threshold)
    r = recall(preds, labels, eps, threshold=threshold)
    return 2 * p * r / (p + r + eps)


def f2_score(preds, labels, eps=1e-7, threshold=0.5):
    p = precision(preds, labels, eps, threshold=threshold)
    r = recall(preds, labels, eps, threshold=threshold)
    return 5 * p * r / (4 * p + r + eps)
