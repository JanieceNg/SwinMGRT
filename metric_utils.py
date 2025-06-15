import torch
from sklearn.metrics import recall_score, f1_score, precision_score


class Metric:
    def __init__(self, num_classes, topk=(1, 5)):
        """
        计算分类任务的指标，并在整个 epoch 中更新状态。

        :param num_classes: 类别总数
        :param topk: 需要计算的 top-k 指标
        """
        self.num_classes = num_classes
        self.topk = topk
        self.reset()

    def update(self, preds, targets):
        """
        更新指标计算的计数器。

        :param preds: 预测的 logits (形状: [batch_size, num_classes])
        :param targets: 真实标签 (形状: [batch_size])
        """
        batch_size = targets.size(0)

        # 计算 Top-k 预测
        _, pred_topk = preds.topk(max(self.topk), dim=1, largest=True, sorted=True)

        for k in self.topk:
            correct_k = pred_topk[:, :k].eq(targets.view(-1, 1).expand_as(pred_topk[:, :k])).sum().item()
            self.correct[k] += correct_k

        self.total += batch_size
        self.all_preds.extend(preds.argmax(dim=1).cpu().tolist())
        self.all_targets.extend(targets.cpu().tolist())

    def compute(self):
        """
        计算指标，包括 Top-k, Recall, F1-score。
        """
        results = {f'Top-{k}': self.correct[k] / self.total * 100 for k in self.topk}

        # 计算 Recall 和 F1-score（支持多类别）
        results["Precision"] = precision_score(self.all_targets, self.all_preds, average="macro", zero_division=0) * 100
        results['Recall'] = recall_score(self.all_targets, self.all_preds, average='macro', zero_division=0) * 100
        results['F1'] = f1_score(self.all_targets, self.all_preds, average='macro', zero_division=0) * 100

        return results

    def reset(self):
        """
        重置指标计数器。
        """
        self.correct = {k: 0 for k in self.topk}
        self.total = 0
        self.all_preds = []
        self.all_targets = []

if __name__ == '__main__':
    # 初始化 metric 计算器
    metric = Metric(num_classes=1000, topk=(1, 5))

    # 模拟一个 epoch 中的数据
    for i in range(10):  # 假设 10 个 batch
        preds = torch.randn(32, 1000)  # 32 个样本，1000 个类别的 logits
        targets = torch.randint(0, 1000, (32,))  # 32 个真实标签

        metric.update(preds, targets)

    # 计算最终指标
    results = metric.compute()
    print(results)

    # 进入新 epoch 前清空数据
    metric_tracker.reset()
