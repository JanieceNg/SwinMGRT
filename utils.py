import os
import sys
import json

import fontTools.ttLib.tables.sbixStrike
import torch.nn.functional as F
import torch
import matplotlib
matplotlib.use('Agg')  # 设置为无图形界面的后端
from tqdm import tqdm
import yaml
from sklearn.metrics import recall_score, f1_score, precision_score
import matplotlib.pyplot as plt
from datetime import datetime
import time
import functools
import psutil
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_fscore_support, accuracy_score,
    top_k_accuracy_score
)
from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_ratio(model_state_dict,state_dict_to_load):
    # 计算成功导入的权重数量
    successful_weights = 0
    for key in state_dict_to_load:
        #print(key,model_state_dict[key].shape)
        if key in model_state_dict:
            if state_dict_to_load[key].shape == model_state_dict[key].shape:
                successful_weights += 1
            else:
                print('-'*10)
                RuntimeError(f'{key} excepted shape is {model_state_dict[key].shape} but be {state_dict_to_load[key].shape}')

    print(f"Total number of weights imported successfully: {successful_weights}/{len(model_state_dict)}")

def show_params(model):
    total_params = 0
    print("👀 Layer-wise Parameter Count:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            print(f"  🔹 {name:<50} | Params: {num_params:,}")
            total_params += num_params
    print(f"\n🧠 Total Parameters: {total_params:,}")

@torch.no_grad()
def Inference(model, dataloader,class_indices, scale_class,device='cuda', save_path="predict.json",):
    num_classes = len(class_indices)
    class_indices_reverse = dict((v, k) for k, v in class_indices.items())
    class_names = [class_indices_reverse[i] for i in range(num_classes)]
    model.eval()
    all_urls = []
    all_preds = []
    all_scores = []

    with torch.no_grad():
        for inputs, targets, urls in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs[-1], dim=1)
            topk_probs, topk_preds = torch.topk(probs,k=20, dim=1)
            all_urls.extend(list(urls))
            all_preds.extend([p.tolist() for p in topk_preds.cpu()])
            all_scores.extend([p.tolist() for p in topk_probs.cpu()])

    ds = []
    for url,pred,score in zip(all_urls,all_preds,all_scores ):
        ds.append({'url':url,'pred':pred,'score':score})
    with open(save_path, 'w') as f:
        json.dump(ds,f,indent=4)
    print('predict result saved in %s'%save_path)

@torch.no_grad()
def Test_val(model, dataloader, class_indices, device='cuda', topk=(1, 5),
             save_csv_path="per_class_metrics.csv", save_cm_path="confusion_matrix.png"):

    num_classes = len(class_indices)
    class_indices_reverse = dict((v,k) for k,v in class_indices.items())
    class_names = [class_indices_reverse[i] for i in range(num_classes)]

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            targets = targets.to(device).transpose(0,1)[-1]

            outputs = model(inputs)
            probs = F.softmax(outputs[-1], dim=1)

            all_preds.append(probs.cpu())
            all_targets.append(targets.cpu())


    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    pred_labels = all_preds.argmax(dim=1)
    all_targets_np = all_targets.numpy()

    # ===== Overall Top-1 Accuracy =====
    top1_acc = accuracy_score(all_targets, pred_labels)

    # ===== Confusion Matrix & Top-1 per class =====
    cm = confusion_matrix(all_targets, pred_labels, labels=range(num_classes))
    classwise_correct = np.diag(cm)
    classwise_total = cm.sum(axis=1)
    top1_per_class = classwise_correct / (classwise_total + 1e-8)

    # ===== Top-k per class (excluding k=1) =====
    topk = [k for k in topk if k > 1]  # remove 1 if present
    topk_preds = torch.topk(all_preds, k=max(topk), dim=1).indices  # shape: (N, k)
    topk_per_class = {}

    for k_val in topk:
        correct_per_class = np.zeros(num_classes)
        total_per_class = np.zeros(num_classes)

        for i in range(len(all_targets_np)):
            true_label = all_targets_np[i]
            total_per_class[true_label] += 1
            if true_label in topk_preds[i, :k_val]:
                correct_per_class[true_label] += 1

        topk_acc_per_class = correct_per_class / (total_per_class + 1e-8)
        topk_per_class[k_val] = topk_acc_per_class

    # ===== Overall Top-k Accuracy =====
    overall_topk_acc = {
        f"Top-{k} Accuracy": top_k_accuracy_score(all_targets, all_preds, k=k, labels=list(range(num_classes)))
        for k in topk
    }

    # ===== Precision, Recall, F1 =====
    precision, recall, f1, support = precision_recall_fscore_support(
        all_targets, pred_labels, labels=range(num_classes), zero_division=0
    )

    # ===== 构建 DataFrame =====
    df_dict = {
        "Class": class_names,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "Top-1 Accuracy": top1_per_class,
        "Support": support,
    }

    for k_val in topk:
        df_dict[f"Top-{k_val} Accuracy"] = topk_per_class[k_val]

    df = pd.DataFrame(df_dict)
    df_sorted = df.sort_values(by="Top-1 Accuracy", ascending=True)
    df_sorted.to_csv(save_csv_path, index=False)
    print(f"\n✅ Per-class metrics saved to: {save_csv_path}")

    # ===== 混淆矩阵图 =====
    # plt.figure(figsize=(12, 10))
    # sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    # plt.title("Confusion Matrix")
    # plt.xlabel("Predicted")
    # plt.ylabel("True")
    # plt.tight_layout()
    # plt.savefig(save_cm_path)
    # print(f"✅ Confusion matrix saved to: {save_cm_path}")
    # plt.close()

    # ===== 输出总览 =====
    print("\n�� Overall Metrics:")
    print(f"Top-1 Accuracy: {top1_acc:.4f}")
    for k, v in overall_topk_acc.items():
        print(f"{k}: {v:.4f}")

    return {
        "top1": top1_acc,
        "topk": overall_topk_acc,
        "per_class_df": df_sorted,
        "confusion_matrix": cm
    }

class TrainRecorder:
    def __init__(self, log_path='log.json'):
        self.log_path = log_path
        self.meta = {
            'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_epochs': 0,
            'notes': '',
        }
        self.logs = {}
        self._epoch_timers = {}  # 临时记录开始时间
        if os.path.exists(self.log_path):
            self.load()

    def update(self, epoch, info_dict):
        """更新指定 epoch 的信息，只更新给定字段，保留原有字段"""
        if str(epoch) not in self.logs:
            self.logs[str(epoch)] = {}
        self.logs[str(epoch)].update(info_dict)
        # print(self.logs[str(epoch)]) #debug
        self.meta['total_epochs'] = max(self.meta['total_epochs'], epoch + 1)

    def save(self):
        with open(self.log_path, 'w') as f:
            json.dump({'meta': self.meta, 'logs': self.logs}, f, indent=2)

    def load(self):
        with open(self.log_path, 'r') as f:
            data = json.load(f)
            self.meta = data.get('meta', self.meta)
            self.logs = data.get('logs', {})

    def check_best(self,key):
        best_epoch = 0
        best_score = 0
        for k,v in self.logs.items():
            s = self.logs[k][key]
            if s>best_score:
                best_epoch = int(k)
                best_score = s
        print(f'The best {key} is in epoch {best_epoch}')
        print(f'The best {key} is {best_score}')
        print(self.logs[str(best_epoch)])


class Metric:
    def __init__(self, num_classes, topk=None):
        """
        计算分类任务的指标，并在整个 epoch 中更新状态。

        :param num_classes: 类别总数
        :param topk: 需要计算的 top-k 指标
        """
        self.num_classes = num_classes
        if not topk:
            topk = (1,5) if num_classes>5 else (1,)
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
        self.reset()
        return results

    def reset(self):
        """
        重置指标计数器。
        """
        self.correct = {k: 0 for k in self.topk}
        self.total = 0
        self.all_preds = []
        self.all_targets = []


class ScaleMetric:
    def __init__(self, num_classes):
        self.metrics = [Metric(n) for n in num_classes]

    def update(self, preds, targets):
        for i,m in enumerate(self.metrics):
            m.update(preds[i], targets[i])

    def compute(self):
        res = [m.compute() for m in self.metrics]
        return res

    def reset(self):
        for m in self.metrics:
            m.reset()


def save_config(config, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        yaml.dump(config._dict, file, allow_unicode=True)

def timing_and_memory(func):
    """装饰器：测量函数运行时间和内存变化"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 获取当前进程
        process = psutil.Process(os.getpid())
        # 获取开始前的内存使用情况
        mem_before = process.memory_info().rss / (1024 * 1024)

        # 记录开始时间
        start_time = time.time()
        # 执行函数
        result = func(*args, **kwargs)
        # 记录结束时间
        end_time = time.time()

        # 获取结束后的内存使用情况
        mem_after = process.memory_info().rss / (1024 * 1024)
        # 计算内存变化
        mem_change = mem_after - mem_before

        # 打印结果
        print(f"Function {func.__name__!r} ran in {(end_time - start_time):.4f}s")
        print(f"Memory usage changed by {mem_change:.4f} MB")

        return result

    return wrapper

def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = '../class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()



def train_one_epoch_scale(model, optimizer, scheduler, data_loader,loss_function, device, epoch):
    model.train()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_sp_top1_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_sp_top5_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_ge_top1_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_ge_top5_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_fa_top1_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    nums_steped = len(data_loader) * epoch
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        fa_labels = labels['family'].to(device)
        ge_labels = labels['genus'].to(device)
        sp_labels = labels['species'].to(device)
        labels = [fa_labels,ge_labels,sp_labels]
        sample_num += images.shape[0]
        preds = model(images.to(device))
        loss = loss_function(preds, labels)
        loss.backward()
        accu_loss += loss.detach()

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step(nums_steped + step)

        sp_labels = sp_labels.to('cpu')
        sp_probs = F.softmax(preds[2].cpu(), dim=1)
        _, top1_sp_preds = sp_probs.topk(1, dim=1)
        _, top5_sp_preds = sp_probs.topk(5, dim=1)
        accu_sp_top1_num += top1_sp_preds.eq(sp_labels.view(-1, 1)).sum().item()
        accu_sp_top5_num += top5_sp_preds.eq(sp_labels.view(-1, 1).expand_as(top5_sp_preds)).sum().item()

        ge_labels = ge_labels.to('cpu')
        ge_probs = F.softmax(preds[1].cpu(), dim=1)
        _, top1_ge_preds = ge_probs.topk(1, dim=1)
        _, top5_ge_preds = ge_probs.topk(5, dim=1)
        accu_ge_top1_num += top1_ge_preds.eq(ge_labels.view(-1, 1)).sum().item()
        accu_ge_top5_num += top5_ge_preds.eq(ge_labels.view(-1, 1).expand_as(top5_ge_preds)).sum().item()

        fa_labels = fa_labels.to('cpu')
        fa_probs = F.softmax(preds[0].cpu(), dim=1)
        _, top1_fa_preds = fa_probs.topk(1, dim=1)
        _, top5_fa_preds = fa_probs.topk(5, dim=1)
        accu_fa_top1_num += top1_fa_preds.eq(fa_labels.view(-1, 1)).sum().item()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, fa_top1: {:.3f}, ge_top1: {:.3f}, ge_top5: {:.3f}, sp_top1: {:.3f}, sp_top5: {:.3f}".format(epoch,
                                                                                                                                                       accu_loss.item() / (
                                                                                                                                                               step + 1),
                                                                                                                                                       accu_fa_top1_num.item() / sample_num,
                                                                                                                                                       accu_ge_top1_num.item() / sample_num,
                                                                                                                                                       accu_ge_top5_num.item() / sample_num,
                                                                                                                                                       accu_sp_top1_num.item() / sample_num,
                                                                                                                                                       accu_sp_top5_num.item() / sample_num)
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

    return accu_loss.item() / (step + 1), accu_fa_top1_num.item() / sample_num, accu_ge_top1_num.item() / sample_num, accu_ge_top5_num.item() / sample_num,accu_sp_top1_num.item() / sample_num, accu_sp_top5_num.item() / sample_num


def train_one_epoch(model, optimizer, scheduler, data_loader,loss_function, device, epoch):

    model.train()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_top1_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_top5_num = torch.zeros(1).to(device)  # 累计预测正确的样本数

    optimizer.zero_grad()

    sample_num = 0
    nums_steped = len(data_loader) * epoch
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step(nums_steped+step)
        labels = labels.to('cpu')
        probs = F.softmax(pred.cpu(), dim=1)
        _, top1_preds = probs.topk(1, dim=1)
        _, top5_preds = probs.topk(5, dim=1)
        accu_top1_num += top1_preds.eq(labels.view(-1,1)).sum().item()
        accu_top5_num += top5_preds.eq(labels.view(-1,1).expand_as(top5_preds)).sum().item()
        data_loader.desc = "[train epoch {}] loss: {:.3f}, top1_acc: {:.3f}, top5_acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_top1_num.item() / sample_num,
                                                                               accu_top5_num.item() / sample_num)
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

    return accu_loss.item() / (step + 1), accu_top1_num.item() / sample_num, accu_top5_num.item() / sample_num


@torch.no_grad()
def evaluate_scale(model, data_loader,loss_function, device, epoch):
    model.eval()

    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_sp_top1_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_sp_top5_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_ge_top1_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_ge_top5_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_fa_top1_num = torch.zeros(1).to(device)  # 累计预测正确的样本数

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        fa_labels = labels['family'].to(device)
        ge_labels = labels['genus'].to(device)
        sp_labels = labels['species'].to(device)
        labels = [fa_labels, ge_labels, sp_labels]
        sample_num += images.shape[0]
        preds = model(images.to(device))
        loss = loss_function(preds, labels)
        accu_loss += loss

        sp_labels = sp_labels.to('cpu')
        sp_probs = F.softmax(preds[2].cpu(), dim=1)
        _, top1_sp_preds = sp_probs.topk(1, dim=1)
        _, top5_sp_preds = sp_probs.topk(5, dim=1)
        accu_sp_top1_num += top1_sp_preds.eq(sp_labels.view(-1, 1)).sum().item()
        accu_sp_top5_num += top5_sp_preds.eq(sp_labels.view(-1, 1).expand_as(top5_sp_preds)).sum().item()

        ge_labels = ge_labels.to('cpu')
        ge_probs = F.softmax(preds[1].cpu(), dim=1)
        _, top1_ge_preds = ge_probs.topk(1, dim=1)
        _, top5_ge_preds = ge_probs.topk(5, dim=1)
        accu_ge_top1_num += top1_ge_preds.eq(ge_labels.view(-1, 1)).sum().item()
        accu_ge_top5_num += top5_ge_preds.eq(ge_labels.view(-1, 1).expand_as(top5_ge_preds)).sum().item()

        fa_labels = fa_labels.to('cpu')
        fa_probs = F.softmax(preds[0].cpu(), dim=1)
        _, top1_fa_preds = fa_probs.topk(1, dim=1)
        _, top5_fa_preds = fa_probs.topk(5, dim=1)
        accu_fa_top1_num += top1_fa_preds.eq(fa_labels.view(-1, 1)).sum().item()

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, fa_top1: {:.3f}, ge_top1: {:.3f}, ge_top5: {:.3f}, sp_top1: {:.3f}, sp_top5: {:.3f}".format(
            epoch,
            accu_loss.item() / (
                    step + 1),
            accu_fa_top1_num.item() / sample_num,
            accu_ge_top1_num.item() / sample_num,
            accu_ge_top5_num.item() / sample_num,
            accu_sp_top1_num.item() / sample_num,
            accu_sp_top5_num.item() / sample_num)

    return (accu_loss.item() / (step + 1),
            accu_fa_top1_num.item() / sample_num,
            accu_ge_top1_num.item() / sample_num,
            accu_ge_top5_num.item() / sample_num,
            accu_sp_top1_num.item() / sample_num,
            accu_sp_top5_num.item() / sample_num)


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_top1_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_top5_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失c

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        labels = labels.to(device)
        sample_num += images.shape[0]
        pred = model(images.to(device))
        loss = loss_function(pred, labels)
        accu_loss += loss

        labels = labels.to('cpu')

        probs = F.softmax(pred.cpu(), dim=1)
        _, top1_preds = probs.topk(1, dim=1)
        _, top5_preds = probs.topk(5, dim=1)

        accu_top1_num += top1_preds.eq(labels.view(-1,1)).sum().item()
        accu_top5_num += top5_preds.eq(labels.view(-1,1).expand_as(top5_preds)).sum().item()
        data_loader.desc = "[valid epoch {}] loss: {:.3f}, top1_acc: {:.3f}, top5_acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_top1_num.item() / sample_num,
                                                                               accu_top5_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_top1_num.item() / sample_num, accu_top5_num.item() / sample_num



def save_one_epoch(save_dir,epoch,model,optim,scheduler,res,best):
    d = {}
    d['epoch'] = epoch
    d['model'] = model.state_dict()
    d['optimizer_'] = optim.state_dict()
    d['scheduler'] = scheduler.state_dict()
    d['result'] = res
    d['best'] = best
    torch.save(d,save_dir)

def resume_from(checkpoint_path,model,optimizer,scheduler,device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    optimizer.load_state_dict(checkpoint['optimizer_'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model'])
    record = checkpoint['result']
    if 'best' in checkpoint:
        best = checkpoint['best']
    else:
        best={'epoch':0,'score':0}
    return epoch,model,optimizer,scheduler,record,best

# 打印模型参数量的函数，包含深度控制
def print_model_parameters(model):
    # 打印每个模块的参数量
    for name, param in model.named_parameters():
        if param.numel()>0:
            print(f'{name:<60}| {" "*10} {param.numel():10} {" "*10} parameters')
    print('-'*100)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'total parameters: {num_params}')
    print('-' * 100)