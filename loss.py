import torch
import torch.nn as nn
import torch.nn.functional as F


def get_loss(config,trainset):
    spmap = trainset.get_species_map()
    Loss = MSloss(config.LOSS,spmap)
    return Loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        """
        :param gamma: 控制易分类样本的忽略程度（默认 2.0）
        :param weight: 类别权重 (shape: [num_classes])，用于类别不均衡
        :param reduction: 'mean' 取均值, 'sum' 取总和, 'none' 逐个样本输出
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        :param logits: 预测值 (batch_size, num_classes)
        :param targets: 真实标签 (batch_size)
        """
        probs = F.softmax(logits, dim=1)  # 转成概率
        targets_one_hot = F.one_hot(targets, num_classes=logits.shape[1])  # one-hot 真实标签
        probs = torch.clamp(probs, min=1e-6, max=1.0)  # 避免 log(0) 错误

        ce_loss = -targets_one_hot * torch.log(probs)  # 计算普通交叉熵
        focal_weight = (1 - probs) ** self.gamma  # 计算 Focal Loss 权重
        loss = focal_weight * ce_loss

        if self.weight is not None:  # 加类别权重（如果提供）
            weight_tensor = self.weight.to(logits.device)
            loss *= weight_tensor[targets].view(-1, 1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # 逐个样本的 loss（适用于调试）

class MSloss(nn.Module):
    def __init__(self,loss_config,spmap=None):
        super(MSloss, self).__init__()

        self.scale_weights = loss_config.SCALE_WEIGHTS
        self.loss_in = loss_config.LOSS_IN
        self.losses = [get_loss_function(name,spmap) for name in loss_config.NAMES]

        # print(self.scale_weights)
        # print(self.loss_in)
        # print(self.losses)

    def forward(self,logits, targets):
        # print(len(self.losses),len(self.loss_in)) #debug
        # print([l.shape for l in logits])
        # losses=[]
        # for i,lin in enumerate(self.loss_in):
        #     w = self.scale_weights[i]
        #     if isinstance(lin,int):
        #         l = logits[i]
        #     else:
        #         l = [logits[ii] for ii in lin]
        #     lab = targets[lin,:]
        #     #print(lab)
        #     losses.append(w*self.losses[i](l,lab))
        # return losses
        # print(targets.shape)
        # print([w*Loss(l,targets[lin,:]) for l,Loss,lin,w in zip(logits,self.losses,self.loss_in,self.scale_weights)])
        return [w*Loss(logits[lin],targets[lin,:]) for l,Loss,lin,w in zip(logits,self.losses,self.loss_in,self.scale_weights)]


class HierarchicalFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, species_to_genus_family=None,
                 alpha=0.6, beta=0.3, gamma_penalty=1.0):
        """
        species_to_genus_family: dict[class_id] = (genus_id, family_id)
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.gamma_penalty = gamma_penalty
        self.species_map = species_to_genus_family  # e.g., {species_id: (genus_id, family_id)}

    def forward(self, logits, targets):
        print(len(logits))
        print([l.shape for l in logits])
        """
        logits: [B, num_classes]
        targets: [B, 3]  # [family_id, genus_id, species_id]
        """
        #print(targets.shape)
        probs = F.softmax(logits, dim=1)
        pred_classes = torch.argmax(probs, dim=1)  # [B]
        true_species = targets[2,:]  # [B]
        true_genus = targets[1,:]
        true_family = targets[0,:]

        pred_genus = torch.tensor([self.species_map[c.item()][0] for c in pred_classes], device=logits.device)
        pred_family = torch.tensor([self.species_map[c.item()][1] for c in pred_classes], device=logits.device)

        # Determine penalties
        same_species = pred_classes == true_species
        same_genus = (pred_genus == true_genus) & (~same_species)
        same_family = (pred_family == true_family) & (~same_genus) & (~same_species)

        penalties = torch.full_like(true_species, self.gamma_penalty, dtype=torch.float32).to(logits.device)
        penalties[same_family] = self.beta
        penalties[same_genus] = self.alpha
        penalties[same_species] = 0.0  # 无惩罚

        # Focal loss
        true_probs = probs[torch.arange(len(true_species)), true_species]
        focal_weight = (1.0 - true_probs) ** self.gamma
        logp = torch.log(true_probs + 1e-12)

        loss = penalties * focal_weight * (-logp)
        return loss.mean()

def get_loss_function(name,spmap=None):
    if name == 'focal_loss':
        loss = FocalLoss()
    elif name == 'hf_loss':
        loss = HierarchicalFocalLoss(species_to_genus_family=spmap)
    else:
        loss = torch.nn.CrossEntropyLoss()
    # if config.DATA.SCALE:
    #     assert config.LOSS.SCALE_WEIGHTS,'need set loss scale weights in config.LOSS.SCALE_WEIGHTS'
    #     Loss = MSloss(config.LOSS.SCALE_WEIGHTS,loss)
    #     return Loss
    # else:
    return loss

if __name__ == '__main__':

    # 示例：假设有 4 个类别
    num_classes = 4
    logits = torch.tensor([[2.0, 1.0, 0.5, 0.2],
                           [0.1, 3.0, 2.5, 0.3],
                           [1.2, 0.8, 3.5, 2.0]])
    targets = torch.tensor([0, 1, 2])  # 真实类别标签

    # 创建 Focal Loss
    criterion = FocalLoss(gamma=2.0)
    loss = criterion(logits, targets)
    print(loss)