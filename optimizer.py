import torch
import yaml
import os
import numpy as np

def build_optimizer(config,model,set,setdir=None):
    base_lr = config.TRAIN.BASE_LR
    weight_decay = config.TRAIN.WEIGHT_DECAY
    no_decay_modules = ['bias',"norm"]
    if set == 'nonerule':
        optimizer_params = model.parameters()
    else:
        rule = get_rule(set)
        optimizer_params = get_optimizer_params(model,base_lr,weight_decay,rule,no_decay_modules,setdir)

    # 选择优化器
    optimizer_type = config.TRAIN.OPTIMIZER.NAME
    if optimizer_type.lower() == "adamw":
        return torch.optim.AdamW(optimizer_params, lr=base_lr,betas=config.TRAIN.OPTIMIZER.BETAS)
    elif optimizer_type.lower() == "sgd":
        return torch.optim.SGD(optimizer_params, lr=base_lr,momentum=config.TRAIN.OPTIMIZER.MOMENTUM)
    else:
        raise ValueError(f"Unsupported optimizer_ type: {optimizer_type}")


def get_rule(set):
    # print('debug:',set)
    if set == 'None':
        rule = {}
    elif set == 'swin_b':
        rule = {
            "patch_embed": { "name": ["backbone.patch_embed"],"scale": 0.1, "weight_decay": 1e-6},  # Patch embedding 层，较低学习率
            "layers.0": {"name": ["backbone.layers_0"],"scale": 0.2},  # 第一阶段 Swin Block，较低学习率
            "layers.1": {"name": ["backbone.layers_1"],"scale": 0.3},  # 第二阶段 Swin Block，中等学习率
            "layers.2": {"name": ["backbone.layers_2"],"scale": 0.5},  # 第三阶段 Swin Block，稍高学习率
            "layers.3": {"name": ["backbone.layers_3"],"scale": 0.8},  # 最后一阶段 Swin Block，更高学习率
            "head": {
                "name": ["head"],
                "scale": 2.0,
                "weight_decay": 1e-4
            }  # 分类头部，高学习率
        }
    elif set == 'swinT_b':
        rule = {"patch_embed": { "name": ["backbone.patch_embed"],"scale": 0.1, "weight_decay": 1e-6},  # Patch embedding 层，较低学习率
                "head": {
                    "name": ["head"],
                    "scale": 2.0,
                    "weight_decay": 1e-4
                }
            }
    elif set == 'swinT_b_finetune':
        rule = {
            "patch_embed": {"name": ["backbone.patch_embed"], "scale": 0.1, "weight_decay": 1e-6},
            # Patch embedding 层，较低学习率
            "layers.0": {"name": ["backbone.layers.0"], "scale": 0.2},  # 第一阶段 Swin Block，较低学习率
            "layers.1": {"name": ["backbone.layers.1"], "scale": 0.3},  # 第二阶段 Swin Block，中等学习率
            "layers.2": {"name": ["backbone.layers.2"], "scale": 0.5},  # 第三阶段 Swin Block，稍高学习率
            "layers.3": {"name": ["backbone.layers.3"], "scale": 0.8},  # 最后一阶段 Swin Block，更高学习率
            "head": {
                "name": ["head"],
                "scale": 2.0,
                "weight_decay": 1e-4
            }  # 分类头部，高学习率
        }
    elif set == 'swinT_b_freeze':
        rule = {
            "patch_embed": {"name": ["backbone.patch_embed"], "freeze": True},
            # Patch embedding 层，较低学习率
            "layers.0": {"name": ["backbone.layers.0"], "freeze":True},  # 第一阶段 Swin Block，较低学习率
            "layers.1": {"name": ["backbone.layers.1"], "freeze":True},  # 第二阶段 Swin Block，中等学习率
            "layers.2": {"name": ["backbone.layers.2"], "freeze":True},  # 第三阶段 Swin Block，稍高学习率
            # "layers.3": {"name": ["backbone.layers.3"], "freeze": True},  # 最后一阶段 Swin Block，更高学习率
            "head": {
                "name": ["head"],
                "scale": 2.0,
                "weight_decay": 1e-4
            }  # 分类头部，高学习率
        }
    elif set == 'swinT_b_hafloss':
        rule = {
            "patch_embed": {"name": ["backbone.patch_embed"], "scale": 0.01, "weight_decay": 1e-6},
            # Patch embedding 层，较低学习率
            "layers.0": {"name": ["backbone.layers.0"], "scale": 0.02},  # 第一阶段 Swin Block，较低学习率
            "layers.1": {"name": ["backbone.layers.1"], "scale": 0.03},  # 第二阶段 Swin Block，中等学习率
            "layers.2": {"name": ["backbone.layers.2"], "scale": 0.05},  # 第三阶段 Swin Block，稍高学习率
            "layers.3": {"name": ["backbone.layers.3"], "scale": 0.08},  # 最后一阶段 Swin Block，更高学习率
            "head": {
                "name": ["head"],
                "scale": 2.0,
                "weight_decay": 1e-4
            }  # 分类头部，高学习率
        }
    elif set == 'swin_MGT_b':
        rule = {"swin_backbone": {"name": ["backbone.patch_embed",
                                           "backbone.layers.0",
                                           "backbone.layers.1",
                                           "backbone.layers.2",], "freeze": True},
            "layers.3": {"name": ["backbone.layers.3"], "scale": 0.08},  # 最后一阶段 Swin Block，更高学习率
            "head": {
                "name": ["head"],
                "scale": 2.0,
                "weight_decay": 1e-4
            }  # 分类头
                }
    elif set == 'refine':
        rule = {"backbone": {"name": ["backbone"], "freeze": True}, # 最后一阶段 Swin Block，更高学习率
            "head": {
                "name": ["head.heads"],"freeze":True,
            },  # 分类头
            "refine":{
                "name":["head.refine"],"scale":2.0,"weight_decay": 1e-4
            }
                }
    else:
        print('-----------------')
        RuntimeError('can not recognize rule %s'%set)
    return rule

def get_optimizer_params(model, base_lr=1e-4, weight_decay=1e-5, special_rules=None, no_decay_modules=None,setdir=None):
    """
    创建优化器，支持:
    - `base_lr` 作为全局基准学习率
    - `special_rules` 通过 `scale` 调整各个模块的学习率
    - `no_decay_modules` 设定哪些参数不进行权重衰减
    - 选择 `AdamW` 或 `SGD`

    Args:
        model (torch.nn.Module): 目标模型
        base_lr (float): 默认学习率
        weight_decay (float): 默认权重衰减
        special_rules (dict): 各模块的学习率 scale 和权重衰减，如:
            {
                "backbone": {"scale": 0.1, "weight_decay": 1e-6},
                "transformer": {"scale": 0.5, "weight_decay": 1e-4},
                "classifier": {"scale": 2.0, "freeze": True}
            }
        no_decay_modules (list): 不做权重衰减的模块，如 ["bias", "LayerNorm"]
        optimizer_type (str): "adamw" 或 "sgd"

    Returns:
        torch.optim.Optimizer: PyTorch 优化器
    """

    special_rules = special_rules if special_rules else {}
    no_decay_modules = no_decay_modules if no_decay_modules else ["norm",'Norm']

    param_groups = {"decay":{}, "no_decay": {}}
    param_name_groups = {"decay": {}, "no_decay": {}}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # 跳过冻结参数

        # 应用特殊规则
        if any(m in name for m in no_decay_modules):
            k = 'no_decay'
        else:
            k = 'decay'

        ruled = False
        for rule_name, settings in special_rules.items():
            for module_name in special_rules[rule_name]['name']:
                if module_name == name[:len(module_name)]:
                    ruled =True
                    if settings.get("freeze", False):
                        param.requires_grad = False
                        break
                    if rule_name not in param_groups[k]:
                        scale = settings.get("scale", 1.0)  # 默认 scale = 1.0
                        lr = base_lr * scale  # 计算学习率
                        wd = settings.get("weight_decay", weight_decay) if k=='decay' else 0.0
                        param_groups[k][rule_name]={'params':[],'weight_decay':wd,'lr':lr}
                        param_name_groups[k][rule_name] = []#{'names': [], 'weight_decay': wd, 'lr': lr}

                    param_groups[k][rule_name]['params'].append(param)
                    param_name_groups[k][rule_name].append(name)#['names'].append(name)

        if not ruled:
            rule_name = 'unruled'
            if rule_name not in param_groups[k]:
                lr = base_lr
                wd = weight_decay if k == 'decay' else 0.0
                param_groups[k][rule_name] = {'params': [], 'weight_decay': wd, 'lr': lr}
                param_name_groups[k][rule_name] = []#{'names': [], 'weight_decay': wd, 'lr': lr}

            param_groups[k][rule_name]['params'].append(param)
            param_name_groups[k][rule_name].append(name)#['names'].append(name)

    # 将字典转换为YAML格式的字符串


    # 将YAML字符串写入文件
    if setdir:
        with open(os.path.join(setdir,'params_optimizer_set.yaml'), 'w') as file:
            yaml_str = yaml.dump(param_name_groups)
            file.write(yaml_str)

    # 整合参数组
    return list(param_groups["decay"].values()) + list(param_groups["no_decay"].values())




