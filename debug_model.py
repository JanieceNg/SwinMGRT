from options import opt, config, model_config
from BaseModel import BaseModel
import torch
from data_process import get_train_dataloader,get_val_dataloader

def show_params(model):
    total_params = 0
    print("👀 Layer-wise Parameter Count:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            print(f"  🔹 {name:<50} | Params: {num_params:,}")
            total_params += num_params
    print(f"\n🧠 Total Parameters: {total_params:,}")

def print_model_structure(model, max_depth=2, prefix='', depth=0):
    if depth > max_depth:
        return

    for name, module in model.named_children():
        print(f"{'  '*depth}{prefix}{name}: {module.__class__.__name__}")
        print_model_structure(module, max_depth, prefix='', depth=depth+1)

def get_keys_at_depth(model, target_depth=2):
    keys = []

    def traverse(module, prefix='', depth=0):
        if depth == target_depth:
            keys.append(prefix.rstrip('.'))
            return
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            traverse(child, full_name, depth + 1)

    traverse(model)
    return keys



if __name__ == '__main__':
    print(config)
    model = BaseModel(config,model_config)
    # depth_2_keys = get_keys_at_depth(model.model, target_depth=4)

    # for k in depth_2_keys:
    #     print(k)
    x = torch.rand(2,3,224,224).to(model.device)
    fs = model.model.backbone(x)
    print([f.shape for f in fs])
    # print(model.criterion)