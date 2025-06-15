from options import opt, config, model_config
from BaseModel import BaseModel
# def show_params(model):
#     total_params = 0
#     print("👀 Layer-wise Parameter Count:")
#     for name, param in model.named_parameters():
#         if param.requires_grad:
#             print(name)
#             num_params = param.numel()
#             total_params += num_params
#     print(f"\n🧠 Total Parameters: {total_params:,}")

def print_trainable_parameters(model):
    print(model)
    total = 0
    trainable = 0
    for name, param in model.named_parameters():
        param_count = param.numel()
        total += param_count
        if param.requires_grad:
            trainable += param_count
            print(f"[Trainable] {name:<60} ({param_count:,} parameters)")
        else:
            print(f"[Frozen   ] {name:<60} ({param_count:,} parameters)")

    print(f"\n�� Total parameters: {total:,}")
    print(f"✅ Trainable parameters: {trainable:,} ({100 * trainable / total:.2f}%)")

if __name__ == '__main__':
    print(config)
    model = BaseModel(config,model_config)
    print_trainable_parameters(model.model)
    if opt.resume:
        checkpoint_path = opt.load if opt.load else 'checkpoint.pth'
        model.load_model(checkpoint_path)
    model.train()