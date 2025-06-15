from options import opt, config, model_config
from data_process import get_val_dataloader,get_test_dataloader
from BaseModel import BaseModel
def show_params(model):
    total_params = 0
    print("👀 Layer-wise Parameter Count:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            total_params += num_params
    print(f"\n🧠 Total Parameters: {total_params:,}")

if __name__ == '__main__':
    print(config)
    model = BaseModel(config,model_config)
    show_params(model.model)
    checkpoint_path = opt.load if opt.load else 'best.pth'
    model.load_model(checkpoint_path)
    test_dl = get_test_dataloader(config)
    model.inference(test_dl)
