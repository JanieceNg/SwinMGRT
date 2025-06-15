
import matplotlib.pyplot as plt
from utils import TrainRecorder
#
# log_dirs = ['runs/plant500_batch32_imgsz224_scalexx2x_baseline_swinTb_finetune_null_s_baseline/log.json',
#             'runs/plant500_batch32_imgsz224_scalexx2x_baseline_swinTb_finetune_null_b/log.json',
#             'runs/plant500_batch32_imgsz224_scalexx2x_baseline_swinTb_nonerule_null_b/log.json',
#             'runs/plant5003_batch32_imgsz224_scalexx2x_baseline_swinTb_nonerule_null_b/log.json',
#             'runs/plant5006_batch32_imgsz224_scalexx2x_baseline_swinTb_nonerule_null_b/log.json']
# log_dirs = ['runs/plant500_batch32_imgsz224_scalexx2x_crossloss_baseline_swinTb_nonerule_null_b/log.json',
#             'runs/plant500_batch32_imgsz224_scalexx2x_crossloss_baseline_swinTb_nonerule_null_s_baseline/log.json']
# log_dirs = ['runs/plant500_batch32_imgsz224_scale012x_w235_swinTb_nonerule_null_ssb/log.json',
#             'runs/plant500_batch32_imgsz224_scale012x_w226_swinTb_nonerule_null_ssb/log.json',
#             'runs/plant500_batch32_imgsz224_scale012x_w111_swinTb_nonerule_null_ssb/log.json',
#             'runs/plant500_batch32_imgsz224_scale012x_w118_swinTb_nonerule_null_ssb/log.json']

log_dirs = []

log_dirs = ['runs/plant500_batch16_imgsz224_scalexx2x_baseline_davit_null_s_baseline/log.json',
            'runs/plant500_batch16_imgsz224_scalexx2x_baseline_efficientnetv2l_null_s_baseline/log.json',
            'runs/plant500_batch16_imgsz224_scalexx2x_baseline_ep60_vgg19_null_s_baseline//log.json',
            'runs/plant500_batch32_imgsz224_scalexx2x_baseline_resnext101_null_s_baseline/log.json',
            'runs/plant500_batch16_imgsz224_scalexx2x_baseline_ep100_resnet101_null_s_baseline/log.json',
            "runs/plant500_batch16_imgsz224_scalexx2x_baseline_vitbase_null_s_baseline/log.json"]

keys = ['Head_0/Top-1/val',
        'Head_0/Top-1/val',
        'Head_0/Top-1/val',
        'Head_0/Top-1/val',
        'Head_0/Top-1/val',
        'Head_0/Top-1/val']


for log_dir,key in zip(log_dirs,keys):
    print(log_dir)
    rec = TrainRecorder(log_path=log_dir)
    rec.check_best(key)


log_dir1 = log_dirs[0]
key1 = keys[0]
rec1 = TrainRecorder(log_path=log_dir1)
log_dir2 = log_dirs[1]
key2 = keys[1]
rec2 = TrainRecorder(log_path=log_dir2)
# 示例数据（你可以替换为自己的训练精度数据）
epochs = list(range(1, 31))  # 训练了30个epoch

accuracy1 = [rec1.logs[str(i)][key1] for i in epochs]
accuracy2 = [rec2.logs[str(i)][key2] for i in epochs]


print('debug: ',accuracy1,accuracy2)

# 创建图形
plt.figure(figsize=(10, 6))
plt.plot(epochs, accuracy1, label='Model 1 Accuracy', marker='o')
plt.plot(epochs, accuracy2, label='Model 2 Accuracy', marker='s')

# 添加标题和标签
plt.title('Training Accuracy Comparison')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(50,100)
plt.grid(True)
plt.legend()

# 保存为PNG文件
plt.savefig('accuracy_comparison.png', dpi=300)

