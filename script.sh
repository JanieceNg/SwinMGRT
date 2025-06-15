# python train.py --config config/cub200_batch32_imgsz224_scalexx2x_crossloss_baseline.yml --model_config config/swinTb_finetune_null_s_baseline.yml --gpu_id 0 --resume
# sleep 300
# python train.py --config config/plant560_batch64_imgsz224_scalexx2x_crossloss_baseline.yml --model_config config/swinTb_finetune_null_s_baseline.yml --gpu_id 0
#python train.py --config config/plant500/plant500_batch32_imgsz224_scale012x_w111.yml --model_config config/model/swinTb_nonerule_null_ssb.yml --gpu_id 0
#sleep 200
#python train.py --config config/plant500/plant500_batch32_imgsz224_scale012x_w118.yml --model_config config/model/swinTb_nonerule_null_ssb.yml --gpu_id 0
#sleep 200
#python train.py --config config/plant500/plant500_batch32_imgsz224_scale012x_w127.yml --model_config config/model/swinTb_nonerule_null_ssb.yml --gpu_id 0
#python train.py --config config/cub200/cub200_batch32_imgsz224_scalexx2x_baseline.yml --model_config config/model/swinTb_finetune_null_s_baseline.yml --gpu_id 0
#sleep 200
#python train.py --config config/cub200/cub200_batch32_imgsz224_scalexx2x_baseline.yml --model_config config/model/swinTb_finetune_null_b.yml --gpu_id 0
# python train.py --config config/plant500/plant500_batch16_imgsz224_scale012x_w111.yml --model_config config/model/swinTb_nonerule_null_p25p5s_drop01_in223.yml --gpu_id 0

#python train.py --config config/plant500/plant500_batch32_imgsz224_scalexx2x_baseline.yml --model_config config/model/vitbase_null_s_baseline.yml --gpu_id 0
#python train.py --config config/plant500/plant500_batch16_imgsz224_scalexx2x_baseline.yml --model_config config/model/swinTb_nonerule_null_s_baseline.yml --gpu_id 0


#新实验
# python train.py --config config/plant500/plant500_batch16_imgsz224_scale012x_w111.yml --model_config config/model/efficientnetv2l_nonerule_null_p25p5b_in223.yml --gpu_id 1
# python train.py --config config/plant500/plant500_batch16_imgsz224_scalexx2x_baseline_ep100.yml --model_config config/model/resnext101_null_s_baseline.yml --gpu_id 0 --resume
# python train.py --config config/plant500/plant500_batch16_imgsz224_scale012x_w111_ep100.yml --model_config config/model/resnet101_nonerule_null_p25p5b_in223.yml --gpu_id 1
python train.py --config config/plant500/plant500_batch16_imgsz224_scalexx2x_baseline_ep60.yml --model_config config/model/resnext101_null_s_baseline.yml --gpu_id 1 --resume

