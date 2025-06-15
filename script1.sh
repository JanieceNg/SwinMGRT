#python train.py --config config/cub200_batch64_imgsz224_scale012x.yml --model_config config/swinTb_finetune_null_sss.yml --gpu_id 1 --resume
#sleep 300
#python train.py --config config/plant560_batch64_imgsz224_scale012x.yml --model_config config/swinTb_finetune_null_sss.yml --gpu_id 1
# python train.py --config config/cub200_batch32_imgsz224_scalexx2x_crossloss_baseline.yml --model_config config/swinTb_finetune_null_s_baseline.yml --gpu_id 0 --resume
# sleep 300
# python train.py --config config/plant560_batch64_imgsz224_scalexx2x_crossloss_baseline.yml --model_config config/swinTb_finetune_null_s_baseline.yml --gpu_id 0
#python train.py --config config/plant500/plant500_batch32_imgsz224_scale012x_w226.yml --model_config config/model/swinTb_nonerule_null_ssb.yml --gpu_id 1
#sleep 200
#python train.py --config config/plant500/plant500_batch32_imgsz224_scale012x_w235.yml --model_config config/model/swinTb_nonerule_null_ssb.yml --gpu_id 1
#sleep 200
#python train.py --config config/plant500/plant500_batch32_imgsz224_scale012x_w334.yml --model_config config/model/swinTb_nonerule_null_ssb.yml --gpu_id 1

# python train.py --config config/plant500/plant5000_batch16_imgsz224_scale012R_w1111_mgtb.yml --model_config config/model/swinMGTb_nonerule_null_p25p5bR_in223.yml --gpu_id 0

# 新实验
# python train.py --config config/plant500/plant500_batch16_imgsz224_scalexx2x_baseline.yml --model_config config/model/davit_null_s_baseline.yml --gpu_id 0

# python train.py --config config/plant500/plant500_batch16_imgsz224_scalexx2R.yml --model_config config/model/swinTb_refine_null_sR.yml --gpu_id 0
# python train.py --config config/plant500/plant500_batch16_imgsz224_scalexx2R.yml --model_config config/model/swinTb_refine_null_sR_top5.yml --gpu_id 0
# python train.py --config config/plant500/plant500_batch16_imgsz224_scalexx2R.yml --model_config config/model/swinTb_refine_null_sR_top20.yml --gpu_id 0
# python train.py --config config/plant500/plant500_batch16_imgsz224_scalexx2R.yml --model_config config/model/swinTb_refine_null_sR_layer2.yml --gpu_id 0
# python train.py --config config/plant500/plant500_batch16_imgsz224_scalexx2R.yml --model_config config/model/swinTb_refine_null_sR_layer6.yml --gpu_id 0

#python train.py --config config/plant500/plant500_batch32_imgsz224_scalexx2x_test.yml --model_config config/model/resnet101_null_s_baseline.yml --gpu_id 0 --resume

#python train.py --config config/plant500/plant500_batch16_imgsz224_scalexx2x_baseline.yml --model_config config/model/resnext101_null_s_baseline.yml --gpu_id 0

python train.py --config config/plant500/plant500_batch64_imgsz224_scalexx2x_test_ep100.yml --model_config config/model/resnet18_null_s_baseline.yml --gpu_id 0 --resume
