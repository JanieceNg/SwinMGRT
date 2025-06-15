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
#python train.py --config config/plant500/plant5002_batch32_imgsz224_scale012x_w118.yml --model_config config/model/swinTb_nonerule_null_p25p5b_drop01_in223.yml --gpu_id 1 --resume
# python train.py --config config/plant500/plant5000_batch16_imgsz224_scale012R_w1111_mgtb.yml --model_config config/model/swinMGTb_freeze_null_p25p5bR_in223.yml --gpu_id 1 --resume
#python train.py --config config/plant500/plant500_batch16_imgsz224_scalexx2x_baseline_ep100.yml --model_config config/model/resnet101_null_s_baseline.yml --gpu_id 1 --resume

# swin 消融
# python eval.py --config config/plant500/plant500_batch16_imgsz224_scalexx2x_baseline.yml --model_config config/model/swinTb_nonerule_null_s_baseline.yml
#python test.py --config config/plant500/plant500_batch16_imgsz224_scalexx2x_baseline.yml --model_config config/model/swinTb_nonerule_null_b.yml
#python test.py --config config/plant500/plant500_batch16_imgsz224_scale012x_w111.yml --model_config config/model/swinTb_nonerule_null_p25p5b_drop01_in223.yml
#python test.py --config config/plant500/plant500_batch16_imgsz224_scale012x_w111_mgtb_update1.yml --model_config config/model/swinMGTb_nonerule_null_p25p5b_in223.yml
# python test.py --config config/plant500/plant500_batch16_imgsz224_scale012R_w1111_mgtb_update1.yml --model_config config/model/swinMGTb_refine_null_p25p5bR_in223.yml
# python eval.py --config config/plant500/plant500_batch16_imgsz224_scalexx2x_baseline.yml --model_config config/model/swinTb_nonerule_null_b.yml
# python eval.py --config config/plant500/plant500_batch16_imgsz224_scale012R_w1111_mgtb_update1.yml --model_config config/model/swinMGTb_refine_null_p25p5bR_in223.yml


# 对比实验
# python test.py --config config/plant500/plant500_batch16_imgsz224_scalexx2x_baseline.yml --model_config config/model/efficientnetv2l_null_s_baseline.yml
# python test.py --config config/plant500/plant500_batch16_imgsz224_scalexx2x_baseline_ep60.yml --model_config config/model/resnet101_null_s_baseline.yml
# python test.py --config config/plant500/plant500_batch16_imgsz224_scalexx2x_baseline_ep60.yml --model_config config/model/resnext101_null_s_baseline.yml
# python test.py --config config/plant500/plant500_batch16_imgsz224_scalexx2x_baseline.yml --model_config config/model/davit_null_s_baseline.yml
# python test.py --config config/plant500/plant500_batch16_imgsz224_scalexx2x_baseline.yml --model_config config/model/vitbase_null_s_baseline.yml
# python test.py --config config/plant500/plant500_batch16_imgsz224_scalexx2x_baseline.yml --model_config config/model/swinTb_nonerule_null_s_baseline.yml

# refine超参数
#1902324
#python eval.py --config config/plant500/plant500_batch16_imgsz224_scalexx2R.yml --model_config config/model/swinTb_refine_null_sR_top5.yml --gpu_id 0
#python eval.py --config config/plant500/plant500_batch16_imgsz224_scalexx2R.yml --model_config config/model/swinTb_refine_null_sR.yml --gpu_id 0
#python eval.py --config config/plant500/plant500_batch16_imgsz224_scalexx2R.yml --model_config config/model/swinTb_refine_null_sR_top20.yml --gpu_id 0
##1374964
#python eval.py --config config/plant500/plant500_batch16_imgsz224_scalexx2R.yml --model_config config/model/swinTb_refine_null_sR_layer2.yml --gpu_id 0
##2429684
#python eval.py --config config/plant500/plant500_batch16_imgsz224_scalexx2R.yml --model_config config/model/swinTb_refine_null_sR_layer6.yml --gpu_id 0
#
#python eval.py --config config/plant500/plant500_batch16_imgsz224_scalexx2x_baseline.yml --model_config config/model/swinTb_nonerule_null_s_baseline.yml
#python test.py --config config/plant500/plant500_batch16_imgsz224_scalexx2x_baseline.yml --model_config config/model/swinTb_nonerule_null_s_baseline.yml

# python eval.py --config config/plant500/plant500_batch32_imgsz224_scalexx2x_test.yml --model_config config/model/resnet101_null_s_baseline.yml --gpu_id 0
python test.py --config config/plant500/plant500_batch16_imgsz224_scale012x_w111.yml --model_config config/model/efficientnetv2l_nonerule_null_p25p5b_in223.yml --gpu_id 1




