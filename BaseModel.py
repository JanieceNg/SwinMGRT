import torch
import sys
import torch.nn as nn
import torch.optim as optim
import os

from torch.distributed.pipeline.sync.checkpoint import checkpoint
from torch.utils.tensorboard import SummaryWriter
from utils import ScaleMetric as Metric
from utils import save_config,TrainRecorder,Test_val,Inference
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from loss import get_loss
from networks.Model import ClassifyModel
from data_process import get_train_dataloader,get_val_dataloader,get_trainset
from tqdm import tqdm

class BaseModel:
    def __init__(self, config,model_config):
        self.config = config
        self.model_config = model_config
        self.device = torch.device(model_config.DEVICE)

        # **日志 & Checkpoint 目录**
        self.save_dir = os.path.join('runs',config.TAG)
        self.checkpoint_dir = os.path.join(self.save_dir, "checkpoints")

        # 记录训练详细信息
        self.log_dir = os.path.join(self.save_dir, "log.json")
        self.recorder = TrainRecorder(log_path=self.log_dir)

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.log_dir), exist_ok=True)

        # 保存配置文件
        save_config(config,os.path.join(self.save_dir,'config.yml'))
        save_config(model_config, os.path.join(self.save_dir, 'model_config.yml'))

        # **训练日志**
        self.history = {
            'train_record': [],
            'val_record': [],
            "best_epoch": 0,
            "best_acc": 0.0,
            "current_epoch": 1,
        }
        # **TensorBoard**
        # self.writer = SummaryWriter(self.log_dir)

        self.trainset = get_trainset(config)

        # **初始化模型**
        self.model = ClassifyModel(model_config).to(self.device)
        self.load_weights(config.LOAD)
        self.optimizer = build_optimizer(config=config, model=self.model, set=model_config.OPTIM_SET,setdir=self.save_dir)
        self.scheduler = build_scheduler(config, self.optimizer, config.TRAIN.LR_SCHEDULER.N_ITER)
        self.criterion = get_loss(config,self.trainset)
        num_classes = model_config.HEAD.SCALE_CLASS+[model_config.HEAD.SCALE_CLASS[-1]] if model_config.HEAD.REFINE.NAME else model_config.HEAD.SCALE_CLASS
        #print(num_classes)
        self.metric = Metric(num_classes=num_classes)

    def load_weights(self,load_dir):
        if load_dir:
            ld = torch.load(load_dir)
            if "model_state_dict" in ld:
                weights_dict = ld["model_state_dict"]
            else:
                weights_dict = ld
            # 删除有关分类类别的权重
            # for k in list(weights_dict.keys()):
            #     if "heads.0" in k:
            #         nk = k.replace('heads.0','heads.2')
            #         weights_dict[nk]=weights_dict[k]
            # for k in list(weights_dict.keys()):
            #     if "heads.2" in k or self.model_config.:
            #         del weights_dict[k]
            self.model.load_state_dict(weights_dict, strict=False)

            print('model load weights from %s' % load_dir)
            #
            # 查看哪些权重成功导入
            loaded_keys = set(self.model.state_dict().keys())
            original_keys = set(weights_dict.keys())

            # 成功导入的权重
            imported_keys = loaded_keys.intersection(original_keys)
            print("成功导入的权重:")
            for key in imported_keys:
                print(key)

            # 未导入的权重（如果有的话）
            missing_keys = original_keys.difference(loaded_keys)
            print("\n未导入的权重:")
            for key in missing_keys:
                print(key)


    def update(self, images, labels, num_iters):
        self.model.train()
        images = images.to(self.device)
        labels = labels.to(self.device).transpose(0,1)
        outputs = self.model(images)

        if isinstance(outputs,(tuple,list)):
            preds = [o.argmax(dim=1).to('cpu') for o in outputs]
        else:
            preds = outputs.argmax(dim=1).to('cpu')

        loss = self.criterion(outputs, labels)

        num_loss = [l.item()  for i,l in enumerate(loss)] #***
        loss = sum(loss)  #***

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step(num_iters)
        self.metric.update(outputs, labels)
        return num_loss, preds  # *** return loss.item()

    def train_epoch(self, train_loader,epoch):
        self.model.train()
        total_loss = [0]
        num_images = 0
        self.metric.reset()
        pass_iters = len(train_loader)*(epoch-1)
        train_loader = tqdm(train_loader)
        for i,sample in enumerate(train_loader):
            # debug
            # if i>5:
            #     break
            images, labels = sample
            n_iter = pass_iters + i + 1
            loss, _ = self.update(images, labels,num_iters=n_iter)
            n = images.size(0)
            if len(total_loss)!=len(loss):
                total_loss = [0]*len(loss)
            total_loss = [t + l * n for l,t in zip(loss,total_loss)]
            num_images += n
            if n_iter % 10==0:
                train_loader.desc=f'train epoch {epoch}, loss: {sum(total_loss)/num_images: .6f}' +', '+ ' '.join(['loss%d: %.6f'%(i+1,l/num_images) for i,l in enumerate(total_loss)])
        epoch_loss = sum(total_loss) / num_images
        metrics = self.metric.compute()
        return epoch_loss, metrics

    def train(self):
        train_dl = get_train_dataloader(self.config)
        assert self.config.TRAIN.LR_SCHEDULER.N_ITER == len(train_dl),f'config.TRAIN.LR_SCHEDULER.N_ITER should be {len(train_dl)}'
        val_dl = get_val_dataloader(self.config)
        start = self.history['current_epoch']
        max_epoch = self.config.TRAIN.EPOCHS

        for epoch in range(start,max_epoch+1):
            epoch_loss,metrics = self.train_epoch(train_dl,epoch)

            res = {"epoch": epoch, "train_loss": epoch_loss, "train_acc": metrics[-1]["Top-1"]}
            self.history['train_record'].append(res)
            self.print_metrics(metrics,'train',epoch)
            self.log_metrics(metrics,'train',epoch)

            if epoch % self.config.MISC.VAL_FREQ == 0:
                val_loss,val_metrics = self.evaluate(val_dl)

                res = {"epoch": epoch, "val_loss": val_loss, "val_acc": val_metrics[-1]["Top-1"]}
                self.history['val_record'].append(res)
                self.print_metrics(val_metrics, 'valid', epoch)
                self.log_metrics(val_metrics, 'val', epoch)

                if val_metrics[-1]['Top-1'] > self.history['best_acc']:
                    self.history['best_acc'] = val_metrics[-1]['Top-1']
                    self.history['best_epoch'] = epoch
                    # 使得resume可以进入下一个epoch
                    self.history['current_epoch'] = epoch + 1
                    self.save_model('best.pth')

            # 使得resume可以进入下一个epoch
            self.history['current_epoch'] = epoch + 1

            # 频率保存
            if epoch % self.config.MISC.SAVE_FREQ == 0 or epoch==max_epoch:
                self.save_model('epoch_%s.pth'%epoch)

            if self.config.MISC.DATA_UPDATE_FREQ > 0 and epoch % self.config.MISC.DATA_UPDATE_FREQ == 0:
                train_dl.dataset.update_paths()
                train_dl = get_train_dataloader(self.config,train_dl.dataset)

            # 临时保存最近epoch
            self.save_model()


    @torch.no_grad()
    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0
        self.metric.reset()
        val_loader = tqdm(val_loader)
        num_images = 0
        for i, sample in enumerate(val_loader):
            #debug
            # if i>5:
            #     break
            images, labels = sample
            images, labels = images.to(self.device), labels.to(self.device).transpose(0,1)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            total_loss += sum(loss).item() * images.size(0)
            num_images += images.size(0)
            self.metric.update(outputs, labels)

        val_loss = total_loss / num_images
        metrics = self.metric.compute()
        # print(metrics)
        return val_loss, metrics

    def test(self,val_loader):
        class_indices = val_loader.dataset.class_indices
        if len(class_indices)>10:
            topk = (1,5)
        else:
            topk=(1,)

        Test_val(self.model, val_loader, class_indices=class_indices, device=self.device, topk=topk,
                       save_csv_path=os.path.join(self.save_dir,"per_class_metrics.csv"), save_cm_path=os.path.join(self.save_dir,"confusion_matrix.png"))

    def inference(self,test_loader,save_url='predict.json'):
        class_indices = test_loader.dataset.class_indices
        scale_class = test_loader.dataset.scale_class
        Inference(self.model,test_loader,class_indices,scale_class,device=self.device,save_path=os.path.join(self.save_dir,save_url))

    def save_model(self, filename="checkpoint.pth"):
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "history": self.history,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"✅ 模型已保存: {checkpoint_path}")

    def load_model(self, filename="checkpoint.pth"):
        if not os.path.dirname(filename):
            checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        else:
            checkpoint_path = filename
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if self.scheduler and checkpoint["scheduler_state_dict"]:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.history = checkpoint["history"]
            print(f"✅ 加载模型成功: {checkpoint_path}")
        else:
            print(f"❌ 未找到 checkpoint: {checkpoint_path}")

    def print_metrics(self, metrics,tag, epoch):
        print('%s epoch: %s, '%(tag,epoch)+" | ".join([", ".join(f"{metric_name}: {value: .3f}" for metric_name, value in m.items()) for m in metrics]))

    def log_metrics(self, metrics,tag, epoch):
        if tag == 'train':
            self.recorder.update(epoch,{"Loss/%s"%tag: self.history["train_record"][-1]['train_loss']} )
        elif tag == 'val':
            self.recorder.update(epoch,{"Loss/%s"%tag: self.history["val_record"][-1]['val_loss']})
        else:
            RuntimeError('tag %s not exist'%tag)

        for i,m in enumerate(metrics):
            for k,v in m.items():
                name = 'Head_%d/%s/%s'%(i,k,tag)
                self.recorder.update(epoch,{name:v})

        self.recorder.save()
