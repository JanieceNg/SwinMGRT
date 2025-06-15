from options import opt, config
from data_process import ClassifyDataset
from data_process import get_train_dataloader,get_val_dataloader,get_test_dataloader
from utils import timing_and_memory
from tqdm import tqdm

@timing_and_memory
def load_data(dl,debug=True):
    for i,sample in enumerate(tqdm(dl)):
        if debug and i>5:
            print(len(sample))
            print(sample[0].shape)
            print(sample[1].shape)
            break
        # img,labels = sample
        # dl.desc=str(img.shape)+str([lab.shape for lab in labels])

if __name__ == '__main__':
    tdl = get_train_dataloader(config)

    vdl = get_val_dataloader(config)
    ttdl = get_test_dataloader(config)
    n = len(tdl.dataset)+len(vdl.dataset)

    print(len(tdl.dataset),len(vdl.dataset),n)
    print(len(tdl.dataset)/n,len(vdl.dataset)/n)