from timm import create_model
import timm
ml = timm.list_models()
for i in ml:
    print(i)