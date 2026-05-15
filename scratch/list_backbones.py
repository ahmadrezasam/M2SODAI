from mmdet.models.builder import BACKBONES
print("REGISTERED BACKBONES:")
for name in BACKBONES.module_dict.keys():
    print(name)
