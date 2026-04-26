import torch
from repro_uda.baseline_model import YOLOv5s

def debug_mapping(weights_path):
    device = torch.device('cpu')
    model = YOLOv5s(nc=1)
    
    ckpt = torch.load(weights_path, map_location='cpu', weights_only=False)
    if 'model' in ckpt:
        model_obj = ckpt['model']
        if hasattr(model_obj, 'state_dict'):
            state_dict = model_obj.state_dict()
        else:
            state_dict = getattr(model_obj, 'model', model_obj).state_dict()
    else:
        state_dict = ckpt
        
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            parts = k.split('.')
            if len(parts) >= 2 and parts[1].isdigit():
                layer_idx = parts[1]
                new_k = f"l{layer_idx}." + ".".join(parts[2:])
                new_state_dict[new_k] = v
    
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print(f"Missing count: {len(missing)}")
    print(f"Missing samples: {missing[:10]}")
    print(f"Unexpected samples count: {len(unexpected)}")

if __name__ == "__main__":
    import sys
    debug_mapping(sys.argv[1])
