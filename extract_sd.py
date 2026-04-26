import torch
import torch.nn as nn
import sys
import os

# Ensure the project root is in path so 'models' is found
sys.path.append(os.getcwd())

def extract_state_dict(path):
    try:
        # Now it should find 'models.yolo.Model' etc. from the stub files
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        print(f"Loaded successfully! Keys: {ckpt.keys()}")
        
        if 'model' in ckpt:
            model = ckpt['model']
            # Standard YOLOv5 checkpoint stores the state dict in the model object
            # Our stub Model class inherits from nn.Module, so it has a __dict__ 
            # where the unpickler put everything.
            
            # Let's see if we can get the state_dict from the stub object
            try:
                sd = model.state_dict()
                print(f"Extracted state_dict with {len(sd)} keys")
                torch.save(sd, "yolov5s_state_dict.pt")
                print("Saved to yolov5s_state_dict.pt")
            except Exception as e:
                print(f"Error calling state_dict(): {e}")
                # Fallback: look at the model's __dict__
                print(f"Manual keys in model: {model.__dict__.keys()}")
                
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    extract_state_dict(sys.argv[1])
