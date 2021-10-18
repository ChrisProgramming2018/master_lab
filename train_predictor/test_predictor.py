from taskonomy_network import TaskonomyEncoder, TaskonomyDecoder, TaskonomyNetwork,  TASKS_TO_CHANNELS
from models import VisualPrior, VisualPriorRepresentation
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from replay_buffer_depth import ReplayBufferDepth
import cv2
import numpy as np


size = 256
memory_small = ReplayBufferDepth((size, size), (size,size,3), (size, size, 3), 51, "cuda")

memory_small.load_memory_normals("../data//sim_real_buffer-valid")

#memory_small.test_surface_normals(42)

model = TaskonomyNetwork()
#model.load_model("trained_real_world_model/model_step_1575_eval_loss_21170.2050781250")
# model.load_model("trained_real_world_model/model_step_150_eval_loss_21182.2265625000") 
model.load_model("trained_realsim_normal_model/model_step_1500_eval_loss_21170.3925781250") 
#model.load_model("trained_realsim_normal_model/model_step_150_eval_loss_21182.4824218750") 

print("model loaded")

device = "cuda"
model.to(device)
RGB_state = memory_small.obses[42]
cv2.imshow("RGB_image", RGB_state)
cv2.waitKey(0)
# RGB_state = torch.as_tensor(RGB_state, device="cuda").float().transpose(1,2,0)
RGB_state = TF.to_tensor(RGB_state).to(device)
RGB_state = RGB_state.unsqueeze(0)

print("shape", RGB_state.shape)

outPut = model(RGB_state).cpu().numpy()
print(outPut.shape)
outPut = np.array(outPut.squeeze(0))
print(outPut.shape)
print(outPut)
#outPut = outPut.permute(1,2,0)
outPut = outPut.transpose(1,2,0)

print(outPut.shape)
cv2.imshow("RGB_image", outPut * 255)
cv2.waitKey(0)
