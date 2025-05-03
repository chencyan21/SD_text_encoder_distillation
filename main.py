import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer
from datasets import DatasetDict, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import uuid
from model import create_student_model, create_student_model_config, create_teacher_model
from safetensors.torch import load_file
from data import load_captions_dataset, preprocess_data
from utils import intermediate_loss
from train import distill_train
# 设置设备
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

# 初始化分词器和模型
teacher_model, tokenizer = create_teacher_model("stabilityai/stable-diffusion-2-1", device)
student_model = create_student_model(create_student_model_config(), use_half_precision=False, device=device)

# 冻结教师模型参数
for param in teacher_model.parameters():
    param.requires_grad = False


# 加载数据集
dataset_dict = load_captions_dataset()
# 处理数据集
train_dataloader, val_dataloader = preprocess_data(dataset_dict, 256,tokenizer)


# 训练
student_model.load_state_dict(
    load_file("student_model_epoch33_28d94d94-36e8-4a7b-b717-2f2baae003d8/model.safetensors", device="cuda:4")
)
distill_train(
    teacher_model=teacher_model,
    student_model=student_model,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    epochs=10,
    lr=1e-5,
    device=device
)
