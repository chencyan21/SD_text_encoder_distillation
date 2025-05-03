from transformers import CLIPTextModel, CLIPTextConfig
import torch
import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from PIL import Image
import numpy as np
import gc

# # 创建教师模型
# def create_teacher_model(model_id, device):
#     # 加载模型
#     pipe = StableDiffusionPipeline.from_pretrained(
#         model_id, torch_dtype=torch.float16
#     ).to(device)
#     teacher_model = pipe.text_encoder
#     tokenizer = pipe.tokenizer
#     # 教师模型输出有两个：last hidden state和pooler output
#     # 1. last hidden state：形状为[batch_size, max_length, hidden_size]
#     # 2. pooler output：形状为[batch_size, hidden_size]
#     return teacher_model,tokenizer


# 创建教师模型，优化显存
def create_teacher_model(model_id, device):
    # 加载StableDiffusionPipeline，仅需text_encoder和tokenizer
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float32
    ).to(device)

    # 提取需要的组件
    teacher_model = pipe.text_encoder
    tokenizer = pipe.tokenizer

    # 显式删除不需要的组件
    del pipe.vae
    del pipe.unet
    del pipe.safety_checker  # 如果有
    del pipe.feature_extractor  # 如果有
    del pipe.scheduler
    del pipe

    # 清理内存
    gc.collect()
    torch.cuda.empty_cache()

    # 返回教师模型和分词器
    return teacher_model, tokenizer


def create_student_model_config(
    vocab_size=49408,
    max_position_embeddings=77,
    hidden_size=1024,
    intermediate_size=2048,
    num_hidden_layers=12,
    num_attention_heads=8,
    hidden_act="gelu",
    layer_norm_eps=1e-5,
    projection_dim=1024,
):
    # 定义学生模型的配置
    student_config = CLIPTextConfig(
        vocab_size=vocab_size,  # 与教师模型相同的词汇表大小
        max_position_embeddings=max_position_embeddings,  # 与教师模型相同的最大序列长度
        hidden_size=hidden_size,  # 设置隐藏维度为1024，与教师模型一致
        intermediate_size=intermediate_size,  # 减小MLP中间层维度（教师模型为4096）
        num_hidden_layers=num_hidden_layers,  # 减少Transformer层数（教师模型为23）
        num_attention_heads=num_attention_heads,  # 减少注意力头数（教师模型可能为16）
        hidden_act=hidden_act,  # 保持激活函数
        layer_norm_eps=layer_norm_eps,  # 保持LayerNorm参数
        projection_dim=projection_dim,  # 设置投影维度为1024，与教师模型一致
    )
    return student_config


# 创建学生模型
def create_student_model(
    student_config,
    use_half_precision=True,
    device="cuda:4" if torch.cuda.is_available() else "cpu",
):
    # 初始化学生模型
    student_model = CLIPTextModel(student_config)
    # 转换为float16精度（如果需要）
    if use_half_precision:
        student_model = student_model.half()

    # 移动到指定设备
    student_model = student_model.to(device)

    return student_model
