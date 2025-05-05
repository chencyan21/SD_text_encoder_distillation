import os
import time
import numpy as np
import timm
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from diffusers import StableDiffusionPipeline
from model import create_student_model_config, create_student_model
from safetensors.torch import load_file
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm


def intermediate_loss(student_hidden, teacher_hidden):
    return nn.MSELoss()(student_hidden, teacher_hidden)


def generate_and_save_image(pipeline, prompt, filename):
    start_time = time.time()
    image = pipeline(prompt).images[0]
    end_time = time.time()
    image.save(filename)
    return end_time - start_time


def generate_all_images(pipeline, dataset_dict, text_encoder_name):
    # 确保images文件夹存在
    os.makedirs(f"{text_encoder_name}_images", exist_ok=True)
    total_inference_time = 0.0
    prompts = dataset_dict["caption"]
    for i, prompt in enumerate(prompts):
        # 在 images 文件夹中保存图片
        filename = f"{text_encoder_name}_images/image_{i}.png"
        inference_time = generate_and_save_image(pipeline, prompt, filename)
        total_inference_time += inference_time
        print(f"Saved image {i} to {filename}")

    print(f"Total inference time: {total_inference_time:.2f} seconds")
    print(
        f"Average inference time per image: {total_inference_time / len(prompts):.2f} seconds"
    )
    return total_inference_time


def create_pipeline(text_encoder_name, device="cuda"):
    model_id = "stabilityai/stable-diffusion-2-1"
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).to(device)
    student_config = create_student_model_config()
    if text_encoder_name == "stu_model":
        stu_model = create_student_model(
            student_config, use_half_precision=True, device=device
        )
        stu_model.load_state_dict(
            load_file(
                f"student_model_epoch40_e457a80a-6f61-40e3-987f-d3d2f655d7f7/model.safetensors",
                device=device,
            )
        )
        pipeline.text_encoder = stu_model
        # print("Student model parameters count:", sum(p.numel() for p in stu_model.parameters()))
    elif text_encoder_name == "tea_model":
        # print("Teacher model parameters count:", sum(p.numel() for p in pipeline.text_encoder.parameters()))
        pass
    pipeline = pipeline.to(device)
    return pipeline


def load_image(image_path, transform):
    """加载并预处理图像"""
    image = Image.open(image_path).convert("RGB")
    return transform(image)


def cosine_similarity(embeddings1, embeddings2):
    """计算两组嵌入的余弦相似度"""
    embeddings1 = embeddings1 / embeddings1.norm(dim=-1, keepdim=True)
    embeddings2 = embeddings2 / embeddings2.norm(dim=-1, keepdim=True)
    similarity = (embeddings1 @ embeddings2.T).squeeze()
    return similarity.item()


def compute_dino_score(
    real_image_paths, generated_image_paths, model_name="vit_small_patch16_224.dino"
):
    """
    计算 DINO 分数：生成图像与真实图像的 DINO 嵌入的平均余弦相似度
    Args:
        real_image_paths: 真实参考图像路径列表
        generated_image_paths: 生成图像路径列表
        model_name: DINO 模型名称（默认 ViT-S/16）
    Returns:
        dino_score: 平均 DINO 分数
    """
    # 加载 DINO 模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(model_name, pretrained=True).to(device)
    model.eval()

    # 图像预处理
    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    similarities = []
    with torch.no_grad():
        for real_path, gen_path in tqdm(zip(real_image_paths, generated_image_paths), total=len(real_image_paths), desc="Computing DINO Score"):
            # 加载并预处理图像
            real_image = load_image(real_path, transform).unsqueeze(0).to(device)
            gen_image = load_image(gen_path, transform).unsqueeze(0).to(device)

            # 提取 DINO 嵌入
            real_embedding = model(real_image)
            gen_embedding = model(gen_image)

            # 计算余弦相似度
            similarity = cosine_similarity(real_embedding, gen_embedding)
            similarities.append(similarity)

    # 计算平均 DINO 分数
    dino_score = np.mean(similarities)
    return dino_score


# 4. 计算 CLIP 分数（CLIP-I 和 CLIP-T）
def compute_clip_scores(
    real_image_paths,
    generated_image_paths,
    prompts,
    model_name="openai/clip-vit-base-patch32",
):
    """
    计算 CLIP 分数：
    - CLIP-I：生成图像与真实图像的 CLIP 图像嵌入的平均余弦相似度
    - CLIP-T：生成图像与文本提示的 CLIP 嵌入的平均余弦相似度
    Args:
        real_image_paths: 真实参考图像路径列表
        generated_image_paths: 生成图像路径列表
        prompts: 文本提示列表
        model_name: CLIP 模型名称
    Returns:
        clip_i_score: 平均 CLIP-I 分数
        clip_t_score: 平均 CLIP-T 分数
    """
    # 加载 CLIP 模型和处理器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()

    clip_i_similarities = []
    clip_t_similarities = []

    with torch.no_grad():
        for real_path, gen_path, prompt in tqdm(
            zip(real_image_paths, generated_image_paths, prompts),
            total=len(real_image_paths),
            desc="Computing CLIP Scores"
        ):
            # 加载并预处理图像
            real_image = Image.open(real_path).convert("RGB")
            gen_image = Image.open(gen_path).convert("RGB")

            # 处理图像和文本
            inputs = processor(
                text=[prompt],
                images=[real_image, gen_image],
                return_tensors="pt",
                padding=True,
            ).to(device)

            # 提取嵌入
            outputs = model(**inputs)
            real_image_embedding = outputs.image_embeds[0]  # 真实图像嵌入
            gen_image_embedding = outputs.image_embeds[1]  # 生成图像嵌入
            text_embedding = outputs.text_embeds[0]  # 文本嵌入

            # 计算 CLIP-I 相似度
            clip_i_similarity = cosine_similarity(
                gen_image_embedding, real_image_embedding
            )
            clip_i_similarities.append(clip_i_similarity)

            # 计算 CLIP-T 相似度
            clip_t_similarity = cosine_similarity(gen_image_embedding, text_embedding)
            clip_t_similarities.append(clip_t_similarity)

    # 计算平均 CLIP 分数
    clip_i_score = np.mean(clip_i_similarities)
    clip_t_score = np.mean(clip_t_similarities)
    return clip_i_score, clip_t_score


def evaluate_and_save_scores(real_dir, gen_dir, captions_file, output_txt):
    """
    评估生成图像的DINO分数和CLIP分数，并将结果保存到txt文件
    Args:
        real_dir: 真实图像文件夹路径
        gen_dir: 生成图像文件夹路径
        captions_file: 文本提示文件路径
        output_txt: 结果保存的txt文件路径
    """
    real_image_paths = [
        os.path.join(real_dir, fname)
        for fname in sorted(os.listdir(real_dir))
        if fname.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    generated_image_paths = [
        os.path.join(gen_dir, fname)
        for fname in sorted(os.listdir(gen_dir))
        if fname.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    with open(captions_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f.readlines()]
    # 保证三者长度一致
    min_len = min(len(real_image_paths), len(generated_image_paths), len(prompts))
    print(f"Number of real images: {len(real_image_paths)}")
    print(f"Number of generated images: {len(generated_image_paths)}")
    print(f"Number of prompts: {len(prompts)}")
    real_image_paths = real_image_paths[:min_len]
    generated_image_paths = generated_image_paths[:min_len]
    prompts = prompts[:min_len]

    dino_score = compute_dino_score(real_image_paths, generated_image_paths)
    clip_i_score, clip_t_score = compute_clip_scores(
        real_image_paths, generated_image_paths, prompts
    )

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(f"DINO Score: {dino_score:.3f}\n")
        f.write(f"CLIP-I Score: {clip_i_score:.3f}\n")
        f.write(f"CLIP-T Score: {clip_t_score:.3f}\n")
    print("Scores have been successfully evaluated and saved.")