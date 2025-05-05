from data import load_captions_dataset
from utils import create_pipeline, generate_all_images

dataset_dict = load_captions_dataset()
val_dataset = dataset_dict["validation"]
print("val_dataset", val_dataset)
val_dataset = val_dataset[:500]
with open("val_captions.txt", "w", encoding="utf-8") as f:
    for item in val_dataset["caption"]:
        f.write(item + "\n")
pipeline = create_pipeline("stu_model", device="cuda:6")
stu_model_inference_time = generate_all_images(pipeline, val_dataset, "stu_model")
pipeline = create_pipeline("tea_model", device="cuda:4")
tea_model_inference_time = generate_all_images(pipeline, val_dataset, "tea_model")
with open("inference_time.txt", "w") as f:
    f.write(f"stu_model_inference_time: {stu_model_inference_time:.2f} seconds\n")
    f.write(f"tea_model_inference_time: {tea_model_inference_time:.2f} seconds\n")
    f.write(
        f"speedup_ratio: {tea_model_inference_time / stu_model_inference_time:.2f}\n"
    )
