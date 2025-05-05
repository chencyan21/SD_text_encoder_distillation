from utils import evaluate_and_save_scores

# 真实图像文件夹
real_dir = 'real_images'
# 生成A组和B组图像文件夹
student_model_images = 'stu_model_images'
teacher_model_images = 'tea_model_images'
# 文本提示文件
captions_file = 'val_captions.txt'
# 结果保存
output_txt_student_model = 'result_stu.txt'
output_txt_teacher_model = 'result_tea.txt'

evaluate_and_save_scores(real_dir, student_model_images, captions_file, output_txt_student_model)
evaluate_and_save_scores(real_dir, teacher_model_images, captions_file, output_txt_teacher_model)