import matplotlib.pyplot as plt
import matplotlib.image as mpimg

stu_images = [
    'stu_model_images/image_1.png',
    'stu_model_images/image_2.png'
]
tea_images = [
    'tea_model_images/image_1.png',
    'tea_model_images/image_2.png'
]
stu_texts = ['Student Model\nMan in apron standing on front of oven with pans and bakeware', 'Student Model\nA baker is working in the kitchen rolling dough.']
tea_texts = ['Teacher Model\nMan in apron standing on front of oven with pans and bakeware', 'Teacher Model\nA baker is working in the kitchen rolling dough.']

# 放大画布
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for i, (img_path, text) in enumerate(zip(stu_images, stu_texts)):
    img = mpimg.imread(img_path)
    axes[0, i].imshow(img)
    axes[0, i].axis('off')
    axes[0, i].set_title('')
    axes[0, i].text(0.5, -0.05, text, size=14, ha='center', va='top', transform=axes[0, i].transAxes)

for i, (img_path, text) in enumerate(zip(tea_images, tea_texts)):
    img = mpimg.imread(img_path)
    axes[1, i].imshow(img)
    axes[1, i].axis('off')
    axes[1, i].set_title('')
    axes[1, i].text(0.5, -0.05, text, size=14, ha='center', va='top', transform=axes[1, i].transAxes)

# 减少图片间的空隙
# plt.subplots_adjust(wspace=0.001, hspace=0.15)

plt.savefig('model_images.png', dpi=300, bbox_inches='tight')
plt.show()
print("ok")
