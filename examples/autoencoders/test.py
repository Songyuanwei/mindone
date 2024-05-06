import os

from PIL import Image
from sklearn.model_selection import train_test_split


# 定义图片路径和分类文件夹
def make_dataset(dir):
    images = []
    for root, subdirs, files in os.walk(dir):
        for file in files:
            if file.endswith(".jpg"):
                images.append(os.path.join(root, file))
    return images


# 将图片分为训练集和测试集
def split_train_test(input_images, test_size=0.2):
    train_images, test_images = train_test_split(input_images, test_size=test_size)
    return train_images, test_images


# 保存图片到新的文件夹
def save_images(images, save_dir, prefix):
    for img_path in images:
        img = Image.open(img_path)
        basename = os.path.basename(img_path)
        save_path = os.path.join(save_dir, basename)
        img.save(save_path)


# 主函数
def main(input_dir, output_train_dir, output_test_dir, test_size=0.2):
    images = make_dataset(input_dir)
    train_images, test_images = split_train_test(images, test_size)
    save_images(train_images, output_train_dir, "train")
    save_images(test_images, output_test_dir, "test")


# 示例使用
if __name__ == "__main__":
    # input_dir = "/disk1/mindone/songyuanwei/datasets/celeba_hq_256xx/"  # 输入图片文件夹路径
    # output_train_dir = "/disk1/mindone/songyuanwei/datasets/celeba_hq_256/train"  # 输出训练集文件夹路径
    # output_test_dir = "/disk1/mindone/songyuanwei/datasets/celeba_hq_256/test"  # 输出测试集文件夹路径
    # main(input_dir, output_train_dir, output_test_dir)
    import numpy as np

    import mindspore
    from mindspore import Tensor, ops

    x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), mindspore.float32)
    y = Tensor(np.array([[2.0, 3.0], [1.0, 2.0], [4.0, 5.0]]), mindspore.float32)
    equation = "ij,jk->ik"
    # output = ops.einsum(equation, x, y)
    # print(output)
    out = ops.matmul(x, y)
    print(out)
