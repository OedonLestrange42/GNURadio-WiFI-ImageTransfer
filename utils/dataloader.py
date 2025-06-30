import os
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image


class ImagenetMini(Dataset):
    def __init__(self, image_dir, transform=None):
        """
        Args:
            image_dir (string): 图片文件夹路径。
            transform (callable, optional): 一个可选的变换函数，用于处理样本。
        """
        self.image_dir = image_dir
        self.transform = transform
        self.images = []

        for subdir, dirs, files in os.walk(image_dir):
            for filename in files:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(subdir, filename)
                    category = subdir.split('\\')[-1]  # 文件名就是类别名
                    self.images.append((image_path, category))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path, category = self.images[idx]
        image = Image.open(image_path).convert('RGB')  # 确保图片为RGB格式

        if self.transform:
            image = self.transform(image)

        return image, category


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # 调整图片大小以匹配AlexNet结构
        # transforms.CenterCrop(224),  # 中心裁剪以得到 224x224 图片
        transforms.ToTensor(),  # 将图片转换为Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
    ])

    dataset = ImagenetMini(image_dir='../input_images', transform=transform)
    print(len(dataset))
