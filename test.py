import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # 可根据类别不平衡程度调整
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [batch_size, num_classes]
        # targets: [batch_size]
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # 预测概率
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
loss_fn = FocalLoss(alpha=0.25, gamma=2)





# 定义类别权重，例如将第一类缺陷的权重设为2，第二类缺陷设为1
class_weights = torch.tensor([2.0, 1.0]).to(device)

loss_fn = nn.CrossEntropyLoss(weight=class_weights)



class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # inputs: [batch_size, num_classes]
        # targets: [batch_size]
        inputs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()

        intersection = (inputs * targets_one_hot).sum(dim=0)
        union = inputs.sum(dim=0) + targets_one_hot.sum(dim=0)

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        loss = 1 - dice.mean()

        return loss



class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()

        TP = (inputs * targets_one_hot).sum(dim=0)
        FP = ((1 - targets_one_hot) * inputs).sum(dim=0)
        FN = (targets_one_hot * (1 - inputs)).sum(dim=0)

        Tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        loss = 1 - Tversky.mean()

        return loss
    
model.head = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(model.head.in_features, 2)
)




    
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=[0, 90, 180, 270]),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.3, 0.8))], p=0.3),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    #随机调整图像的色彩属性，包括以下方面：
    #亮度（brightness）：在原始亮度的基础上随机调整，范围为 ±20%。
    #对比度（contrast）：在原始对比度的基础上随机调整，范围为 ±20%。
    #饱和度（saturation）：在原始饱和度的基础上随机调整，范围为 ±20%。
    #色调（hue）：在原始色调的基础上随机调整，范围为 ±0.1。
    #kernel_size 决定了高斯模糊的影响范围：越大，模糊程度越强。
    #sigma 决定了模糊的强度：越大，模糊程度越强。
    #在小尺寸图像和细节敏感的任务中，应选择较小的 kernel_size 和 sigma，以防止过度模糊导致细节丢失。
    #建议您在实际训练中，通过实验调整 kernel_size 和 sigma 的值，找到最适合您任务的参数组合。  
])