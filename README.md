# Ass3 Work - Shelf Stock-Out Detection Project

这份文件是给 CNN / YOLO / ResNet 初学者看的项目说明。目标是让组员不用先理解所有深度学习细节，也能知道每个文件在做什么、数据怎么流动、应该怎么运行、结果应该怎么看。

## 1. 项目要解决什么问题

我们的 Assignment 3 项目是一个货架缺货检测系统。

输入是一张超市货架图片，系统需要完成两件事：

1. 找出货架上可能缺货的位置。
2. 根据缺货位置附近的商品，推断这个空位可能属于哪一类商品。

我们不做精确 SKU 识别。也就是说，系统不会判断：

```text
Coca-Cola 375ml can is out of stock
```

而是判断：

```text
Potential stock-out: beverage
Potential stock-out: snack
Potential stock-out: packaged_food
Potential stock-out: personal_care
```

这样做的原因是：精准 SKU 识别需要商品数据库、货架陈列图和 SKU 级标注，当前公开数据集不支持。

## 2. 项目有两个版本

### 2.1 YOLO-only 系统

这个版本只训练一个 YOLO11 模型。

作用：

```text
货架图片 -> YOLO11 -> 缺货空位框
```

输出是 bounding box，例如：

```text
void box: x1, y1, x2, y2, confidence
```

这个版本回答的问题是：

```text
哪里可能缺货？
```

### 2.2 双系统 YOLO + ResNet

这个版本使用两个模型。

第一个模型：

```text
YOLO11
```

负责检测缺货空位。

第二个模型：

```text
ResNet18
```

负责判断缺货空位附近商品的大致类别。

整体流程：

```text
货架图片
  |
  v
YOLO11 检测 void 空位
  |
  v
裁剪空位左边和右边的邻近区域
  |
  v
ResNet18 判断邻近区域属于什么商品类别
  |
  v
输出可能缺货类别
```

这个版本回答的问题是：

```text
哪里可能缺货？
这个空位可能是哪一类商品缺货？
```

## 3. CNN / YOLO / ResNet 的简单解释

### 3.1 CNN 是什么

CNN 是 Convolutional Neural Network，中文是卷积神经网络。它适合处理图片。

CNN 会从图片中学习视觉特征，例如：

```text
边缘
颜色
纹理
形状
包装图案
物体轮廓
```

在我们的项目里，CNN 用来学习：

```text
空货架区域长什么样
饮料、零食、包装食品、个人护理商品大概长什么样
```

### 3.2 YOLO 是什么

YOLO 是一种 object detection 模型。

Object detection 的任务是：

```text
找出图片中物体的位置 + 判断物体类别
```

YOLO 的输出不是一个类别，而是一组框：

```text
box 1: void, confidence 0.91
box 2: void, confidence 0.76
```

所以 YOLO 适合做：

```text
缺货空位检测
```

### 3.3 ResNet 是什么

ResNet 是一种 CNN 分类模型。

Image classification 的任务是：

```text
输入一张图片 -> 输出一个类别
```

例如：

```text
输入一张商品图片 -> 输出 snack
```

ResNet 不会同时识别一张图里的多个物体。它会输出这张图最像哪一个类别。

在我们的项目里，ResNet 用来判断：

```text
缺货空位旁边的商品 crop 最像哪一类商品
```

## 4. 文件结构

当前简化项目文件夹是：

```text
Ass3/Ass3_work/
├── merge_datasets.py
├── yolo_system.py
├── resnet_system.py
├── gui_app.py
├── requirements.txt
└── README.md
```

每个文件作用如下。

### 4.1 merge_datasets.py

作用：准备数据集。

它会做两件事：

1. 合并 YOLO 缺货空位数据集。
2. 把 MIMEX 的 28 个细分类合并成 4 个粗类别。

输入：

```text
Ass3/dataset/YOLO/
Ass3/dataset/MIMEX/images/
```

输出：

```text
Ass3/dataset/Datasset/
```

如果 `Ass3/dataset/Datasset/` 已经存在，通常不需要重新运行。

### 4.2 yolo_system.py

作用：训练、测试、预测 YOLO11 缺货检测模型。

它读取：

```text
Ass3/dataset/Datasset/void_detection/
```

训练后输出：

```text
runs/yolo11_void/best.pt
runs/yolo11_void/test_metrics.json
```

`best.pt` 是训练好的 YOLO 模型权重。

### 4.3 resnet_system.py

作用：训练、测试、预测 ResNet18 商品粗分类模型。

它读取：

```text
Ass3/dataset/Datasset/category_classification/
```

训练后输出：

```text
runs/resnet18_mimex_coarse/best_resnet18.pth
runs/resnet18_mimex_coarse/classification_report.txt
runs/resnet18_mimex_coarse/confusion_matrix_normalized.png
```

`best_resnet18.pth` 是训练好的 ResNet 模型权重。

### 4.4 gui_app.py

作用：本地 GUI 演示。

它会连接两个训练好的模型：

```text
YOLO11 -> 检测缺货空位
ResNet18 -> 判断附近商品类别
```

GUI 可以用于：

```text
上传货架图片
显示缺货检测框
显示推断类别
截图放进报告
```

## 5. 数据集说明

### 5.1 void_detection 数据集

路径：

```text
Ass3/dataset/Datasset/void_detection/
```

结构：

```text
void_detection/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── void_dataset.yaml
```

`images` 文件夹放图片。

`labels` 文件夹放 YOLO 标注文件。

每张图片对应一个同名 txt 文件，例如：

```text
images/train/example.jpg
labels/train/example.txt
```

txt 每一行是一个缺货框：

```text
class_id x_center y_center width height
```

例如：

```text
0 0.219531 0.567188 0.070312 0.107813
```

含义：

```text
0          -> 类别编号，0 表示 void
0.219531   -> 框中心点 x 坐标，已经归一化到 0-1
0.567188   -> 框中心点 y 坐标，已经归一化到 0-1
0.070312   -> 框宽度，已经归一化到 0-1
0.107813   -> 框高度，已经归一化到 0-1
```

YOLO 会根据这些标注学习：

```text
图片中的哪些区域是空货架 / 缺货空位
```

### 5.2 category_classification 数据集

路径：

```text
Ass3/dataset/Datasset/category_classification/
```

结构：

```text
category_classification/
├── train/
│   ├── beverage/
│   ├── packaged_food/
│   ├── personal_care/
│   └── snack/
├── val/
│   ├── beverage/
│   ├── packaged_food/
│   ├── personal_care/
│   └── snack/
└── test/
    ├── beverage/
    ├── packaged_food/
    ├── personal_care/
    └── snack/
```

这里的文件夹名就是分类 label。

PyTorch 的 `ImageFolder` 会自动把文件夹名转换成数字：

```text
beverage      -> 0
packaged_food -> 1
personal_care -> 2
snack         -> 3
```

也就是说，ResNet 不需要手动读取 CSV。它直接根据文件夹名学习类别。

## 6. MIMEX 28 类到 4 类的映射

我们把 MIMEX 原始 28 类合并成 4 类：

```text
snack
beverage
packaged_food
personal_care
```

对应关系：

```text
snack:
  rocher_chocolate
  milka_chocolate
  kinder_chocolate
  toblerone_white
  toblerone_black
  lays_classic
  lays_chill
  pringles_original
  pringles_paprika

beverage:
  nestle_water
  sanpellegrino_water
  redbull_energydrink
  monster_energydrink

packaged_food:
  heinz_ketchup
  heinz_mayo
  barilla_pesto
  barilla_pomodoro
  barilla_lasagne

personal_care:
  loreal_shampoo
  dove_soap
  sensodyne_toothpaste
  colgate_toothpaste
  sensodyne_mouthwash
  nivea_rollon
  rexona_spray
  dove_spray
  nivea_baby
  johnson_baby
```

## 7. 安装依赖

本地 Mac 运行：

```bash
python3 -m pip install -r Ass3/Ass3_work/requirements.txt
```

Colab 运行：

```python
!pip install ultralytics scikit-learn pyyaml
```

如果本地没有 GPU，训练会很慢。当前建议：

```text
本地：用于 smoke test 和 GUI demo
Colab：用于正式训练
```

## 8. 本地快速跑通测试

这个测试不是为了得到好模型，只是为了确认代码能跑通。

### 8.1 测试 YOLO

```bash
python3 Ass3/Ass3_work/yolo_system.py train \
  --epochs 1 \
  --imgsz 320 \
  --batch 2 \
  --device cpu \
  --smoke-test \
  --output-dir runs/smoke_yolo
```

如果成功，会生成：

```text
runs/smoke_yolo/best.pt
runs/smoke_yolo/test_metrics.json
```

### 8.2 测试 ResNet

```bash
python3 Ass3/Ass3_work/resnet_system.py train \
  --epochs 1 \
  --batch 8 \
  --device cpu \
  --smoke-test \
  --output-dir runs/smoke_resnet
```

如果成功，会生成：

```text
runs/smoke_resnet/best_resnet18.pth
runs/smoke_resnet/classification_report.txt
runs/smoke_resnet/confusion_matrix_normalized.png
```

## 9. 正式训练 YOLO11

本地训练：

```bash
python3 Ass3/Ass3_work/yolo_system.py train \
  --epochs 100 \
  --imgsz 640 \
  --batch 8 \
  --output-dir runs/yolo11_void
```

Colab 训练：

```bash
python3 Ass3/Ass3_work/yolo_system.py train \
  --dataset-root /content/drive/MyDrive/25694148/Datasset/void_detection \
  --output-dir /content/drive/MyDrive/25694148/training_outputs/yolo11_void \
  --device 0 \
  --epochs 100 \
  --imgsz 640 \
  --batch 16
```

训练结果里重点看：

```text
best.pt
test_metrics.json
```

YOLO 常见指标：

```text
precision: 预测为空位的框里有多少是真的
recall: 实际空位里有多少被模型找到了
mAP50: IoU=0.5 时的平均检测性能
mAP50-95: 更严格的平均检测性能
```

## 10. 正式训练 ResNet18

本地训练：

```bash
python3 Ass3/Ass3_work/resnet_system.py train \
  --epochs 25 \
  --batch 64 \
  --output-dir runs/resnet18_mimex_coarse
```

Colab 训练：

```bash
python3 Ass3/Ass3_work/resnet_system.py train \
  --dataset-root /content/drive/MyDrive/25694148/Datasset/category_classification \
  --output-dir /content/drive/MyDrive/25694148/training_outputs/resnet18_mimex_coarse \
  --device cuda \
  --epochs 25 \
  --batch 64
```

训练结果里重点看：

```text
best_resnet18.pth
classification_report.txt
confusion_matrix_normalized.png
```

ResNet 常见指标：

```text
accuracy: 总体预测正确率
precision: 某类预测为该类时，有多少是真的
recall: 某类真实样本里，有多少被找出来了
F1-score: precision 和 recall 的综合指标
```

标准化混淆矩阵的含义：

```text
每一行代表真实类别
每一列代表预测类别
每一行加起来约等于 1
```

图中对角线越深，说明分类越准确。

## 11. 单张图片预测

### 11.1 YOLO 检测图片

```bash
python3 Ass3/Ass3_work/yolo_system.py predict \
  --weights runs/yolo11_void/best.pt \
  --source path/to/shelf_image.jpg \
  --output-dir runs/yolo_predictions
```

输出图片会保存在：

```text
runs/yolo_predictions/images/
```

### 11.2 ResNet 输出类别概率

```bash
python3 Ass3/Ass3_work/resnet_system.py predict \
  --weights runs/resnet18_mimex_coarse/best_resnet18.pth \
  --image path/to/product_crop.jpg
```

输出类似：

```text
{
  "personal_care": 0.73,
  "packaged_food": 0.12,
  "beverage": 0.08,
  "snack": 0.07
}
```

## 12. 运行 GUI Demo

训练完两个模型后运行：

```bash
python3 Ass3/Ass3_work/gui_app.py \
  --yolo-weights runs/yolo11_void/best.pt \
  --resnet-weights runs/resnet18_mimex_coarse/best_resnet18.pth
```

GUI 操作：

1. 点击 `Open Image` 选择货架图片。
2. 点击 `Run Detection`。
3. 左侧显示检测框。
4. 右侧显示每个缺货区域的类别推断概率。
5. 点击 `Save Result` 保存截图。
