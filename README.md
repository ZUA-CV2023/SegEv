# SegEv: semantic segmentation performance verification and evaluation software



With the widespread application of semantic segmentation  technology in fields such as remote sensing and industrial inspection, the evaluation of model performance and visualization of training processes have become key issues. This paper develops an integrated evaluation software based on PyQt5 and TensorBoard, which supports the calculation of eight metrics including Precision, Recall, and mIoU, and provides functions such as multi-algorithm comparison and batch processing.  
Through TensorBoard, the software realizes the visualization of model architectures, feature maps, heatmaps, and loss maps, intuitively displaying the differences between segmentation results and ground truth labels to assist in parameter optimization. With its modular design, the software combines both evaluation and visualization capabilities, providing efficient tool support for the development and deployment of segmentation models.

## Installation

Install the requirements specified.

| No.  | Library Name       | Version      |
| ---- | ------------------ | ------------ |
| 1    | torch              | 1.5.1+cu101  |
| 2    | torchvision        | 0.6.1+cu101  |
| 3    | python             | 3.7.16       |
| 4    | Gitpython          | 3.1.32       |
| 5    | ipython            | 7.34.0       |
| 6    | Labelme            | 5.3.1        |
| 7    | Matplotlib         | 3.5.3        |
| 8    | Numpy              | 1.21.6       |
| 9    | opencv-python      | 4.8.0.74     |
| 10   | pandas             | 1.3.5        |
| 11   | pillow             | 9.5.0        |
| 12   | pip                | 22.3.1       |
| 13   | vc                 | 14.2         |
| 14   | vs2015_runtime     | 14.27.29016  |
| 15   | scipy              | 1.7.3        |
| 16   | scikit-image       | 0.19.3       |
| 17   | pycocotools        | 2.0.7        |
| 18   | tensorboard        | 2.11.2       |
| 19   | tensorflow         | 2.11.0       |
| 20   | torchsummary       | 1.5.1        |
| 21   | seaborn            | 0.12.2       |
| 22   | psutil             | 5.9.5        |
| 23   | pyQt5              | 5.15.9       |
| 24   | pyqt5-tools        | 5.15.9.3.3   |
| 25   | PySide2            | 5.15.2.1     |
| 26   | pytz               | 2023.3.post1 |
| 27   | PyYAML             | 6.0.1        |
| 28   | qt5-applications   | 5.15.2.2.3   |
| 29   | qt5-tools          | 5.15.2.1.3   |
| 30   | requests           | 2.31.0       |
| 31   | Gitdb              | 4.0.10       |
| 32   | tqdm               | 4.66.1       |
| 33   | ipdb               | 0.13.13      |
| 34   | absl-py            | 2.0.0        |
| 35   | cachetools         | 5.3.2        |
| 36   | click              | 8.1.7        |
| 37   | cycler             | 0.11.0       |
| 38   | cython             | 3.0.5        |
| 39   | imgviz             | 1.2.6        |
| 40   | importlib-metadata | 6.7.0        |
| 41   | joblib             | 1.3.2        |
| 42   | kiwisolver         | 1.4.5        |
| 43   | lightgbm           | 4.2.0        |
| 44   | mahotas            | 1.4.13       |
| 45   | Markdown           | 3.4.4        |
| 46   | packaging          | 23.1         |
| 47   | pycocotools        | 2.0.7        |
| 48   | scikit-learn       | 1.0.2        |
| 49   | scipy              | 1.7.3        |
| 50   | seaborn            | 0.12.2       |
| 51   | setuptools         | 65.6.3       |
| 52   | six                | 1.16.0       |
| 53   | timm               | 0.9.7        |
| 54   | typing-extensions  | 4.7.1        |
| 55   | wheel              | 0.38.4       |
| 56   | zipp               | 3.15.0       |
| 57   | torch              | 1.8.1+cu101  |
| 58   | torchvision        | 0.9.1+cu101  |

## Usage

Click on **window.py** in the **SegEv** folder to run the software.

1、Quantitative Metric Evaluation for Semantic Segmentation

（1）**Operation Process for Semantic Segmentation Performance Metrics Comparison**

Click the **"Compare weight A"** button, and the user can select the image folder to be evaluated from local files. The system will then pop up a window to select the weight file ("Weight A"). After all selections are completed, the "Algorithm ①" window will display a comparison between the "segmentation result" (generated using Weight A) and the "ground truth label," including metrics such as class names, class confidence scores, and mIoU.

Click the **"Compare  weight B"** button, and the user can similarly select the image folder and "Weight B." Upon completion, the "Algorithm ②" window will show the comparison for Weight B in the same format.

Note: During this process, users can press the **"A"** key to view the previous image or the **"D"** key to view the next image.

Then, click the **"Accuracy"** button on the right side of the main interface. The lower window will display the segmentation accuracy of both "Weight A" and "Weight B," allowing users to compare them visually.

（2）**Quantitative Metrics Comparison Workflow**

Click the **"Select Mask File A"** button to choose the target mask folder for evaluation from local files. Upon successful file addition, the system will display a notification: **"Folder added successfully!"**

Once the file path is confirmed, click the metric buttons on the right panel to visualize the evaluation results.

Click the **"Select Mask File B"** button to add another mask folder for comparison. The system will again confirm with **"Folder added successfully!"** Then, click the **"mPA"** button to display the mPA visualization in the **"B"** window, enabling side-by-side comparison with the mPA results from Mask File A.

（3）**Real-time Evaluation Workflow**

Click the **"Select File"** button to choose the infrared image folder for evaluation from local files. Upon successful selection, the system will display a notification: **"Folder added successfully!"**

After selecting the folder, click the **"Start"** button to initiate the analysis and computation. The system will simultaneously display real-time status notifications to keep users informed about the current computation progress.

Upon completion of processing, the system will show a **"Processing complete"** notification. Users can then click the metric buttons on the right panel to visualize the evaluation results.

2、**Semantic Segmentation Model Visualization Operation**

  (1) **Visualization**
Click the **"Visualize"** button to select the infrared image for processing from local files. When an image is selected, the system will display a notification: **"Image selected successfully!"**

A **"Processing started!"** notification will then appear. Click the **"OK"** button in this prompt to initiate the image processing. The backend algorithm will generate the corresponding **feature maps** and **heatmaps** for the selected image.

Upon completion of the processing, the system will display a **"Processing complete!"** notification. The generated visualization results (feature maps and heatmaps) will be automatically displayed in the designated viewing window.

（2）**Heatmap**
Click the **"Heatmap"** button on the left panel to display the heatmap of the selected image in the **"A"** window.

（3）**Feature Map**
Click the **"Feature Map"** button on the left panel to display the feature map of the selected image in the **"A"** window.

（4）**Loss map **
Click the **"Loss map"** button on the left panel to visualize the variation of the loss function data. By observing the loss graph, you can see that the training loss gradually decreases and stabilizes over time, maintaining relatively consistent changes within a certain period. A stable loss function indicates that the model is gradually converging, demonstrating effective learning and smooth training progress.

（5）**Structure map**
Click the **"Structure map"** button on the left panel to view the model's structural diagram.



## Authors

Jingjing Yan

Xiaoyan Shao

Lingling Li

Xuezhuan Zhao

Xiaoyu Hao
