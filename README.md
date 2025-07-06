# Wall-Ceiling-segmentation-Yolo-Model
Wall and Ceiling Segmentation Developed a custom YOLOv11-inspired semantic segmentation model using TensorFlow/Keras for real-time wall and ceiling detection in augmented reality applications. 



Wall and Ceiling Segmentation AR Project
=======================================

Overview
--------
This project implements a YOLOv11-inspired semantic segmentation model using TensorFlow/Keras for real-time wall and ceiling detection in augmented reality (AR) applications. The model processes 256x256 RGB images, segmenting three classes (background, wall, ceiling) with a target accuracy of >96% and IoU >0.9. It features custom layers (`SpatialDropout2D`, `ChannelAttention`) and a combined loss function (IoU, Dice, Focal) to optimize segmentation performance. The model is trained on a custom dataset, converted to TFLite with int8 quantization for deployment in a React Native AR app, and tested for <0.7s inference on mid-range devices.

Key Features
------------
- Custom YOLOv11-inspired architecture with `SpatialDropout2D` and `ChannelAttention` layers.
- Combined loss function: 0.4 IoU + 0.4 Dice + 0.2 Focal to handle class imbalance.
- Data augmentation using Albumentations for robust training.
- Training with cosine decay learning rate and callbacks (`ModelCheckpoint`, `EarlyStopping`, `TensorBoard`).
- TFLite conversion with int8 quantization (~1-2MB model size).
- Testing pipeline for visualizing segmentation masks on test images.
- Deployment-ready for React Native AR applications.

Dataset
-------
- **Location**: `E:\YOLO Model\Annotated_Dataset - Copy`
- **Structure**:
  - Training: `train/images`, `train/masks`
  - Validation: `validation/images`, `validation/masks`
  - Test: `For testing the model dataset`
- **Format**: Images (JPG/PNG), masks (grayscale PNG with classes 0=background, 1=wall, 2=ceiling).
- **Preprocessing**: Images resized to 256x256, normalized; masks one-hot encoded for 3 classes.

Requirements
------------
- Python 3.11.6
- TensorFlow 2.17.0
- OpenCV 4.10.0.84
- NumPy 1.26.4
- Albumentations 1.4.18
- Matplotlib 3.9.2

Install dependencies:
```
pip install tensorflow==2.17.0 opencv-python==4.10.0.84 numpy==1.26.4 albumentations==1.4.18 matplotlib==3.9.2
```

Project Structure
-----------------
- `wall_ceiling_segmentation.ipynb`: Jupyter Notebook containing the complete pipeline (data loading, model definition, training, TFLite conversion, testing).
- `YOLO_Inspired_Model.keras`: Saved Keras model after training.
- `YOLO_Inspired_Model.tflite`: Converted TFLite model for deployment.
- `logs/`: TensorBoard logs for training visualization.
- `README.txt`: This file.

Usage
-----
1. **Setup Dataset**:
   - Place your dataset in `E:\YOLO Model\Annotated_Dataset - Copy` with the structure above.
   - Ensure test images are in `E:\YOLO Model\For testing the model dataset`.

2. **Run the Notebook**:
   - Open `wall_ceiling_segmentation.ipynb` in Jupyter Notebook.
   - Update paths in Cells 2 and 9 if your dataset is in a different location.
   - Run cells sequentially:
     - Cells 1-6: Data loading, augmentation, and model definition.
     - Cell 7: Custom loss functions (IoU, Dice, Focal, Combined).
     - Cell 8: Model training (40 epochs, ~48 hours on NVIDIA RTX 3080).
     - Cell 9: TFLite conversion with int8 quantization.
     - Cell 10: Test TFLite model on test images with visualization.

3. **Training**:
   - Uses cosine decay learning rate and callbacks for efficient training.
   - Saves best model as `YOLO_Inspired_Model.keras` based on validation IoU.

4. **TFLite Conversion**:
   - Converts `YOLO_Inspired_Model.keras` to `YOLO_Inspired_Model.tflite` with int8 quantization.
   - Uses a representative dataset from validation images for calibration.

5. **Testing**:
   - Tests the TFLite model on images in `For testing the model dataset`.
   - Outputs visualizations of input images and predicted segmentation masks (0=background, 1=wall, 2=ceiling).

6. **React Native Integration**:
   ```javascript
   import TFLite from 'react-native-tflite';
   const model = new TFLite();
   model.loadModel({ model: 'YOLO_Inspired_Model.tflite' });
   model.runModelOnImage({
     path: imagePath,
     imageMean: -128,
     imageStd: 255,
     numResults: 3,
     threshold: 0.5
   }).then(result => {
     console.log('Segmentation mask:', result);
   });
   ```

Troubleshooting
---------------
- **Low IoU (~0.33)**: Training log showed low IoU. Adjust loss weights in Cell 7 (e.g., 0.5 IoU, 0.3 Dice, 0.2 Focal) or verify mask labels:
  ```python
  mask = cv2.imread('E:\\YOLO Model\\Annotated_Dataset - Copy\\validation\\masks\\sample.png', cv2.IMREAD_GRAYSCALE)
  plt.imshow(mask, cmap='jet')
  plt.show()
  ```
- **Quantization Issues**: If predictions are inaccurate, try float32 quantization in Cell 9:
  ```python
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.target_spec.supported_types = [tf.float32]
  converter.inference_input_type = tf.float32
  converter.inference_output_type = tf.float32
  converter.representative_dataset = representative_dataset
  ```
- **Errors**: Ensure `YOLO_Inspired_Model.keras` and dataset paths exist. Check console for specific error messages and share for debugging.

Skills Demonstrated
-------------------
- Deep Learning (TensorFlow/Keras)
- Model Architecture Design (YOLOv11-inspired)
- Loss Function Engineering
- Data Preprocessing and Augmentation
- Model Training and Optimization
- TFLite Conversion and Quantization
- Error Debugging
- Mobile Deployment (React Native)
- Visualization (Matplotlib, TensorBoard)

License
-------
MIT License

Author
------
Bilal Saif

Date
----
July 6, 2025
