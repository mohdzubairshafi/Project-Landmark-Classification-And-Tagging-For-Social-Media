# Landmark Classification & Tagging for Social Media

## Project Overview

- In the world of photo sharing and storage services, having location data for each photo is essential. This information enables advanced features like automatic tag suggestions and seamless photo organization, enhancing the user experience. However, many photos lack location metadata, whether due to the absence of GPS or privacy concerns.

- To address this, the project aims to automatically classify and tag photos by identifying discernible landmarks in the images. Given the vast number of landmarks globally and the immense volume of uploaded photos, manual classification is impractical.

## Project Steps

- Create a CNN to Classify Landmarks (from Scratch): This phase involves visualizing the dataset, data preprocessing, and building a custom convolutional neural network (CNN) for landmark classification. It includes key decisions about data processing and network architecture, followed by exporting the best network using Torch Script.

- Create a CNN to Classify Landmarks (using Transfer Learning): Here, you explore various pre-trained models and select one for the classification task. You'll explain the choice of the pre-trained model, train and test it, and export the best transfer learning solution using Torch Script.

- Deploy your algorithm in an app: In the final step, you use the best model to create a user-friendly app for automatically tagging and classifying images. You'll test the model and reflect on its strengths and weaknesses.

## Environment and Dependencies

- Download and install Miniconda.
- Create a new conda environment with Python 3.7.6:

       conda create --name udacity python=3.7.6 pytorch=1.11.0 torchvision torchaudio cudatoolkit -c pytorch

- Activate the environment:

      conda activate udacity

###### NOTE: You must activate the environment each time you open a new terminal.

- Install the required packages for the project:

      pip install -r requirements.txt

- Verify GPU functionality (only for machines with NVIDIA GPUs):

      python -c "import torch; print(torch.cuda.is_available())"

- You should receive a "True" response. If not, please check your NVIDIA drivers.

- Install and open Jupyter Lab:

      pip install jupyterlab
      jupyter lab

## The Data

- The dataset, named [Landmark Classification](https://udacity-dlnfd.s3-us-west-1.amazonaws.com/datasets/landmark_images.zip), comprises images of over 50 landmarks.

## CNN from Scratch

- A custom Convolutional Neural Network (CNN) architecture was designed and trained from scratch to classify landmarks. The model was tailored to the project's requirements and achieved 54% accuracy.

  - Model Architecture: The network consists of 5 convolutional layers to capture complex features. Dropout layers were used to combat overfitting. The model outputs a 50-dimensional vector to match the 50 available landmark classes.
  - Data Preprocessing: Images were resized to 256 and then cropped to 224, aligning with pytorch's pre-trained models. Data augmentation was performed using RandAugment to enhance model robustness.
  - Training and Validation: Training involved 250 epochs with an SGD optimizer and a learning rate scheduler. Weights were saved with the lowest loss.

  Accuracy: 54%

## Transfer Learning

- Leveraging pre-trained CNN models for fine-tuning significantly boosts performance. In this approach, the ResNet50 model was chosen as the base model, considering its depth and extensive training on a large dataset.

  - Pre-trained Model Selection: ResNet50 was adopted due to its depth and popularity.
  - Training and Validation: Similar to the CNN from scratch with minor adjustments:
    - Batch Size: Set to 64 for stochastic gradient descent (or Adam).
    - Number of Epochs: Reduced to 30.
    - Learning Rate: Increased to 0.002.
    - Optimizer: Utilized Adam for optimization.
    - Regularization: Weight decay set to 0.0001 to combat overfitting.

  Accuracy: 70%

## Future Upgrade Scope

- While the transfer learning approach achieved a commendable 70% accuracy, there is ample room for improvement. Several enhancements can be explored to further boost model performance:

  - **Extended Training**: The model was trained for 30 epochs. Consider extending the training duration, which often leads to improved performance. Experiment with varying the number of epochs to find the optimal balance between accuracy and training time.

  - **Hyperparameter Tuning**: Fine-tuning hyperparameters such as learning rate, batch size, and weight decay can significantly impact model accuracy. A systematic hyperparameter search can help in achieving higher accuracy levels.

  - **Advanced Data Augmentation**: Implement advanced data augmentation techniques to increase the model's robustness and generalization capabilities. Techniques like CutMix, MixUp, and AutoAugment can be beneficial.

  - **Ensemble Methods**: Explore ensemble methods that combine multiple models or variations of the same model to enhance prediction accuracy. This can provide a substantial accuracy boost.

  - **Model Selection**: While ResNet50 was used in this project, experimenting with different pre-trained models can lead to better results. Models such as EfficientNet, Inception, or VGG might perform differently on this specific task.

- By addressing these aspects and continuous model refinement, it's possible to push the model's accuracy to the range of 80-90%, making it even more reliable for landmark classification and tagging in social media applications.
