# Chessman Image Classification

Chessman Image Classification is a deep learning project that aims to classify different chess pieces (pawns, rooks, knights, bishops, queens, and kings) from digital images of a chessboard. The purpose of this project is to develop an automated system for recognizing and categorizing chess pieces in a digital image, which can be useful for various chess-related applications such as game analysis, chess engines, and AI-powered chess tutors. The functionality of this project involves the use of computer vision and machine learning algorithms to extract features from the digital images and train a classifier to accurately categorize the chess pieces. The output of this project is a model that can recognize and classify new chess pieces with high accuracy, based on the images it has seen during training.


## Requirements

The following packages are required to run the notebook:

- matplotlib
- numpy
- tensorflow
- pandas
- splitfolders
- opencv-python
- seaborn
- scikit-learn
- squarify

You can install these packages using the following pip command:

```bash
  pip install matplotlib numpy tensorflow pandas split-folders opencv-python seaborn scikit-learn squarify
```

## Usage

Here's how you can use the code in this project:

1. Clone or download the repository to your local machine.
 
2. Download the dataset from Kaggle (link provided below).

3. Open the Jupyter Notebook file in the repository.

4. Run the cells in the notebook to train the model and make predictions.

5. You can modify the code to fit your needs, such as changing the model architecture, training parameters, and prediction logic.

Note: Make sure to activate the virtual environment each time you work on the project to have the required packages installed.

## Dataset

The dataset used in this project is the Chessman Image Dataset and can be obtained from [Kaggle](https://www.kaggle.com/datasets/niteshfre/chessman-image-dataset).

The dataset contains 556 total images divided into 6 categories: "Bishop", "King", "Knight", "Pawn", "Queen", and "Rook".

Note: Please review and abide by the terms of use and license information for the dataset before using it in your own projects.

## Model


The model used in this project is based on VGG19, with additional layers added for the chessman image classification task. The model architecture is as follows:

- A preprocessing layer for data augmentation
- A rescaling layer to scale pixel values to the range [0, 1]
- The VGG19 model, with the top layer removed, as a base model
- A dropout layer with a rate of 0.4 to prevent overfitting
- Two Conv2D layers with 256 filters each, activation function ReLU, and padding set to 'same'
- Another dropout layer with a rate of 0.5
- A GlobalMaxPooling2D layer
- Another dropout layer with a rate of 0.6
- A dense layer with 6 neurons
- An activation layer with the activation function set to softmax

The model was compiled with categorical crossentropy loss, Adam optimizer, and accuracy metric.

## Training Process


The model was trained for 25 epochs using the `fit` method from the Keras API. The training data was passed to the method as an argument, along with the number of steps per epoch and the validation data. 

Three callbacks were added to monitor the training process: model checkpoint, learning rate reduction, and a CSV logger. 

After the initial training phase with the layers frozen, the model was fine-tuned by unfreezing 10 layers of the base model. The layers were set to be trainable and the model was trained again using the same process as before. 

## Evaluation

The model was evaluated on a test dataset which was 10% of the original dataset. The model achieved an accuracy score of 92.86% on the test dataset. A classification report and a confusion matrix were also generated to provide a more comprehensive understanding of the model's performance. The classification report displayed the precision, recall and f1-score for each of the 6 chessman categories. The confusion matrix provided a graphical representation of the number of correct and incorrect predictions made by the model. Both the classification report and the confusion matrix were plotted to visualize the results and to gain further insights into the model's performance.

## Future Work

- Training the model with more data or different datasets to improve the accuracy score.
- Experimenting with other image classification models such as ResNet, InceptionV3, and Xception to compare their performance.
- Incorporating additional layers to enhance the model's ability to identify chessman images.
- Exploring the use of Generative Adversarial Networks to generate new images of chessmen to add to the dataset.

## License

MIT


## Author

[Umar Saeed](https://www.linkedin.com/in/umar-saeed-16863a21b/)
