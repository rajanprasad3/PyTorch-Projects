## PyTorch Projects

A collection of hands-on **Deep Learning projects implemented using PyTorch**, covering Computer Vision, Natural Language Processing (NLP), Audio Classification, and Tabular Data Classification. This repository demonstrates practical implementations of neural networks, convolutional neural networks, transfer learning, pretrained transformer models, and custom dataset pipelines.

Each project focuses on applying deep learning concepts to different types of data, from images and audio signals to text and structured datasets. The notebooks include data preprocessing, dataset preparation, model development, training, and evaluation.

## Projects Overview

| No. | Project | Approach |
|---|---|---|---|
| 1 | Image Classification | CNN |
| 2 | Image Classification Using Pretrained Models | Transfer Learning |
| 3 | Text Classification Using BERT | Transformer |
| 4 | Audio Classification | Audio Deep Learning | CNN on Spectrograms |
| 5 | Tabular Classification | Structured Data | Neural Network |

## Projects and Datasets

### 1. Image Classification

**Notebook:** `Image_Classification.ipynb`

**Domain:** Computer Vision

**Dataset:** [Animal Faces Dataset on Kaggle](https://www.kaggle.com/datasets/andrewmvd/animal-faces)

This project focuses on building an image classification model using PyTorch to identify animal categories from images. It demonstrates how image data can be processed and passed through a neural network to learn visual patterns and predict class labels.

**Key concepts covered:**
- Image loading and preprocessing
- Image transformations using PyTorch
- Custom Dataset and DataLoader implementation
- Building a CNN-based classification model
- Training and validation workflows
- Loss calculation and backpropagation
- Model evaluation and prediction

**Objective:** Understand the fundamentals of image classification and learn how to build and train deep learning models for visual recognition tasks.

### 2. Image Classification Using Pretrained Models

**Notebook:** `Image_Classification_Pretrained.ipynb`

**Domain:** Computer Vision

**Dataset:** [Bean Leaf Lesions Classification Dataset on Kaggle](https://www.kaggle.com/datasets/marquis03/bean-leaf-lesions-classification)

This project explores transfer learning by applying a pretrained image classification architecture to a bean leaf image dataset. Instead of training an entire network from scratch, it uses pretrained visual representations to support classification on a specific dataset.

**Key concepts covered:**
- Loading and preprocessing image datasets
- Working with pretrained models using `torchvision.models`
- Transfer learning for image classification
- Custom image datasets and data loaders
- Model training and optimization
- Prediction using a pretrained architecture
- Evaluating classification performance

**Objective:** Understand how pretrained deep learning models can be adapted to new image classification tasks and how transfer learning can reduce the need for training large models from scratch.

### 3. Text Classification Using BERT

**Notebook:** `Text_classification_using_Bert.ipynb`

**Domain:** Natural Language Processing (NLP)

**Dataset:** [News Headlines Dataset for Sarcasm Detection on Kaggle](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection)

This project applies BERT, a pretrained transformer-based language model, to a text classification task involving sarcasm detection in news headlines. It demonstrates how contextual language representations can be used to classify text based on its linguistic patterns and meaning.

**Key concepts covered:**
- Text data preparation and preprocessing
- Tokenization and attention masks
- Working with pretrained BERT models
- Using the Hugging Face Transformers library with PyTorch
- Preparing inputs for transformer architectures
- Model training and fine-tuning
- Binary text classification
- Evaluating model predictions

**Objective:** Learn how pretrained transformer models can be applied to NLP classification problems and understand the practical workflow of using BERT for contextual text understanding.

### 4. Audio Classification

**Notebook:** `Audio_Classification.ipynb`

**Domain:** Audio Deep Learning

**Dataset:** [Quran Recitations for Audio Classification on Kaggle](https://www.kaggle.com/datasets/mohammedalrajeh/quran-recitations-for-audio-classification)

This project focuses on classifying Quran recitations using PyTorch. Audio recordings are processed into mel-spectrogram representations, allowing a convolutional neural network to learn patterns associated with different reciters.

**Key concepts covered:**
- Loading and processing audio files
- Audio feature extraction using Librosa
- Generating mel-spectrograms
- Converting audio signals into image-like representations
- Building a custom PyTorch Dataset
- Creating training, validation, and testing datasets
- CNN-based audio classification
- Model training and performance evaluation

**Objective:** Understand how raw audio signals can be transformed into meaningful features and used with deep learning models for audio classification.

### 5. Tabular Classification

**Notebook:** `Tabular_Classification.ipynb`

**Domain:** Structured Data and Deep Learning

**Dataset:** [Rice Type Classification Dataset on Kaggle](https://www.kaggle.com/datasets/mssmartypants/rice-type-classification)

This project demonstrates the application of PyTorch neural networks to structured tabular data. It uses numerical features from rice samples to build a classification model capable of predicting rice types.

**Key concepts covered:**
- Loading and exploring structured datasets
- Handling missing values
- Removing unnecessary columns
- Feature and target separation
- Data preprocessing and train-validation-test splitting
- Converting data into PyTorch tensors
- Implementing a custom Dataset class
- Building a neural network using `nn.Module`
- Forward propagation and loss calculation
- Model optimization and evaluation

**Objective:** Understand how PyTorch can be used beyond image and text data to solve classification problems involving structured numerical features.

## Technologies and Libraries

The projects use the following tools and libraries, as applicable to each notebook:

- **Python:** Core programming language
- **PyTorch:** Neural network development and model training
- **Torchvision:** Image transformations and pretrained vision models
- **Hugging Face Transformers:** Pretrained transformer models such as BERT
- **Librosa:** Audio loading and feature extraction
- **NumPy:** Numerical operations
- **Pandas:** Data manipulation and preprocessing
- **Matplotlib:** Data visualization and training plots
- **Scikit-learn:** Data splitting, label encoding, and evaluation utilities
- **Jupyter Notebook:** Interactive development and experimentation

## Repository Structure

```text
PyTorch-Projects/
│
├── Image_Classification.ipynb
│
├── Image_Classification_Pretrained.ipynb
│
├── Text_classification_using_Bert.ipynb
│
├── Audio_Classification.ipynb
│
├── Tabular_Classification.ipynb
│
└── README.md
```

## Key Learning Outcomes

Through these projects, I have gained hands-on experience with the following concepts:

- Understanding PyTorch tensors and tensor operations
- Building neural networks using `torch.nn`
- Implementing custom Dataset classes and DataLoaders
- Preparing datasets for different deep learning applications
- Developing training and evaluation loops
- Applying loss functions, optimizers, and backpropagation
- Working with convolutional neural networks for visual and audio classification
- Applying transfer learning with pretrained models
- Using transformer-based architectures for text classification
- Converting audio signals into spectrogram representations
- Training neural networks on structured tabular data
- Evaluating model predictions and experimenting with model performance

## How to Use This Repository

1. Clone the repository:

   ```bash
   git clone https://github.com/rajanprasad3/PyTorch-Projects.git
   ```

2. Navigate to the project directory:

   ```bash
   cd PyTorch-Projects
   ```

3. Install the core dependencies:

   ```bash
   pip install torch torchvision numpy pandas matplotlib scikit-learn jupyter
   ```

4. Install additional libraries when required by a specific notebook:

   ```bash
   pip install transformers librosa opendatasets
   ```

5. Open Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

6. Select the notebook you want to explore and run its cells in order.

**Important:** Some notebooks were originally configured for Google Colab and use `/content/` paths. Update those paths when running locally. Download each dataset from its linked source and follow the dataset's directory structure. Access to Kaggle datasets may require a Kaggle account and acceptance of the applicable dataset terms.

## Future Improvements

- Experiment with additional CNN architectures and pretrained vision models.
- Compare model performance using different optimizers and hyperparameters.
- Explore more transformer architectures for NLP tasks.
- Experiment with advanced audio feature extraction techniques.
- Add classification reports, confusion matrices, and additional evaluation metrics.
- Save trained model weights and implement inference pipelines.
- Deploy selected projects as interactive applications using Streamlit or FastAPI.

## References

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Torchvision Documentation](https://pytorch.org/vision/stable/index.html)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)
- [Librosa Documentation](https://librosa.org/doc/latest/index.html)
- [Kaggle Datasets](https://www.kaggle.com/datasets)

## Author

**Rajan Prasad**

- GitHub: [rajanprasad3](https://github.com/rajanprasad3)

---

This repository documents my hands-on learning journey with PyTorch and deep learning, focusing on practical implementations across computer vision, natural language processing, audio classification, and structured data.

I will continue expanding this repository with new experiments, improved models, and additional deep learning projects as I strengthen my understanding of modern AI techniques.
