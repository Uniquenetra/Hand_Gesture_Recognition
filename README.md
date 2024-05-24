# Hand_Gesture_Recognition

This project is based on the architecture outlined in [this paper](https://github.com/guillaumephd/deep_learning_hand_gesture_recognition).

# I. Introduction

Gesture Recognition constitutes a vital subset within the realm of "computer vision," dedicated to deciphering human gestures through mathematical algorithms. While gestures can emanate from various bodily motions or states, they commonly originate from facial expressions or hand movements. This discipline encompasses diverse focuses, including emotion recognition from facial cues and the interpretation of hand gestures, enabling users to interact with devices through simple gestures without physical contact.

Numerous approaches have been explored, leveraging cameras and computer vision algorithms to interpret sign language. However, the scope of gesture recognition extends beyond sign language interpretation, encompassing the identification and understanding of postures, gaits, proxemics, and human behaviors. It represents a pivotal advancement, enabling computers to grasp human body language, thereby facilitating more intuitive interaction between humans and machines compared to conventional text-based or graphical user interfaces.

Gesture capture typically relies on two main methods:

1. **Wearable Sensors**: These include various sensor types such as flex, tactile, accelerometer, and gyroscope integrated into hand gloves. However, the utilization of wearable sensors presents limitations, notably in terms of cost-effectiveness, prompting current research to gravitate towards more straightforward computer vision methodologies.

2. **Computer Vision**: This technology harnesses a diverse array of cameras, including RGB, Depth, TOF (Time of Flight), Infrared, Thermal, and Stereo cameras, to capture a wide spectrum of images. Subsequently, machine learning algorithms process this data to segment and extract features such as skin color, skeletal structure, depth information, 3D models, and motion dynamics for further analysis.

Despite significant advancements, challenges persist in gesture recognition, spanning constraints imposed by diverse environmental conditions, such as scene background limitations and varying illumination conditions, as well as algorithmic accuracy for feature extraction, dataset characteristics, classification methodologies, and application-specific requirements.

# II. Problem Description

## Problem Statement:

"We require a model capable of identifying and classifying human hand gestures into predefined categories, either in real-time at each time step or upon completion of the gesture, utilizing captured 3D hand-skeletal data sequences."

Human gesture recognition falls within the broader framework of pattern recognition, involving two primary processes:

1. The acquisition process, which translates physical gestures into numerical data.
2. The interpretation process, which assigns meaning to the captured data.

### A. Importance of the Problem:

Gestures serve as a natural mode of communication, facilitating nonverbal intent communication among humans and enhancing human-computer interaction. By incorporating hand gestures, communication gains an additional layer of richness and nuance, akin to the subtle facial expressions accompanying verbal communication.

In today's context, where hygiene is paramount, gesture-based interaction offers a touch-free alternative, fostering safer human-computer interaction environments.

Moreover, hand gesture recognition holds significant potential for aiding sign language users, particularly individuals with hearing or speech impairments, enabling seamless communication within their community and with individuals who do not use sign language.

Furthermore, hand gestures play a pivotal role in surgical settings, allowing surgeons with covered mouths to communicate effectively while maintaining sterile conditions. The integration of gesture recognition technology in robotics within the medical field offers promising prospects, facilitating doctors' interactions with assistant robots during surgical procedures.

### B. Approaches to Addressing the Problem

Various approaches to hand gesture recognition based on computer vision have been explored, with recent advancements encompassing the following methods:

1. **Color-Based Recognition**: This approach relies on identifying skin color or colored gloves. Skin color detection typically involves converting the image to the YUV color space and applying a threshold to isolate skin tones. However, challenges arise under varying lighting conditions and when multiple objects share similar colors.

2. **Appearance-Based Recognition**: This method involves extracting image features to detect regions of interest (ROIs). Techniques such as Haar-like feature extraction and Adaboost classification are employed, along with background and foreground extraction. However, issues persist concerning lighting variations and background complexities.

3. **Motion-Based Recognition**: This technique involves extracting objects across a series of image frames using frame difference subtraction. It typically consists of two stages: hand detection and tracking in the first stage, followed by gesture classification in the second stage. Challenges include handling dynamic backgrounds.

4. **Skeleton-Based Recognition**: This approach utilizes geometric attributes such as joint orientation and spacing to detect complex features. It often integrates depth capture technologies like Kinect cameras and employs machine learning algorithms such as parallel convolutional neural networks (CNNs) or support vector machines (SVMs), along with Hidden Markov Models (HMMs) for classification.

5. **Depth-Based Recognition/3D Model**: This method leverages cameras such as Kinect to obtain 3D geometric information, enabling depth-based recognition.

6. **Deep Learning-Based Recognition**: This approach primarily utilizes convolutional neural networks (CNNs), which have demonstrated significant success in gesture recognition tasks.

While these techniques have shown promising results, most applications require classification of a small set of classes. Sign language recognition poses additional challenges due to the larger class set size.

In this project, we propose to build a model based on the methodology outlined in [2], falling under the category of deep learning-based recognition. Another relevant study, akin to our approach, utilizes depth images captured by Kinect sensors for color image segmentation, followed by skin color modeling and classification using CNNs and SVMs [3].


## C. Our Adaptation:

We began with a pre-trained deep model trained on 14 hand gestures using a multi-channel convolutional neural network (MC-DCNN) architecture. The model was trained on the Dynamic Hand Gesture-14/28 (DHG) dataset, which consists of 3D hand skeletal representations captured by the Intel RealSense depth camera. We adapted the model to recognize an additional hand gesture, "Rock on," by retraining it using transfer learning.

![C. Our Adaptation:](images/handgesture.jpeg)

This approach can be divided into three main stages:

### Hand Gesture Capture
This stage involves segmenting the hand region using depth and color information, resulting in a 3D sequence representing the hand gesture.

### Feature Extraction
Features and parameters are estimated from the 3D sequence inputs. This stage plays a crucial role in preparing the data for classification.

### Gesture Recognition or Classification
The focus of this study lies in the classification stage, where features extracted from hand-skeletal data are used as inputs to a deep learning model to classify them into known labels.

In the Capture stage, most models utilize RGB-D image sequences, but our approach utilizes hand skeletal data sequences, which allows for quicker processing.

Feature extraction can be achieved through explicit hand-crafted features or implicit feature learning using deep learning algorithms. While deep learning algorithms implicitly learn features that often better describe the data, they may suffer from drawbacks such as overfitting and lack of generalization to unseen data.

The skeletal data consists of a sequence of 22 points representing hand-skeleton joints, where each joint represents a distinct articulation or part of the hand.

![Gesture Recognition or Classification](images/hg1.png)

Convolutional Neural Networks (CNNs) process data through layers that perform convolution, pooling, and output dense layers. To effectively learn features, CNNs require a large volume of training data. Transfer learning and fine-tuning techniques are commonly employed to adapt pre-trained models to new tasks or datasets.

Incorporating external expert knowledge can lead to the use of simpler models or pre-processing techniques tailored to the specific task. For example, input data can be pre-processed to emphasize important features known to domain experts.

Sequence data exhibit time-domain dependencies, requiring specialized models such as Recurrent Neural Networks (RNNs) and Transformers. However, in this study, we propose a convolution-based architecture called Multi-channel Deep Convolutional Neural Networks (MC-DCNN), which does not use recurrent cells or attention mechanisms. MC-DCNNs process parallel unidimensional sequences and perform feature learning on each univariate sequence individually. This architecture is well-suited for skeleton data and offers advantages over other models by being agnostic to the skeleton's structure.

Overall, our approach combines deep learning techniques with specialized architectures tailored to the unique characteristics of hand gesture recognition tasks.

### Equations:

#### Centroid Calculation:
C = ∑ i=1 to k x_i/k

Where:
- C is the centroid of the set of points.
- x_i are the individual points in the set.
- k is the number of points in the set.

#### Sequence Data Representation:
s = (p_1, p_2, ..., p_n)^T 
p_i = (x(i), y(i), z(i))

Where:
- s represents the sequence data of hand-skeletal joints.
- p_i represents a specific skeletal joint.
- (x(i), y(i), z(i) are the three components representing the position of the i-th skeletal joint.

## Decision Highlights:

1. **Use Transfer Learning or Train from Scratch?**
   - **Decision**: Transfer learning was chosen over training from scratch due to its efficiency and suitability for small datasets.
   - **Explanation**: Transfer learning involves leveraging pre-trained models on large datasets and fine-tuning them for the specific task at hand. This approach is beneficial when working with limited data as it allows the model to leverage knowledge learned from other tasks or domains, leading to faster convergence and potentially better performance.

2. **Feature Selection or Pre-Processing for the Missing Palm Landmark?**
   - **Decision**: Pre-processing was chosen to ensure dataset completeness without sacrificing performance.
   - **Explanation**: Pre-processing techniques were employed to address the issue of missing palm landmarks in the dataset. This involved handling missing data through interpolation or other methods to ensure that the dataset was complete and suitable for training the model. By ensuring dataset completeness, the model could learn effectively without being hindered by missing information.

3. **Data Augmentation using SMOTE Techniques?**
   - **Decision**: Traditional data augmentation techniques were not applicable due to the nature of the data.
   - **Explanation**: SMOTE (Synthetic Minority Over-sampling Technique) is commonly used to address class imbalance by generating synthetic samples for minority classes. However, in the context of gesture recognition, the nature of the data may not lend itself well to traditional augmentation techniques. Since gestures are inherently diverse and complex, generating synthetic samples may not accurately represent real-world variations in gestures. Therefore, alternative approaches to address class imbalance or enhance dataset diversity may be more appropriate.

4. **Does the Model Over-fit if Re-trained on the Same 14 Gestures?**
   - **Decision**: Experiments demonstrated the model's robustness and validated the effectiveness of transfer learning.
   - **Explanation**: The experiments conducted involved re-training the model on the same 14 gestures to assess its performance and potential for overfitting. Overfitting occurs when a model learns to memorize the training data instead of generalizing to unseen data. However, the results of the experiments indicated that the model remained robust and did not exhibit signs of overfitting, thereby validating the effectiveness of transfer learning in this context.

5. **Do We Need Architectural Changes?**
   - **Decision**: Minor modifications to the existing architecture were made to balance complexity and performance effectively.
   - **Explanation**: While the existing architecture may have performed adequately, minor modifications were deemed necessary to optimize its performance further. These modifications could include adjusting hyperparameters, adding regularization techniques, or introducing architectural changes to improve the model's ability to capture complex patterns in the data. By balancing complexity and performance, the modified architecture could achieve better results without introducing unnecessary overhead.


