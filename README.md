# TF-GB-Forest

Use TF's Gradient Boosted Trees model in binary classification of structured data 

Build a decision forests model by specifying the input feature usage.
Implement a custom Binary Target encoder as a Keras Preprocessing layer to encode the categorical features with respect to their target value co-occurrences, and then use the encoded features to build a decision forests model.
The model is implemented using Tensorflow 7.0 or higher. The US Census Income Dataset containing approximately 300k instances with 41 numerical and categorical variables was used to train it. This is a binary classification problem to determine whether a person makes over 50k a year.

Author: Khalid Salama
Adapted implementation: Tannia Dubon



These files were used to create a model card and a space in HuggingFace. 

Source: https://keras.io/examples/structured_data/classification_with_tfdf/
Space on Hugging Face: https://huggingface.co/spaces/keras-io/TF-GB-Forest
Model on Hugging Face: https://huggingface.co/keras-io/TF_Decision_Trees
