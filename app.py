# -*- coding: utf-8 -*-
import tensorflow_decision_forests as tfdf
import pandas as pd
import gradio as gr
import urllib
from tensorflow import keras


input_path = "https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income"
input_column_header = "income_level"

#Load data

BASE_PATH = input_path
CSV_HEADER = [ l.decode("utf-8").split(":")[0].replace(" ", "_")
  for l in urllib.request.urlopen(f"{BASE_PATH}.names")
  if not l.startswith(b"|")][2:]

CSV_HEADER.append(input_column_header)

train_data = pd.read_csv(f"{BASE_PATH}.data.gz", header=None, names=CSV_HEADER)
test_data = pd.read_csv(f"{BASE_PATH}.test.gz", header=None, names=CSV_HEADER)

#subset data
train_data = train_data.loc[:, ["education", "sex", "capital_gains", "capital_losses", "income_level"]]
test_data = test_data.loc[:, ["education", "sex", "capital_gains", "capital_losses", "income_level"]]

def encode_df(df):
    sex_mapping = {" Male": 0, " Female": 1}
    df = df.replace({"sex": sex_mapping})
    education_mapping = {" High school graduate": 1, " Some college but no degree": 2, 
                         " 10th grade": 3, " Children": 4, " Bachelors degree(BA AB BS)": 5, 
                         " Masters degree(MA MS MEng MEd MSW MBA)": 6, " Less than 1st grade": 7,
                         " Associates degree-academic program": 8, " 7th and 8th grade": 9,
                         " 12th grade no diploma": 10, " Associates degree-occup /vocational": 11,
                         " Prof school degree (MD DDS DVM LLB JD)": 12, " 5th or 6th grade": 13,
                         " 11th grade": 14, " Doctorate degree(PhD EdD)": 15, " 9th grade": 16,
                         " 1st 2nd 3rd or 4th grade": 17}
    df = df.replace({"education": education_mapping})
    income_mapping = {' - 50000.': 0, ' 50000+.': 1}
    df = df.replace({"income_level": income_mapping})
    return df

train_data = encode_df(train_data)
test_data = encode_df(test_data)

feature_a = tfdf.keras.FeatureUsage(name="education", semantic=tfdf.keras.FeatureSemantic.CATEGORICAL)
feature_b = tfdf.keras.FeatureUsage(name="sex", semantic=tfdf.keras.FeatureSemantic.CATEGORICAL)
feature_c = tfdf.keras.FeatureUsage(name="capital_gains", semantic=tfdf.keras.FeatureSemantic.CATEGORICAL)
feature_d = tfdf.keras.FeatureUsage(name="capital_losses", semantic=tfdf.keras.FeatureSemantic.CATEGORICAL)

# Convert the dataset into a TensorFlow dataset.
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_data, label="income_level")
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_data, label="income_level")

# Train a GB Trees model
model = tfdf.keras.GradientBoostedTreesModel(
      features = [feature_a, feature_b, feature_c, feature_d],
      exclude_non_specified_features = True,
      growing_strategy = "BEST_FIRST_GLOBAL",
      num_trees = 350,
      max_depth = 7,
      min_examples = 6,
      subsample = 0.65,
      sampling_method = "GOSS",
      validation_ratio = 0.1,
      task = tfdf.keras.Task.CLASSIFICATION,
      loss = "DEFAULT",
      verbose=0)

model.compile(metrics=[keras.metrics.BinaryAccuracy(name="accuracy")])
model.fit(train_ds)
model.evaluate(test_ds)

#prepare user input for the model
def process_inputs(education, sex, capital_gains, capital_losses):
  df = pd.DataFrame.from_dict(
      {
          "education": [edu_in], 
          "sex": [sex_in],
          "capital_gains": [cap_gains_in],
          "capital_losses": [cap_losses_in]    
      }
  )
  df = encode_df(df)
  
  feature_a = tfdf.keras.FeatureUsage(name="education", semantic=tfdf.keras.FeatureSemantic.CATEGORICAL)
  feature_b = tfdf.keras.FeatureUsage(name="sex", semantic=tfdf.keras.FeatureSemantic.CATEGORICAL)
  feature_c = tfdf.keras.FeatureUsage(name="capital_gains", semantic=tfdf.keras.FeatureSemantic.CATEGORICAL)
  feature_d = tfdf.keras.FeatureUsage(name="capital_losses", semantic=tfdf.keras.FeatureSemantic.CATEGORICAL)
  
  df = tfdf.keras.pd_dataframe_to_tf_dataset(df)
  
  pred = model.predict(df)
  if pred > .5:
    pred_bi = 1
    return {"> $50,000": pred_bi}
  elif pred <=.5:
    pred_bi = 0
    return {"<= $50,000": pred_bi}

iface = gr.Interface(
    process_inputs,
    [
     gr.inputs.Dropdown([" 1st 2nd 3rd or 4th grade", " High school graduate", 
                         " Bachelors degree(BA AB BS)", " Masters degree(MA MS MEng MEd MSW MBA)", 
                         " Prof school degree (MD DDS DVM LLB JD)",
                         " Doctorate degree(PhD EdD)"], type="index", label="education"), 
     gr.inputs.Radio([" Male", " Female"], label="sex", type="index"),
     gr.inputs.Slider(minimum = 0, maximum = 99999, label="capital_gains"),
     gr.inputs.Slider(minimum = 0, maximum = 4608, label="capital_losses")
    ],
    gr.outputs.Label(num_top_classes=2),
    live=True,
    analytics_enabled=False
)
