#  Loading CSV data processed for Tensorflow
#  Alejandro Castro iii
#  1/9/2020

library(keras)
library(tidyverse)
library(tfdatasets)


# Download and create datasets --------------------------------------------

# URLs : note the use of eval rather than test
train_data_url <- "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
test_data_url <- "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

train_file <- get_file("train.csv", train_data_url)
test_file <- get_file("eval.csv", test_data_url)

# Use a loader to prepare the data for tensorflow
train_ds <- make_csv_dataset(
  train_file,
  field_delim = ',',
  batch_size = 5,
  num_epochs = 1
)

test_ds <- make_csv_dataset(
  test_file,
  field_delim = ",",
  batch_size = 5,
  num_epochs = 1
)

# Look at an element of the data : each column is a tensor
train_ds %>%
  reticulate::as_iterator() %>%
  reticulate::iter_next() %>%
  reticulate::py_to_r()


# Data processing ---------------------------------------------------------

# Going to use tensorflow's built-in processing system: feature_spec

# Define the spec - continuous data
spec <- feature_spec(train_ds, survived ~ .)

spec <- spec %>%
  step_numeric_column(all_numeric())






