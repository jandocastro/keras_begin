## keras :: Save and Restore Models
## Alejandro Castro iii
## 1/7/2020


# Import and Setup --------------------------------------------------------

# Load the usual libraries
library(keras)
library(tidyverse)

# Load data
mnist <- dataset_mnist()

c(train_images, train_labels) %<-% mnist$train
c(test_images, test_labels) %<-% mnist$test

# Take a look at the first train image/data
dim(train_images)
train_images[1, , ]

# Take a look at the first test image/data
dim(test_images)
test_images[1, , ]

# Limit to only first 1000 examples, reshape images, normalize by /255
train_images <- train_images[1:1000, , ] %>%
  array_reshape(c(1000, 28 * 28))
train_images <- train_images / 255

test_images <- test_images[1:1000, , ] %>%
  array_reshape(c(1000, 28 * 28))
test_images <- test_images / 255

train_labels <- train_labels[1:1000]
test_labels <- test_labels[1:1000]


# Define the model --------------------------------------------------------

# Function to facilitate building models
create_model <- function() {
  model <- keras_model_sequential() %>%
    layer_dense(units = 512, activation = 'relu', input_shape = 784) %>%
    layer_dropout(0.2) %>%
    layer_dense(units = 10, activation = 'softmax')
  
  model %>% compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = list("accuracy")
  )
  model
}

model <- create_model()
model %>% summary()


# Save the model ----------------------------------------------------------

# Will utilize the HDF5 format (Hierarchical Data Format)
# https://support.hdfgroup.org/HDF5/whatishdf5.html

model %>% fit(train_images, train_labels, epochs = 5)

model %>% save_model_hdf5('my_model.h5')

# if only wanted weights could use:
#model %>% save_model_weights_hdf5('my_model_weights.h5')

# Restoring the model
new_model <- load_model_hdf5('my_model.h5')
new_model %>% summary()


# Save checkpoints --------------------------------------------------------

# Useful to save checkpoints: may not need to retrain model (early stopping),
# pick up where you left off...

# Only going to focus on saving and restoring weights: would need model def on restore

checkpoint_dir <- "checkpoints"
dir.create(checkpoint_dir, showWarnings = FALSE)  # nice

filepath <- file.path(checkpoint_dir, "weights.{epoch:02d}-{val_loss:.2f}.hdf5")

# Create checkpoint callback
cp_callback <- callback_model_checkpoint(
  filepath = filepath,
  save_weights_only = TRUE,
  verbose = 1
)

model <- create_model()

model %>% fit(
  train_images,
  train_labels,
  epochs = 10,
  validation_data = list(test_images, test_labels),
  callbacks = list(cp_callback)
)

list.files(checkpoint_dir)

# Let's check: fresh untrained model, expect ~10% accuracy (10 classes)
new_model <- create_model()

score <- new_model %>% evaluate(test_images, test_labels)

cat('Test loss: ', score$loss, '\n')
cat('Test accuracy: ', score$acc, '\n')

# Now let's load the pre-trained weights : note, needs same architecture (from our function)
new_model %>% load_model_weights_hdf5(
  file.path(checkpoint_dir, "weights.10-0.40.hdf5")
)

score <- new_model %>% evaluate(test_images, test_labels)

cat('Test loss: ', score$loss, '\n')
cat('Test accuracy: ', score$acc, '\n')


# Save checkpoints: other options -----------------------------------------

# Every nth epoch
unlink(checkpoint_dir, recursive = TRUE)  # Delete the directory

dir.create(checkpoint_dir)

# New cp_callback
cp_callback <- callback_model_checkpoint(
  filepath = filepath,
  save_weights_only = TRUE,
  period = 5,
  verbose = 1
)

model <- create_model()

model %>% fit(
  train_images,
  train_labels,
  epochs = 10,
  validation_data = list(test_images, test_labels),
  callbacks = list(cp_callback)
)

list.files(checkpoint_dir)


# Or only best model using validation loss
unlink(checkpoint_dir, recursive = TRUE)

dir.create(checkpoint_dir)

# New cp_callback
cp_callback <- callback_model_checkpoint(
  filepath = filepath,
  save_weights_only = TRUE,
  save_best_only = TRUE,
  verbose = 1
)

model <- create_model()

model %>% fit(
  train_images,
  train_labels,
  epochs = 10,
  validation_data = list(test_images, test_labels),
  callbacks = list(cp_callback)
)

list.files(checkpoint_dir)


