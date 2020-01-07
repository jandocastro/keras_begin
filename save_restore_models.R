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



