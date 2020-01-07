## Boston Housing Prices NNs
## Alejandro  Castro iii
## 12/10/2019

library(tidyverse)
library(keras)


# Import and explore data -------------------------------------------------

boston_housing <- dataset_boston_housing()

c(train_data, train_labels) %<-% boston_housing$train
c(test_data, test_labels) %<-% boston_housing$test

paste0("Training entries: ", dim(train_data)[1],
       ", features: ", dim(train_data)[2],
       ", labels: ", length(train_labels))

train_data[1, ]  # to see the different measurement scales
summary(train_data)

column_names <- c('crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age',
                  'dis', 'rad','tax', 'ptratio', 'b', 'lstat')
train_df <- as.tibble(train_data)
colnames(train_df) <- column_names

train_df

train_labels[1:10]  # in thousands of dollars (mid-1970's prices)


# Normalize data ----------------------------------------------------------

# Test data is NOT used when normalizing
train_data <- scale(train_data)
summary(train_data)

# Use means and SD from training set to normalize test data
col_means_train <- attr(train_data, "scaled:center")
col_stddevs_train <- attr(train_data, "scaled:scale")
test_data <- scale(test_data, center = col_means_train, scale = col_stddevs_train)

train_data[1, ]


# Create the Model --------------------------------------------------------

build_model <- function() {
  
  model <- keras_model_sequential() %>%
    layer_dense(units = 64, activation = 'relu',
                input_shape = dim(train_data)[2]) %>%
    layer_dense(units = 64, activation = 'relu') %>%
    layer_dense(units = 1)
  
  model %>% compile(
    loss = 'mse',
    optimizer = optimizer_rmsprop(),
    metrics = list("mean_absolute_error"))
  
  model
}

model <- build_model()
summary(model)


# Train the model ---------------------------------------------------------

# Custom display for training progress
print_dot_callback <- callback_lambda(
  on_epoch_end = function(epoch, logs) {
    if (epoch  %% 80 == 0) cat("\n")
    cat(".")
  }
)

epochs <- 500

# Fit model and store training stats
history <- model %>% fit(
  train_data,
  train_labels,
  epochs = epochs,
  validation_split = 0.2,
  verbose = 0,
  callbacks = list(print_dot_callback)
)

plot(history, metrics = "mean_absolute_error", smooth = FALSE) +
  coord_cartesian(ylim = c(0, 5))

# Early stopping, patience parameter is # of epochs to check for improvement
early_stop <- callback_early_stopping(monitor = 'val_loss', patience = 20)

model <- build_model()
history <- model %>% fit(
  train_data,
  train_labels,
  epochs = epochs,
  validation_split = 0.2,
  verbose = 0,
  callbacks = list(early_stop, print_dot_callback)
)

plot(history, metrics = "mean_absolute_error", smooth = FALSE) + 
  coord_cartesian(xlim = c(0, 150), ylim = c(0, 5))

# Test set performance
c(loss, mae) %<-% (model %>% evaluate(test_data, test_labels, verbose = 0))

paste0("Mean absolute error on test set: $", sprintf("%.2f", mae * 1000))


# Predict -----------------------------------------------------------------

test_predictions <- model %>% predict(test_data)

test_predictions[, 1]
test_labels
