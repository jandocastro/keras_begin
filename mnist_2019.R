## MNIST and MNIST clothing neural nets
## Alejandro Castro iii
## 2019


library(keras)
#install_keras()
library(tidyverse)

mnist <- dataset_mnist()

glimpse(mnist)

# Load the data
x_train <- mnist$train$x
y_train <- mnist$train$y

x_test <- mnist$test$x
y_test <- mnist$test$y


# Reshaping
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))

# Rescale
x_train <- x_train / 255
x_test <-x_test / 255

# One-hot encoding for y
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)


# Let's create the model
model <- keras_model_sequential()
model %>%
  layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = 'softmax')

summary(model)

# Compile the model
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)


# Training and Evaluation
history <- model %>% fit(
  x_train, y_train, 
  epochs = 30, batch_size = 128,
  validation_split = 0.2
)

plot(history)

model %>% evaluate(x_test, y_test)

model %>% predict_classes(x_test)



## Fashion MNIST example

fashion_mnist <- dataset_fashion_mnist()

glimpse(fashion_mnist)

c(train_images, train_labels) %<-% fashion_mnist$train
c(test_images, test_labels) %<-% fashion_mnist$test

class_names <- c('T-shirt/top',
                'Trouser',
                'Pullover',
                'Dress',
                'Coat', 
                'Sandal',
                'Shirt',
                'Sneaker',
                'Bag',
                'Ankle boot')

dim(train_images)
dim(train_labels)

train_labels[1:20]

dim(test_images)
dim(test_labels)

image_1a <- as.data.frame(train_images[1, , ])
colnames(image_1a) <- seq_len(ncol(image_1a))
image_1a$y <- seq_len(nrow(image_1a))
image_1 <- gather(image_1a, "x", "value", -y)
image_1$x <- as.integer(image_1$x)

ggplot(image_1, aes(x = x, y = y, fill = value)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "black", na.value = NA) +
  scale_y_reverse() +
  theme_minimal() +
  theme(panel.grid = element_blank())   +
  theme(aspect.ratio = 1) +
  xlab("") +
  ylab("")

# Normalize the data : Preprocess
train_images <- train_images / 255
test_images <- test_images / 255

# Visualize the first 25 of the train data
par(mfcol=c(5,5))
par(mar=c(0, 0, 1.5, 0), xaxs='i', yaxs='i')
for (i in 1:25) { 
  img <- train_images[i, , ]
  img <- t(apply(img, 2, rev)) 
  image(1:28, 1:28, img, col = gray((0:255)/255), xaxt = 'n', yaxt = 'n',
        main = paste(class_names[train_labels[i] + 1]))
}


# Let's build the model!

model <- keras_model_sequential()
model %>%
  layer_flatten(input_shape = c(28, 28)) %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')

model %>% compile(
  optimizer = 'adam',
  loss = 'sparse_categorical_crossentropy',
  metrics = c('accuracy')
)

model %>% fit(train_images, train_labels, epochs = 10)

score <- model %>% evaluate(test_images, test_labels)

cat('Test loss:', score$loss, "\n")
cat('Test accuracy:', score$acc, "\n")


# Make Predictions
predictions <- model %>% predict(test_images)

predictions[1, ]

which.max(predictions[1, ])

class_pred <- model %>% predict_classes(test_images)
class_pred[1:20]

test_labels[1]

# Print test images with prediction labels
par(mfcol = c(5, 5))
par(mar = c(0, 0, 1.5, 0), xaxs = 'i', yaxs = 'i')
for (i in 1:25) {
  img <- test_images[i, , ]
  img <- t(apply(img, 2, rev))
  predicted_label <- class_pred[i]
  true_label <- test_labels[i]
  if (predicted_label == true_label) {
    color <- '#008800'
  } else {
    color <- '#bb0000'
  }
  image(1:28, 1:28, img, col = gray((0:255)/255), xaxt = 'n', yaxt = 'n',
        main = paste0(class_names[predicted_label + 1], " (",
                      class_names[true_label + 1], ")"),
        col.main = color)
  
}

# Overfitting because train accuracy higher than test accuracy
# Could use dropout, L2 regularization
# Then maybe an additional layer or convulational layers?





