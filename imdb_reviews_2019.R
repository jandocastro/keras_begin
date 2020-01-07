## IMDB review neural nets
## Alejandro Castro iii
## 2019

library(keras)
library(tidyverse)


# Download / Explore Data -------------------------------------------------

imdb <- dataset_imdb(num_words = 10000)

c(train_data, train_labels) %<-% imdb$train
c(test_data, test_labels) %<-% imdb$test

word_index <- dataset_imdb_word_index()  # Word index mapping words to integers

paste0("Training entries: ", length(train_data), ", labels: ", length(train_labels))

#imdb$train$x[1]
train_data[[1]]

length(train_data[[1]])
length(train_data[[2]])  # Reviews can be different lengths, NNs need same length (will resolve)

word_index_df <- data.frame(
  word = names(word_index),
  idx = unlist(word_index, use.names = FALSE),
  stringsAsFactors = FALSE
)

head(word_index_df)

# First indices are reserved (think of deeplearning.ai)
word_index_df <- word_index_df %>%
  mutate(idx = idx + 3)
word_index_df <- word_index_df %>%
  add_row(word = "<PAD>", idx = 0) %>%
  add_row(word = "<START>", idx = 1) %>%
  add_row(word = "<UNK>",  idx = 2) %>%
  add_row(word = "<UNUSED>", idx = 3)

word_index_df <- arrange(word_index_df, idx)

decode_review <- function(text) {
  paste(map(text, function(number) word_index_df %>%
              filter(idx == number) %>%
              select(word) %>%
              pull()),
        collapse = " ")
}


# Prepare the Data --------------------------------------------------------

train_data <- pad_sequences(
  train_data,
  value = word_index_df %>% filter(word == "<PAD>") %>% select(idx) %>% pull(),
  padding = "post",
  maxlen = 256
)

length(train_data[1, ])
train_data[1, ]
dim(train_data)

test_data <- pad_sequences(
  test_data,
  value = word_index_df %>% filter(word == "<PAD>") %>% select(idx) %>% pull(),
  padding = "post",
  maxlen = 256
)

length(test_data[1, ])
test_data[1, ]


# Build the Model ---------------------------------------------------------

vocab_size <- 10000

model <- keras_model_sequential()
model %>%
  layer_embedding(input_dim = vocab_size, output_dim = 16) %>%
  layer_global_average_pooling_1d() %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

model %>% summary()
#summary(model)

model %>% compile(
  optimizer = 'adam',
  loss = 'binary_crossentropy',
  metrics = 'accuracy'
)


# Validation Set ----------------------------------------------------------

x_val <- train_data[1:10000, ]
partial_x_train <- train_data[10001:nrow(train_data), ]

y_val <- train_labels[1:10000]
partial_y_train <- train_labels[10001:length(train_labels)]


# Train the model ---------------------------------------------------------

history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 40,
  batch_size = 512,
  validation_data = list(x_val, y_val),
  verbose = 1
)


# Evaluate the model ------------------------------------------------------

results <- model %>% evaluate(test_data, test_labels)
results

plot(history)

preds <- model %>% predict(test_data)
preds_bin <- ifelse(preds >= 0.5, 1, 0)

head(preds_bin == test_labels, 20)

wrong_preds <- which(preds_bin != test_labels)
head(wrong_preds)

decode_review(test_data[4, ])
decode_review(test_data[9, ])
decode_review(test_data[23, ])  # Movie is Thursday


