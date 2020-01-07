## Sunspot Timeseries Predictions using Keras and Tensorflow
## Alejandro Castro iii
## 8/9/2019
## R vers. 3.5.1

library(tidyverse)
library(glue)
library(forcats)

library(timetk)
library(tidyquant)
library(tibbletime)

library(cowplot)
library(recipes)

library(rsample)
library(yardstick)

library(keras)
library(tfruns)

#install_keras()


# Download and explore data -----------------------------------------------

## Data
sun_spots <- sunspot.month %>%
  tk_tbl() %>%
  mutate(index = as_date(index)) %>%
  as_tbl_time(index = index)

## Graphing timeseries
p1 <- ggplot(sun_spots, aes(index, value))

p1a <- p1 + geom_point() + 
  theme_tq() + 
  labs(title = "1749 to 2013")

p2 <- sun_spots %>%
  filter_time("start" ~ "1800") %>%
  ggplot(aes(index, value))

p2a <- p2 + geom_line(color = palette_light()[[1]], alpha = 0.5) +
  geom_point() +
  geom_smooth(span = 0.2, se = FALSE) +
  theme_tq() +
  labs(title = "Zoom")

p_title <- ggdraw() +
  draw_label("Sunspots", size = 18, fontface = "bold")

plot_grid(p_title, p1a, p2a, ncol = 1, rel_heights = c(0.1, 1, 1))


# Training/Test data ------------------------------------------------------

## Create training and testing data
periods_train <- 12 * 100
periods_test <- 12 * 50
skip_span <- 12 * 22 - 1

rolling_origin_resamples <- rolling_origin(
  sun_spots,
  initial = periods_train,
  assess = periods_test,
  cumulative = FALSE,
  skip = skip_span
)

rolling_origin_resamples
# typeof(rolling_origin_resamples)
# glimpse(rolling_origin_resamples)
# length(rolling_origin_resamples$splits)
# rolling_origin_resamples$splits[[2]]


## Plotting the sampling data
plot_split <- function(split, expand_y_axis = TRUE,
                       alpha = 1, size = 1, base_size = 14) {
  # Manipulate data
  train_tbl <- training(split) %>%
    add_column(key = "training")
  
  test_tbl <- testing(split) %>%
    add_column(key = "testing")
  
  data_manipulated <- bind_rows(train_tbl, test_tbl) %>%
    as_tbl_time(index = index) %>%
    mutate(key = fct_relevel(key, "training", "testing"))
  
  # Collecting Attributes
  train_time_summary <- train_tbl %>%
    tk_index() %>%
    tk_get_timeseries_summary()
  
  test_time_summary <- test_tbl %>%
    tk_index() %>%
    tk_get_timeseries_summary()
  
  # Visualize
  g <- data_manipulated %>%
    ggplot(aes(x = index, y = value, color = key)) +
    geom_line(size = size, alpha = alpha) +
    # theme_tq(base_size = base_size) +
    # scale_color_tq() + 
    labs(title = glue("{split$id}"),
         subtitle = glue("{train_time_summary$start} to ",
                         "{test_time_summary$end}"),
         y = "", x = "") +
    theme(legend.position = "none")
  
  if (expand_y_axis) {
    sun_spots_time_summary <- sun_spots %>%
      tk_index() %>%
      tk_get_timeseries_summary()
    
    g <- g + scale_x_date(limits = c(sun_spots_time_summary$start,
                                     sun_spots_time_summary$end))
  
  }
  
  g

}

rolling_origin_resamples$splits[[1]] %>%
  plot_split(expand_y_axis = TRUE) +
  theme(legend.position = "bottom")

plot_sampling_plan <- function(sampling_tbl, expand_y_axis = TRUE,
                               ncol = 3, alpha = 1, size = 1, base_size = 14,
                               title = "Sampling Plan") {
  
  # Map plot_split() to sampling_tbl
  sampling_tbl_with_plots <- sampling_tbl %>%
    mutate(gg_plots = map(splits, plot_split,
                          expand_y_axis = expand_y_axis,
                          alpha = alpha, base_size = base_size))
  
  # Make plots with cowplots
  plot_list <- sampling_tbl_with_plots$gg_plots
  
  p_temp <- plot_list[[1]] + theme(legend.position = "bottom")
  legend <- get_legend(p_temp)
  
  p_body <- plot_grid(plotlist = plot_list, ncol = ncol)
  
  p_title <- ggdraw() +
    draw_label(title, size = 14, fontface = "bold", 
               colour = palette_light()[[1]])
  
  g <- plot_grid(p_title, p_body, legend, ncol = 1,
                 rel_heights = c(0.5, 1, 0.05))
  
  g
  
}

rolling_origin_resamples %>%
  plot_sampling_plan(expand_y_axis = TRUE, ncol = 3, alpha = 1,
                     size = 1, base_size = 10,
                     title = "Backtesting Strategy")


# Process Data --------------------------------------------------------

## LSTM model
example_split <- rolling_origin_resamples$splits[[6]]
example_split_id <- rolling_origin_resamples$id[[6]]

plot_split(example_split, expand_y_axis = FALSE, size = 0.5) +
  theme(legend.position = "bottom") +
  ggtitle(glue("{example_split_id}"))

# Get data.frame from split object
df_trn <- analysis(example_split)[1:800, , drop = FALSE]
df_val <- analysis(example_split)[801:1200, , drop = FALSE]
df_tst <- assessment(example_split)

df <- bind_rows(df_trn %>% add_column(key = "training"),
                df_val %>% add_column(key = "validation"),
                df_tst %>% add_column(key = "testing")) %>%
  as_tbl_time(index = index)
df

## Center and Scale (and sqrt transform to reduce variance)
rec_obj <- recipe(value ~ ., df) %>%
  step_sqrt(value) %>%
  step_center(value) %>%
  step_scale(value) %>%
  prep()

df_processed_tbl <- bake(rec_obj, df)

df_processed_tbl

# Not sure if centering ALL the data is the right thing!!
df %>% group_by(key) %>% summarize(mean = mean(sqrt(value)))

## Get mean and scale
center_history <- rec_obj$steps[[2]]$means["value"]
scale_history <- rec_obj$steps[[3]]$sds["value"]

c(center_history, scale_history)


## Reshape the data for Keras and Tensorflow
n_timesteps <- 12
n_predictions <- n_timesteps
batch_size <- 10

build_matrix <- function(tseries, overall_timesteps) {
  t(sapply(1:(length(tseries) - overall_timesteps + 1), function(x)
    tseries[x:(x + overall_timesteps - 1)]))
}

# test <- 1:30
# test1 <- build_matrix(test, 3)
# test2 <- sapply(1:(length(test) - 3 + 1), function(x) test[x:(x + 3 - 1)])
# t(test2)

reshape_X_3d <- function(X) {
  dim(X) <- c(dim(X)[1], dim(X)[2], 1)
  X
}

# extract values from data frame
train_vals <- df_processed_tbl %>%
  filter(key == "training") %>%
  select(value) %>%
  pull()

valid_vals <- df_processed_tbl %>%
  filter(key == "validation") %>%
  select(value) %>%
  pull()

test_vals <- df_processed_tbl %>%
  filter(key == "testing") %>%
  select(value) %>%
  pull()


# Building matrices
train_matrix <- build_matrix(train_vals, n_timesteps + n_predictions)
valid_matrix <- build_matrix(valid_vals, n_timesteps + n_predictions)
test_matrix <- build_matrix(test_vals, n_timesteps + n_predictions)


# Separate matrices parts
X_train <- train_matrix[, 1:n_timesteps]
y_train <- train_matrix[, (n_timesteps + 1):(n_timesteps * 2)]
X_train <- X_train[1:(nrow(X_train) %/% batch_size * batch_size), ]
y_train <- y_train[1:(nrow(y_train) %/% batch_size * batch_size), ]

X_valid <- valid_matrix[, 1:n_timesteps]
y_valid <- valid_matrix[, (n_timesteps + 1):(n_timesteps * 2)]
X_valid <- X_valid[1:(nrow(X_valid) %/% batch_size * batch_size), ]
y_valid <- y_valid[1:(nrow(y_valid) %/% batch_size * batch_size), ]

X_test <- test_matrix[, 1:n_timesteps]
y_test <- test_matrix[, (n_timesteps + 1):(n_timesteps * 2)]
X_test <- X_test[1:(nrow(X_test) %/% batch_size * batch_size), ]
y_test <- y_test[1:(nrow(y_test) %/% batch_size * batch_size), ]

# Adding required 3d axis
X_train <- reshape_X_3d(X_train)
X_valid <- reshape_X_3d(X_valid)
X_test <- reshape_X_3d(X_test)

y_train <- reshape_X_3d(y_train)
y_valid <- reshape_X_3d(y_valid)
y_test <- reshape_X_3d(y_test)

# FLAGS :: See web for description of particular flags
FLAGS <- flags(flag_boolean("stateful", FALSE),
               flag_boolean("stack_layers", FALSE),
               flag_integer("batch_size", 10),
               flag_integer("n_timesteps", 12),
               flag_integer("n_epochs", 100),
               flag_numeric("dropout", 0.2),
               flag_numeric("recurrent_dropout", 0.2),
               flag_string("loss", "logcosh"),
               flag_string("optimizer_type", "sgd"),
               flag_integer("n_units", 128),
               flag_numeric("lr", 0.003),
               flag_numeric("momentum", 0.9),
               flag_integer("patience", 10))

n_predictions <- FLAGS$n_timesteps
n_features <- 1
optimizer <- switch(FLAGS$optimizer_type,
                    sgd = optimizer_sgd(lr = FLAGS$lr,
                                        momentum = FLAGS$momentum))

callbacks <- list(callback_early_stopping(patience = FLAGS$patience))


# Build the model ---------------------------------------------------------

# -----------------------------------
# Create the model
# First, the long, more flexible version where you can stack layers and use stateful layers
model <- keras_model_sequential()

model %>%
  layer_lstm(units = FLAGS$n_units,
             batch_input_shape = c(FLAGS$batch_size, FLAGS$n_timesteps, n_features),
             dropout = FLAGS$dropout,
             recurrent_dropout = FLAGS$recurrent_dropout,
             return_sequences = TRUE,
             stateful = FLAGS$stateful)
if (FLAGS$stack_layers) {
  model %>%
    layer_lstm(units = FLAGS$n_units,
               dropout = FLAGS$dropout,
               recurrent_dropout = FLAGS$recurrent_dropout,
               return_sequences = TRUE,
               stateful = FLAGS$stateful)
}

model %>% time_distributed(layer_dense(units = 1))

model %>% compile(loss = FLAGS$loss,
                  optimizer = optimizer,
                  metrics = list("mean_squared_error"))

if (!FLAGS$stateful) {
  model %>% fit(x = X_train,
                y = y_train,
                validation_data = list(X_valid, y_valid),
                batch_size = FLAGS$batch_size,
                epochs = FLAGS$n_epochs,
                callbacks = callbacks)
} else {
  for(i in 1:FLAGS$n_epochs) {
    model %>% fit(x = X_train,
                  y = y_train,
                  validation_data = list(X_valid, y_valid),
                  callbacks = callbacks,
                  batch_size = FLAGS$batch_size,
                  epochs = 1,
                  shuffle = FALSE)
    model %>% reset_states()
  }
}

# Now to build the simpler model that still performs well
model <- keras_model_sequential()

model %>%
  layer_lstm(units = FLAGS$n_units,
             batch_input_shape = c(FLAGS$batch_size, FLAGS$n_timesteps, n_features),
             dropout = FLAGS$dropout,
             recurrent_dropout = FLAGS$recurrent_dropout,
             return_sequences = TRUE) %>%
  time_distributed(layer_dense(units = 1))

model %>% compile(loss = FLAGS$loss,
                  optimizer = optimizer,
                  metrics = list("mean_squared_error"))

history <- model %>% fit(x = X_train,
                         y = y_train,
                         validation_data = list(X_valid, y_valid),
                         batch_size = FLAGS$batch_size,
                         epochs = FLAGS$n_epochs,
                         callbacks = callbacks)

plot(history, metrics = "loss")


# Let's look at the predicted vs train
pred_train <- model %>%
  predict(X_train, batch_size = FLAGS$batch_size) %>%
  .[, , 1]

# Back-transform data
pred_train <- (pred_train * scale_history + center_history) ^2
compare_train <- df %>% filter(key == "training")

for(i in 1:nrow(pred_train)) {
  varname <- paste0("pred_train", i)
  compare_train <- 
    mutate(compare_train, 
           !!varname := c(rep(NA, FLAGS$n_timesteps + i - 1),
                          pred_train[1, ],
                          rep(NA, nrow(compare_train) - FLAGS$n_timesteps * 2 - i + 1)))
}









