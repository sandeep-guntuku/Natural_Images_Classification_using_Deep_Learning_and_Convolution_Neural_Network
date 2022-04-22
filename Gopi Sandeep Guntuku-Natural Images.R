#Initializing the libraries

library(keras)
library(tensorflow)
library(tidyverse)
library(imager)
library(caret)

#Setting original data set directories

original_dataset_airplane_dir <- "natural_images/airplane"
original_dataset_car_dir <- "natural_images/car"
original_dataset_cat_dir <- "natural_images/cat"
original_dataset_dog_dir <- "natural_images/dog"
original_dataset_flower_dir <- "natural_images/flower"
original_dataset_fruit_dir <- "natural_images/fruit"
original_dataset_motorbike_dir <- "natural_images/motorbike"
original_dataset_person_dir <- "natural_images/person"

#Setting base directory 

base_dir <- "natural_images_small" # to store a subset of data that we are going to use
dir.create(base_dir)

#Setting train, validation and train directories in base directory

train_dir <- file.path(base_dir, "train")
dir.create(train_dir)
validation_dir <- file.path(base_dir, "validation")
dir.create(validation_dir)
test_dir <- file.path(base_dir, "test")
dir.create(test_dir)

#Creating train folders to organize the data

train_airplane_dir <- file.path(train_dir, "airplane")
dir.create(train_airplane_dir)
train_car_dir <- file.path(train_dir, "car")
dir.create(train_car_dir)
train_cat_dir <- file.path(train_dir, "cat")
dir.create(train_cat_dir)
train_dog_dir <- file.path(train_dir, "dog")
dir.create(train_dog_dir)
train_flower_dir <- file.path(train_dir, "flower")
dir.create(train_flower_dir)
train_fruit_dir <- file.path(train_dir, "fruit")
dir.create(train_fruit_dir)
train_motorbike_dir <- file.path(train_dir, "motorbike")
dir.create(train_motorbike_dir)
train_person_dir <- file.path(train_dir, "person")
dir.create(train_person_dir)

#Creating validation folders to organize the data

validation_airplane_dir <- file.path(validation_dir, "airplane")
dir.create(validation_airplane_dir)
validation_car_dir <- file.path(validation_dir, "car")
dir.create(validation_car_dir)
validation_cat_dir <- file.path(validation_dir, "cat")
dir.create(validation_cat_dir)
validation_dog_dir <- file.path(validation_dir, "dog")
dir.create(validation_dog_dir)
validation_flower_dir <- file.path(validation_dir, "flower")
dir.create(validation_flower_dir)
validation_fruit_dir <- file.path(validation_dir, "fruit")
dir.create(validation_fruit_dir)
validation_motorbike_dir <- file.path(validation_dir, "motorbike")
dir.create(validation_motorbike_dir)
validation_person_dir <- file.path(validation_dir, "person")
dir.create(validation_person_dir)

#Creating test folders to organize the data

test_airplane_dir <- file.path(test_dir, "airplane")
dir.create(test_airplane_dir)
test_car_dir <- file.path(test_dir, "car")
dir.create(test_car_dir)
test_cat_dir <- file.path(test_dir, "cat")
dir.create(test_cat_dir)
test_dog_dir <- file.path(test_dir, "dog")
dir.create(test_dog_dir)
test_flower_dir <- file.path(test_dir, "flower")
dir.create(test_flower_dir)
test_fruit_dir <- file.path(test_dir, "fruit")
dir.create(test_fruit_dir)
test_motorbike_dir <- file.path(test_dir, "motorbike")
dir.create(test_motorbike_dir)
test_person_dir <- file.path(test_dir, "person")
dir.create(test_person_dir)

#Copying images from original data set to airplane train folder

fnames <- paste0("airplane_000", 1:9, ".jpg")
fnames1 <- paste0("airplane_00", 10:99, ".jpg")
fnames2 <- paste0("airplane_0", 100:560, ".jpg")
file.copy(file.path(original_dataset_airplane_dir, fnames), file.path(train_airplane_dir))
file.copy(file.path(original_dataset_airplane_dir, fnames1), file.path(train_airplane_dir))
file.copy(file.path(original_dataset_airplane_dir, fnames2), file.path(train_airplane_dir))

#Copying images from original data set to car train folder

fnames <- paste0("car_000", 1:9, ".jpg")
fnames1 <- paste0("car_00", 10:99, ".jpg")
fnames2 <- paste0("car_0", 100:560, ".jpg")
file.copy(file.path(original_dataset_car_dir, fnames), file.path(train_car_dir))
file.copy(file.path(original_dataset_car_dir, fnames1), file.path(train_car_dir))
file.copy(file.path(original_dataset_car_dir, fnames2), file.path(train_car_dir))

#Copying images from original data set to cat train folder

fnames <- paste0("cat_000", 1:9, ".jpg")
fnames1 <- paste0("cat_00", 10:99, ".jpg")
fnames2 <- paste0("cat_0", 100:560, ".jpg")
file.copy(file.path(original_dataset_cat_dir, fnames), file.path(train_cat_dir))
file.copy(file.path(original_dataset_cat_dir, fnames1), file.path(train_cat_dir))
file.copy(file.path(original_dataset_cat_dir, fnames2), file.path(train_cat_dir))

#Copying images from original data set to dog train folder

fnames <- paste0("dog_000", 1:9, ".jpg")
fnames1 <- paste0("dog_00", 10:99, ".jpg")
fnames2 <- paste0("dog_0", 100:560, ".jpg")
file.copy(file.path(original_dataset_dog_dir, fnames), file.path(train_dog_dir))
file.copy(file.path(original_dataset_dog_dir, fnames1), file.path(train_dog_dir))
file.copy(file.path(original_dataset_dog_dir, fnames2), file.path(train_dog_dir))

#Copying images from original data set to flower train folder

fnames <- paste0("flower_000", 1:9, ".jpg")
fnames1 <- paste0("flower_00", 10:99, ".jpg")
fnames2 <- paste0("flower_0", 100:560, ".jpg")
file.copy(file.path(original_dataset_flower_dir, fnames), file.path(train_flower_dir))
file.copy(file.path(original_dataset_flower_dir, fnames1), file.path(train_flower_dir))
file.copy(file.path(original_dataset_flower_dir, fnames2), file.path(train_flower_dir))

#Copying images from original data set to fruit train folder

fnames <- paste0("fruit_000", 1:9, ".jpg")
fnames1 <- paste0("fruit_00", 10:99, ".jpg")
fnames2 <- paste0("fruit_0", 100:560, ".jpg")
file.copy(file.path(original_dataset_fruit_dir, fnames), file.path(train_fruit_dir))
file.copy(file.path(original_dataset_fruit_dir, fnames1), file.path(train_fruit_dir))
file.copy(file.path(original_dataset_fruit_dir, fnames2), file.path(train_fruit_dir))

#Copying images from original data set to motorbike train folder

fnames <- paste0("motorbike_000", 1:9, ".jpg")
fnames1 <- paste0("motorbike_00", 10:99, ".jpg")
fnames2 <- paste0("motorbike_0", 100:560, ".jpg")
file.copy(file.path(original_dataset_motorbike_dir, fnames), file.path(train_motorbike_dir))
file.copy(file.path(original_dataset_motorbike_dir, fnames1), file.path(train_motorbike_dir))
file.copy(file.path(original_dataset_motorbike_dir, fnames2), file.path(train_motorbike_dir))

#Copying images from original data set to person train folder

fnames <- paste0("person_000", 1:9, ".jpg")
fnames1 <- paste0("person_00", 10:99, ".jpg")
fnames2 <- paste0("person_0", 100:560, ".jpg")
file.copy(file.path(original_dataset_person_dir, fnames), file.path(train_person_dir))
file.copy(file.path(original_dataset_person_dir, fnames1), file.path(train_person_dir))
file.copy(file.path(original_dataset_person_dir, fnames2), file.path(train_person_dir))

#Copying images from original data set to airplane validation folder

fnames <- paste0("airplane_0", 561:630, ".jpg")
file.copy(file.path(original_dataset_airplane_dir, fnames), file.path(validation_airplane_dir))

#Copying images from original data set to car validation folder

fnames <- paste0("car_0", 561:630, ".jpg")
file.copy(file.path(original_dataset_car_dir, fnames), file.path(validation_car_dir))

#Copying images from original data set to cat validation folder

fnames <- paste0("cat_0", 561:630, ".jpg")
file.copy(file.path(original_dataset_cat_dir, fnames), file.path(validation_cat_dir))

#Copying images from original data set to dog validation folder

fnames <- paste0("dog_0", 561:630, ".jpg")
file.copy(file.path(original_dataset_dog_dir, fnames), file.path(validation_dog_dir))

#Copying images from original data set to flower validation folder

fnames <- paste0("flower_0", 561:630, ".jpg")
file.copy(file.path(original_dataset_flower_dir, fnames), file.path(validation_flower_dir))

#Copying images from original data set to fruit validation folder

fnames <- paste0("fruit_0", 561:630, ".jpg")
file.copy(file.path(original_dataset_fruit_dir, fnames), file.path(validation_fruit_dir))

#Copying images from original data set to motorbike validation folder

fnames <- paste0("motorbike_0", 561:630, ".jpg")
file.copy(file.path(original_dataset_motorbike_dir, fnames), file.path(validation_motorbike_dir))

#Copying images from original data set to person validation folder

fnames <- paste0("person_0", 561:630, ".jpg")
file.copy(file.path(original_dataset_person_dir, fnames), file.path(validation_person_dir))



#Copying images from original data set to airplane test folder

fnames <- paste0("airplane_0", 631:700, ".jpg")
file.copy(file.path(original_dataset_airplane_dir, fnames), file.path(test_airplane_dir))

#Copying images from original data set to car test folder

fnames <- paste0("car_0", 631:700, ".jpg")
file.copy(file.path(original_dataset_car_dir, fnames), file.path(test_car_dir))

#Copying images from original data set to cat test folder

fnames <- paste0("cat_0", 631:700, ".jpg")
file.copy(file.path(original_dataset_cat_dir, fnames), file.path(test_cat_dir))

#Copying images from original data set to dog test folder

fnames <- paste0("dog_0", 631:700, ".jpg")
file.copy(file.path(original_dataset_dog_dir, fnames), file.path(test_dog_dir))

#Copying images from original data set to flower test folder

fnames <- paste0("flower_0", 631:700, ".jpg")
file.copy(file.path(original_dataset_flower_dir, fnames), file.path(test_flower_dir))

#Copying images from original data set to fruit test folder

fnames <- paste0("fruit_0", 631:700, ".jpg")
file.copy(file.path(original_dataset_fruit_dir, fnames), file.path(test_fruit_dir))

#Copying images from original data set to motorbike test folder

fnames <- paste0("motorbike_0", 631:700, ".jpg")
file.copy(file.path(original_dataset_motorbike_dir, fnames), file.path(test_motorbike_dir))

#Copying images from original data set to person test folder

fnames <- paste0("person_0", 631:700, ".jpg")
file.copy(file.path(original_dataset_person_dir, fnames), file.path(test_person_dir))


#Generating image data generators for train, validation and test data

train_datagen <- image_data_generator(rescale = 1/255)
validation_datagen <- image_data_generator(rescale = 1/255)
test_datagen <- image_data_generator(rescale = 1/255)

train_generator <- flow_images_from_directory(
  train_dir, # Target directory
  train_datagen, # Training data generator
  target_size = c(150, 150), # Resizes all images to 150 × 150
  batch_size = 20, # 20 samples in one batch
  class_mode = "categorical" # Because we use categorical_crossentropy loss,
  # we need categorical labels.
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "categorical"
)

test_generator <- flow_images_from_directory(
  test_dir,
  test_datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "categorical"
)

#Building convnet for data set

model_v1 <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                input_shape = c(150, 150, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 8, activation = "softmax")

model_v1 %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(),
  metrics = c("acc")
)


#Summarizing the model

summary(model_v1)

history_v1 <- model_v1 %>%
  fit_generator(
    train_generator,
    steps_per_epoch = 100,
    epochs = 30,
    validation_data =
      validation_generator,
    validation_steps = 50
  )

#Plotting the accuracy and loss for training and validation data

plot(history_v1)

#Evaluating the model on test data

model_v1 %>%
  evaluate_generator(test_generator, steps = 50)

#Saving the model


saveRDS(history_v1,"./natural_images_fit_history.rds")

save_model_hdf5(model_v1, "./natural_images_model.h5")




