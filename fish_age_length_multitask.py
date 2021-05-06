"""
Apply inceptionV3 to predict fish age from otolith images using multi-task learning 
with fish length 
@author: Dimitris Politikos, May 2021
e-mail: dimpolitik@gmail.com
"""

# Load libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import regularizers
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import model_evaluation_utils as meu
import numpy as np
from tensorflow.keras.layers import GlobalAveragePooling2D
from sklearn.metrics import mean_squared_error,confusion_matrix
import matplotlib.pyplot as plt
import pickle

# --- CONFIG PARAMETERS --- #
IMG_WIDTH, IMG_HEIGHT = 400, 400

BATCH_SIZE = 16

EPOCHS = 150

LEARNING_RATE = 0.0004 

# --- Import biological data --- #
fish_biol = pd.read_csv('Biological_data.csv')
fish_biol.columns = ['Image','Age','Length']

# Match image names with biological data
fish_images = 'Images/'
fish_biol['image_path'] = fish_biol(lambda row: (fish_images + row["Image"]), 
                                             axis=1)
# Age labels
age_labels = fish_biol['Age']

# Length labels
length_labels = fish_biol['Length'].astype(float)

# Normalize fish length
len_max = length_labels.max()
length_labels = length_labels/len_max

# --- Filenames to save trained model and history--- #
csv_file = 'loss_mullus_multiple.csv'
model_weights = 'model_weights.h5'
model_history = 'model_history.p'


# --- Functions --- #
def create_train_val_test(data, labels, df_type):
     '''
     Split dataset 
     '''
     x_train = data["x_train"]
     x_val = data["x_val"]
     x_test = data["x_test"]
     y_train = data["y_train"]
     y_val = data["y_val"]
     y_test = data["y_test"]
     
     # load images as multidimensional array
     train_data = np.array([img_to_array(load_img(img, target_size=(IMG_WIDTH, IMG_HEIGHT)))
                           for img in data['image_path'].values.tolist()
                          ]).astype('float32')

     # create train and test datasets
     x_train, x_test, y_train, y_test = train_test_split(train_data, labels,
                                                    test_size=0.3,
                                                    stratify=np.array(labels),
                                                    random_state=42)

     # create train and validation datasets
     x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                  test_size=0.15,
                                                  stratify=np.array(y_train),
                                                  random_state=42)    
     

     # if variable of interest is age apply one-hot encoding
     if (df_type == 0):
         y_train_ohe = pd.get_dummies(y_train.reset_index(drop=True)).astype(float) 
         y_val_ohe = pd.get_dummies(y_val.reset_index(drop=True)).astype(float)
         y_test_ohe = pd.get_dummies(y_test.reset_index(drop=True)).astype(float) 
     else:
         y_train_ohe = y_train
         y_val_ohe = y_val
         y_test_ohe = y_test

     print('Train, Validation and Test Datasets Size:', x_train.shape, x_val.shape, x_test.shape)
     
     # Create train generator
     train_image_gen = ImageDataGenerator(rescale=1./255,  
                                          rotation_range=360)
                                       
     train_generator = train_image_gen.flow(x_train, y_train_ohe,  
                                            batch_size=BATCH_SIZE, seed=1)
                                     
     # Create validation generator
     val_image_gen = ImageDataGenerator(rescale=1./255)
     val_generator = val_image_gen.flow(x_val, y_val_ohe,  
                                        batch_size=BATCH_SIZE, seed=1) 

     return x_train, x_val, x_test, y_train_ohe, y_val_ohe, y_test_ohe, y_test, train_generator, val_generator                                                                     

def join_generators(generator1, generator2, X, Y1, Y2):
    while True:
            Yi1 = generator1.next() 
            Yi2 = generator2.next()
            yield Yi1[0], [Yi1[1], Yi2[1]]


def build_train_model(x_train, x_test, train_generator1, val_generator1, y_train_ohe1, y_test_ohe1,   
                                       train_generator2, val_generator2, y_train_ohe2, y_test_ohe2,
                                       model_save, hist_save):
     '''
     Create inceptionV3 model, and make model.fit
     '''
     # Pre-trained model
     pre_trained_model = InceptionV3(input_shape = (IMG_WIDTH, IMG_HEIGHT, 3), 
                                      include_top = False, # Leave out the last fully connected layer
                                      weights = 'imagenet')

     # Flatten the output layer to 1 dimension
     x = pre_trained_model.output
     x = GlobalAveragePooling2D()(x)
     
     x = layers.Flatten()(x)
     x = layers.Dropout(0.2)(x)
     x = layers.Dense(1024, activation='relu',kernel_regularizer=regularizers.l2(0.025))(x)
     x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.025))(x)
     x = layers.Dropout(0.25)(x)

     # Add two final layes for classification (age) and regression (length) 
     total_ages = y_train_ohe1.shape[1]
     total_length = 1 
     age_predictions = layers.Dense(total_ages, activation='softmax', name = 'age')(x)           
     length_predictions = layers.Dense(total_length, activation='linear', name = 'length')(x)

     # Final model
     model = Model(pre_trained_model.input, outputs = [age_predictions, length_predictions])

     # Compile model
     model.compile(optimizer = Adam(lr = LEARNING_RATE),
             loss = {'age': 'categorical_crossentropy',
                     'length': 'mean_squared_error'}, 
             metrics = {'age': 'accuracy',
                        'length': 'mean_squared_error'},
             loss_weights = [1, 475]
                   )
          
     # Train the model

     # Callbacks
     callback_check = ModelCheckpoint(filepath=model_save, monitor='val_age_accuracy', verbose =2, save_best_only=True)
     callback_early = EarlyStopping(monitor='val_age_accuracy', patience=25, restore_best_weights= True)

     # Join generators
     join_train = join_generators(train_generator1, train_generator2, x_train, y_train_ohe1, y_train_ohe2)
     join_val = join_generators(val_generator1, val_generator2, x_test, y_test_ohe1, y_test_ohe2)
     
     train_steps_per_epoch = x_train.shape[0] // BATCH_SIZE
     val_steps_per_epoch = x_test.shape[0] // BATCH_SIZE

     # Train the model
     history = model.fit_generator(
                                join_train,
                                validation_data = join_val,
                                steps_per_epoch = train_steps_per_epoch,
                                epochs = EPOCHS,
                                validation_steps = val_steps_per_epoch,
                                verbose = 2,
                                callbacks=[callback_check,  callback_early])

     # Save weights
     model.save(model_save)

     # Save history
     with open(hist_save, 'wb') as file_pi:
          pickle.dump(history.history, file_pi)
        
     return history.history

def plot_loss_accuracy(trained_model, csvfile):
    '''
    Compute accuracy/mse and loss for training and test of age & length
    '''
    # Accuracy & mse
    age_acc = trained_model['age_accuracy']
    len_acc = trained_model['length_mean_squared_error']
    val_age_acc = trained_model['val_age_accuracy']
    val_len_acc = trained_model['val_length_mean_squared_error']

    # Training Losses
    age_loss =trained_model['age_loss']
    len_loss = trained_model['length_loss']

    # Validation losses
    val_age_loss = trained_model['val_age_loss']
    val_len_loss = trained_model['val_length_loss']

    # Number of epochs 
    epochs = range(1, len(age_acc))

    # Plot losses
    fig = plt.figure(figsize=(18,6))
    plt.subplot(1,2,1)
    plt.plot(epochs, age_loss[:120], 'b-', label='Training loss')
    plt.plot(epochs, val_age_loss[:120], 'r-', label='Validation loss')
    plt.ylim([0, 4])
    plt.xlabel('Epochs', fontsize = 16)
    plt.ylabel('Loss', fontsize = 16)
    plt.xlim([-5,125])
    plt.xticks(np.arange(0, 140, 20), fontsize = 14)
    plt.yticks(np.arange(0, 4.5, 0.5), fontsize = 14)
    plt.title('Fish age-length multitask network -age  loss', fontsize = 20)
    plt.legend()
    plt.grid()

    plt.subplot(1,2,2)
    plt.plot(epochs, len_loss[:120], 'b-', label='Training loss')
    plt.plot(epochs, val_len_loss[:120], 'r-', label='Validation loss')
    plt.legend()
    plt.grid()
    plt.title('Model length loss', fontsize = 16)
    plt.xlabel('Epochs', fontsize = 14)
    plt.ylabel('Mean squared error', fontsize = 14)
    plt.xticks(np.arange(0, 140.0, 20.0), fontsize = 12)
    plt.title('Fish age-length multitask network - length loss', fontsize = 20)
    plt.legend()
    plt.grid()
    
    # plot ratio age_loss/length_loss 
    age_length_loss_ratio = [i / j for i, j in zip(age_loss[:120],len_loss[:120])] 

    fig = plt.figure(figsize=(10,6))
    plt.plot(epochs, age_length_loss_ratio, 'k-')
    plt.xlabel('Epochs', fontsize = 14)
    plt.ylabel('Age loss / Length loss', fontsize = 14)
    plt.xticks(np.arange(0, 140.0, 20.0), fontsize = 12)
    plt.legend()
    plt.grid()
    
    # Age loss and accuracy
    fig = plt.figure(figsize=(18,6))
    plt.subplot(1,2,1)
    plt.plot(epochs, age_loss[:120], 'b-', label='Training loss')
    plt.plot(epochs, val_age_loss[:120], 'r-', label='Validation loss')
    plt.ylim([0, 4])
    plt.xlabel('Epochs', fontsize = 16)
    plt.ylabel('Loss', fontsize = 16)
    plt.xlim([-5,125])
    plt.xticks(np.arange(0, 140, 20), fontsize = 14)
    plt.yticks(np.arange(0, 4.5, 0.5), fontsize = 14)
    plt.title('Fish age-length multitask network - loss', fontsize = 20)
    plt.legend()
    plt.grid()
    plt.subplot(1,2,2)
    plt.plot(epochs, age_acc[:120], 'b', label='Training accuracy')
    plt.plot(epochs, val_age_acc[:120], 'r-', label='Validation accuracy')
    plt.title('Fish age-length multitask network - accuracy', fontsize = 20)
    plt.xlabel('Epochs', fontsize = 16)
    plt.ylabel('Accuracy', fontsize = 16)
    plt.xlim([-5,125])
    plt.xticks(np.arange(0, 140, 20), fontsize = 14)
    plt.yticks(np.arange(0.2, 1.1, 0.1), fontsize = 14)
    plt.legend()
    plt.grid()
        
    # Length loss and accuracy
    fig = plt.figure(figsize=(18,6))
    plt.subplot(1,2,1)
    plt.plot(epochs, len_loss[:120], 'b-', label='Training loss')
    plt.plot(epochs, val_len_loss[:120], 'r-', label='Validation loss')
    plt.legend()
    plt.grid()
    plt.title('Model length loss', fontsize = 16)
    plt.xlabel('Epochs', fontsize = 14)
    plt.ylabel('Mean squared error', fontsize = 14)
    plt.xticks(np.arange(0, 140.0, 20.0), fontsize = 12)
    plt.subplot(1,2,2)
    plt.plot(epochs, len_acc[:120], 'b-', label='Training accuracy')
    plt.plot(epochs, val_len_acc[:120], 'r-', label='Validation accuracy')
    plt.legend()
    plt.grid()
    plt.title('Model length accuracy', fontsize = 16)
    plt.xlabel('Epochs', fontsize = 14)
    plt.ylabel('Mean squarer error', fontsize = 14)
    plt.xticks(np.arange(0, 140.0, 20.0), fontsize = 12)
   
def evaluate_model(saved_model, labels1, labels2, x_test, y_test1, y_test_ohe1, y_test2):
    '''
    Evaluate model on the testing set
    '''

    # Load the model after training
    model = load_model(saved_model)
    
    # scaling test features
    x_test /= 255.

    # getting model predictions
    labels_ohe1 = pd.get_dummies(labels1, sparse=True)
    #labels_ohe_names2 = pd.get_dummies(df_targets2, sparse=True)
    pred1, pred2 = model.predict(x_test)
    predictions = pd.DataFrame(pred1, columns=labels_ohe1.columns)
    predictions = list(predictions.idxmax(axis=1))
    test_labels = list(y_test1)

    # evaluate model performance
    print('------ Age performance -------')

    meu.display_classification_report(true_labels=test_labels, 
                                      predicted_labels=predictions, 
                                      classes=list(labels_ohe1.columns))
 
    meu.get_metrics(true_labels=test_labels,
                    predicted_labels=predictions)

    y_age_lst = y_test_ohe1.values.tolist()
    y_age_actual = [r.index(1) for r in y_age_lst]

    y_pred_ohe = pd.get_dummies(predictions)
    y_pred_lst = y_pred_ohe.values.tolist()
    y_age_predicted = [r.index(1) for r in y_pred_lst]

    print(y_age_actual[2:10])
    print(y_age_predicted[2:10])

    # compute mse for age
    rmse = mean_squared_error(y_age_predicted, y_age_actual, squared = False)
    print('Mean Square error (age):', rmse)

    cm = confusion_matrix(y_age_actual, y_age_predicted)
    print(cm)

    # compute mse for length
    mse = mean_squared_error(y_test2, pred2, squared = False)
    print('Mean Square error (length):', mse)

    # plot y=x for ages
    xx = np.arange(0,1.1,0.1)

    # Plot predicted vs actual length
    plt.figure()
    plt.plot(y_test2, pred2, 'b.')
    plt.plot(xx, xx, 'k-', linewidth = 2)
    plt.xlabel('Actual Length')
    plt.ylabel('Predicted Length')
    plt.title('Actual vs Predicted Lengths')
    plt.grid()
    plt.savefig('regression_length.png', dpi = 300)

    plt.figure()
    hist_error = np.asarray(y_age_predicted)-np.asarray(y_age_actual)
    stars = pd.Series(hist_error)
    vc = stars.value_counts().sort_index()
    vc.plot(kind='bar')
    plt.xticks(rotation=0)
    plt.xlabel('Predicted age - True age (years)', fontsize = 12)
    plt.ylabel('Number of occurences', fontsize = 12)
    plt.title('Age difference between model and ground truth', fontsize = 14)
    plt.grid()
    
if __name__ == '__main__':
    # Split dataset for age
    x_train, x_val, x_test, y_train_ohe_age, y_val_ohe_age, y_test_ohe_age, y_test_age, train_generator_age, val_generator_age = create_train_val_test(fish_biol, age_labels, 0)
    
    # Split dataset for length                                             
    _, _, _, y_train_ohe_length, y_val_ohe_length, y_test_ohe_length, y_test_length, train_generator_length, val_generator_length = create_train_val_test(fish_biol, length_labels, 1)
    
    # Train model
    trained_model = build_train_model(x_train, x_val, train_generator_age, val_generator_age, y_train_ohe_age, y_val_ohe_age,
                                                  train_generator_length, val_generator_length, y_train_ohe_length, y_val_ohe_length,
                                                  model_weights, model_history)
    # Plot loss and accuracy
    plot_loss_accuracy(trained_model)
    
    # Evaluate model
    evaluate_model(model_weights, age_labels, length_labels, x_test, y_test_age, y_test_ohe_age, y_test_length) 
