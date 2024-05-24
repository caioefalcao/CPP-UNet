
from time import time
import tensorflow as tf
import albumentations as A
import matplotlib.pyplot as plt
#####################################
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout,concatenate, Conv2D, Conv2DTranspose, UpSampling2D, MaxPooling2D, BatchNormalization, Activation,Input, add,multiply,Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TerminateOnNaN
import tensorflow.keras.backend as K
#####################################
import segmentation_models as sm
sm.set_framework('tf.keras')
#####################################
import numpy as np
from glob import glob
###################
import os
import matplotlib.pyplot as plt
import sys
import tensorflow as tf
from tensorflow import keras
from os import path
from tensorflow.keras import layers
from numpy import argmax

########### Dataset Imports ########
sys.path.insert(1, '/Codes/DatasetCodes')
from kitsDataset import createBackboneFolders2,get_validation_augmentation,get_training_augmentation,createFolder,defineTrainTestValidation, visualize, Dataset,setResultsPath, setDatabasePath,setImgSize, Dataloader, get_preprocessing, denormalize
sys.path.insert(1, '/Codes/Approaches/LossFunctions/')
from lossFunctions import FocalTverskyLoss,TverskyLoss,weighted_categorical_crossentropy,dsc,tversky_loss,dice_coef,focal_tversky_loss
from utils import PatientSliceGenerator
sys.path.insert(1, '/Codes/utils')
from utils import getApproach

from tensorflow.keras.models import save_model
#####################################


APPROACH_LIST = ["PPM_ASPP_UNET"]
IMAGE_SIZE = 256
BATCH_SIZE = 8

CLASSES = ['background','kidney','tumor','cyst']
N_CLASSES = len(CLASSES)
LR = 0.0001
EPOCHS = 100
ACTIVATION ='softmax'

print("######### Hyperparameters #########")
print("MODEL_CLASSES:", CLASSES)
print("BATCH_SIZE:", BATCH_SIZE)
print("N_CLASSES:", N_CLASSES)
print("EPOCHS:", EPOCHS)
print("APPROACH_LIST:",APPROACH_LIST)


pathDataset_kits21 = "/Dataset/kits21_Dataset/kits21/kits21/data/"
pathDataset_kits23 = "/Dataset/kits23_Dataset/kits23/kits23/data/"
listDataset = [pathDataset_kits21,pathDataset_kits23]

resultsPath = "Codes/Output/"
for dataset in listDataset:
    setDatabasePath(dataset)

    resultsPath = os.path.join(resultsPath, dataset.split("/")[5] +"_PPM_ASPP_UNET"+ "/") 
    createFolder(resultsPath)
    setResultsPath(resultsPath)
    setImgSize(IMAGE_SIZE)

    for APPROACH_NAME in APPROACH_LIST:

        #Define Train Test and Validation from kits19 dataset
        x_train_dir,y_train_dir,x_test_dir,y_test_dir,x_valid_dir,y_valid_dir,test_case = defineTrainTestValidation(70,20,10)

        
        ################################################################################
        # Create Datasets - Train, Test and Validation
        ################################################################################
        preprocess_input = sm.get_preprocessing("resnet50")
        
        train_dataset = Dataset(
            x_train_dir, 
            y_train_dir,
            classes=CLASSES, 
            #augmentation=get_training_augmentation(),
            preprocessing=get_preprocessing(preprocess_input)
        )

        test_dataset = Dataset(
            x_test_dir, 
            y_test_dir,
            classes=CLASSES, 
            #augmentation=get_validation_augmentation(),
            preprocessing=get_preprocessing(preprocess_input)
        )

        # Dataset for validation images
        valid_dataset = Dataset(
            x_valid_dir, 
            y_valid_dir,
            classes=CLASSES, 
            #augmentation=get_validation_augmentation(),
            preprocessing=get_preprocessing(preprocess_input)
        )
        train_dataloader = Dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        valid_dataloader = Dataloader(valid_dataset, batch_size=1, shuffle=False)
        test_dataloader = Dataloader(test_dataset, batch_size=1, shuffle=False)  

        # check shapes for errors

        print(train_dataloader[0][0].shape)
        print(train_dataloader[0][1].shape)
    
        print((BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, N_CLASSES))
        assert train_dataloader[0][0].shape == (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3)
        assert train_dataloader[0][1].shape == (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, N_CLASSES)
        ################################################################################
        # Create the Model
        ################################################################################

        
        #Create struture directory 
        backboneGeneralPath = createBackboneFolders2(APPROACH_NAME)
        

        print("Number of Classes: ",N_CLASSES)
        print("Activation: ",ACTIVATION)
        print("Approach: ",APPROACH_NAME)
        
        
        
        model = getApproach(APPROACH_NAME,IMAGE_SIZE,N_CLASSES,ACTIVATION)
        

        
        ################################################################################
        # Define Loss Function, Metrics and Callbacks
        ################################################################################
        optim = keras.optimizers.Adam(LR)
        dice_loss = sm.losses.DiceLoss()
    
        focal_loss = sm.losses.BinaryFocalLoss() if N_CLASSES == 1 else sm.losses.CategoricalFocalLoss()
        total_loss = dice_loss + (1 * focal_loss)
        
        #########################################################################
        #Metrics
        iou_default = sm.metrics.IOUScore(  threshold=0.5)
        iou_bg = sm.metrics.IOUScore(       threshold=0.5,   class_indexes=0, name="iou_BG")
        iou_kidney = sm.metrics.IOUScore(   threshold=0.5,   class_indexes=1,name="iou_Kidney")
        iou_tumor = sm.metrics.IOUScore(    threshold=0.5,   class_indexes=2, name="iou_Tumor")
        iou_cyst = sm.metrics.IOUScore(     threshold=0.5,   class_indexes=3, name="iou_Cyst")

        dice_score_default = sm.metrics.FScore( threshold=0.5)
        dice_score_bg = sm.metrics.FScore(      threshold=0.5,  class_indexes=0, name="dice_score_BG")
        dice_score_kidney = sm.metrics.FScore(  threshold=0.5,  class_indexes=1, name="dice_score_Kidney")
        dice_score_tumor = sm.metrics.FScore(   threshold=0.5,  class_indexes=2, name="dice_score_Tumor")
        dice_score_cyst = sm.metrics.FScore(    threshold=0.5,  class_indexes=3, name="dice_score_Cyst")
    
    
        metrics = [iou_default,iou_bg, iou_kidney, iou_tumor,iou_cyst,
                    dice_score_default, dice_score_bg, dice_score_kidney,dice_score_tumor,dice_score_cyst]

        model.compile(optimizer=optim, loss=total_loss, metrics=metrics)

        pathOutputModel = backboneGeneralPath  +"/Model/" 
        filename = 'best_model_'+ APPROACH_NAME +'.h5'
        
        callbacks = [
            keras.callbacks.ModelCheckpoint(filepath = pathOutputModel+filename, monitor="val_loss", save_weights_only=True, save_best_only=True, mode='min'),
            keras.callbacks.EarlyStopping(patience=10),
            keras.callbacks.ReduceLROnPlateau(),
        ]

        history = model.fit(
            train_dataloader, 
            steps_per_epoch=len(train_dataloader), 
            epochs=EPOCHS, 
            callbacks=callbacks, 
            validation_data=valid_dataloader, 
            validation_steps=len(valid_dataloader),
            shuffle=True
        )
        
        # ################################################################################
        # # Evaluation Train
        # ################################################################################
        pathOutputEvaluationTrain = backboneGeneralPath +"/EvaluationTrain/"+ APPROACH_NAME + "_" +"EvaluateTrain" 
        # Plot training & validation iou_score values
        
        plt.figure(figsize=(30, 5))
        plt.subplot(121)
        plt.plot(history.history['iou_score'])
        plt.plot(history.history['val_iou_score'])
        plt.title('Model iou_score')
        plt.ylabel('iou_score')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        # Plot training & validation loss values
        plt.subplot(122)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig(pathOutputEvaluationTrain+'.png')
        plt.show()
    
        ################################################################################
        # Save Model
        ################################################################################
        model.save(pathOutputModel+"keras_model.h5")
        
        ################################################################################
        # Model Evaluation with Test Dataset
        ################################################################################
        
        pathOutputEvaluationTest = backboneGeneralPath +"/EvaluationTest/"+ APPROACH_NAME  
        pathOutputScoreTest = backboneGeneralPath  +"/Score/"+ APPROACH_NAME  
        fileScore = open(pathOutputScoreTest + ".txt", "w")
        fileScore.write("#####################\n")
        fileScore.write("Result Segmentation ")
        fileScore.write("#####################\n")


        """# Evaluation on Test Data"""
        pathOutputEvaluationTest = backboneGeneralPath +"/EvaluationTest/"+ APPROACH_NAME  
        pathOutputScoreTest = backboneGeneralPath  +"/Score/"+ APPROACH_NAME  
        pathOutputScoreTest = pathOutputScoreTest+"_Result2"
    
        scores = model.evaluate_generator(test_dataloader)

        fileScore = open(pathOutputScoreTest + ".txt", "w")
        fileScore.write("#####################\n")
        fileScore.write("Result Segmentation ")
        fileScore.write("#####################\n")
        

        print("Loss: {:.5}".format(scores[0]))
        fileScore.write("Loss: {:.5}".format(scores[0])+"\n")
        for metric, value in zip(metrics, scores[1:]):
            print("mean {}: {:.5}".format(metric.__name__, value))
            fileScore.write("mean {}: {:.5}".format(metric.__name__, value))
            fileScore.write("\n")
    

        ################################################################################
        # Visualization of results on test dataset
        ################################################################################

        n = 500
        ids = np.random.choice(np.arange(len(test_dataset)), size=n)

        for i in ids:
            
            image, gt_mask = test_dataset[i]
            image = np.expand_dims(image, axis=0)
            pr_mask = model.predict(image)
            
            visualize(
                image=denormalize(image.squeeze()),
                gt_Kidney   =gt_mask[..., 1].squeeze(),
                pr_Kidney   =pr_mask[..., 1].squeeze(),
                gt_Tumor    =gt_mask[..., 2].squeeze(),
                pr_Tumor    =pr_mask[..., 2].squeeze(),
                gt_Cyst     =gt_mask[..., 3].squeeze(),
                pr_Cyst     =pr_mask[..., 3].squeeze(),
                
                path=pathOutputEvaluationTest,
                count = i
            )

    