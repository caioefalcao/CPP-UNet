import sys
sys.path.insert(1, '/home/viplab/Documents/GitHub/caio_falcao_doc/Codes/DatasetCodes')
from kitsDataset import getDatabasePath


sys.path.insert(1, '/home/viplab/Documents/GitHub/caio_falcao_doc/Codes/Approaches/DeeplabV3Plus_Default/')
from DeeplabV3Plus import DeeplabV3Plus
sys.path.insert(1, '/home/viplab/Documents/GitHub/caio_falcao_doc/Codes/Approaches/UNet_Default/')
from Unet_Model import U_NetBase
sys.path.insert(1, '/home/viplab/Documents/GitHub/caio_falcao_doc/Codes/Approaches/DeepLabPPM/')
from DeepLabPPM import DeeplabV3PlusPPM
from DeepLabPPM_SegundoArq import DeeplabV3PlusPPM_Arq
sys.path.insert(1, '/home/viplab/Documents/GitHub/caio_falcao_doc/Codes/Approaches/UNet_PPM/')
from UNet_PPM_Model import PPM_Model

sys.path.insert(1, '/home/viplab/Documents/GitHub/caio_falcao_doc/Codes/Approaches/HRNET_Experiments/')
from HRNET_32 import HRNet_Default
from HRNET_Standard_Version import hrnet18_v2
from HRNET_32_ASPP import HRNet_ASPP
from HRNET_32_DeeplabDecoder import HRNet_DeepLab_Decoder
from HRNET_32_PPM import HRNetWithPPM
from HRNET_32_PPM_4_Levels import HRNetWithPPM_4_Levels
from HRNET_32_PPM_Begins import HRNet_StartWithPPM

sys.path.insert(1, '/home/viplab/Documents/GitHub/caio_falcao_doc/Codes/Approaches/PPM_Default/')
from PPM_Default import PPM_Last_Model


####### PPM Variantes Experimentos 31012024######
sys.path.insert(1, '/home/viplab/Documents/GitHub/caio_falcao_doc/Codes/Approaches/PPM_Variants/')
from PPM_ASPP_UNet import PPM_With_ASPP_Unet

###### Conformer Functions
sys.path.insert(1, '/home/viplab/Documents/GitHub/caio_falcao_doc/Codes/Approaches/Conformer_Variants/')
from Comformer_UNet import COMFORMER_CONV_UNET


###### Conformer Functions Gabriel e Marcus
sys.path.insert(1, '/home/viplab/Documents/GitHub/caio_falcao_doc/Codes/Approaches/Conformer_Variants/Marcus_Gabriel/')
from Conformer_Conv_UNet_CBAM import CONFORMER_CONV_UNET_CBAM 

sys.path.insert(1, '/home/viplab/Documents/GitHub/caio_falcao_doc/Codes/Approaches/Conformer_Variants/Marcus_Gabriel/')
from Conformer_Conv_UNet_PPM_v1 import CONFORMER_CONV_UNET_PPM 

def getApproach(approachName, IMAGE_SIZE,NUM_CLASSES,ACTIVATION):
    model = ''
    if(approachName == "DeepLabv3_plus_Default"):
        model = DeeplabV3Plus(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES, activation=ACTIVATION)
    elif(approachName == "DeepLabv3_plus_PPM"):
        model = DeeplabV3PlusPPM(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES, activation=ACTIVATION)
    elif(approachName == "DeepLabv3_plus_PPM_Arq"):
        model = DeeplabV3PlusPPM_Arq(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES, activation=ACTIVATION)   
    elif(approachName == "UNet_Default"):
        model = U_NetBase(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES, activation=ACTIVATION)
    elif(approachName == "UNet_PPM"):
        model = PPM_Model(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES, activation=ACTIVATION)
    elif(approachName == "HRNET_Default"):
        model = HRNet_Default(image_size=IMAGE_SIZE, n_class=NUM_CLASSES, activation=ACTIVATION)  
    elif(approachName == "HRNET_32_ASPP"):
        model = HRNet_ASPP(image_size=IMAGE_SIZE, n_class=NUM_CLASSES, activation=ACTIVATION) 
    elif(approachName == "HRNET_32_DEEPLAB_DECODER"):
        model = HRNet_DeepLab_Decoder(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES, activation=ACTIVATION) 
    elif(approachName == "HRNetWithPPM"):
        model = HRNetWithPPM(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES, activation=ACTIVATION) 
    elif(approachName == "HRNET_32_PPM_4_Levels"):    
        model = HRNetWithPPM_4_Levels(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES, activation=ACTIVATION) 
    elif(approachName == "PPM_Last_Model"):    
        model = PPM_Last_Model(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES, activation=ACTIVATION) 
    elif(approachName == "HRNet_StartWithPPM"):    
        model = HRNet_StartWithPPM(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES, activation=ACTIVATION) 
    elif(approachName == "PPM_ASPP_UNET"):    
        model = PPM_With_ASPP_Unet(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES, activation=ACTIVATION) 
    elif(approachName == "COMFORMER_CONV_UNET"):    
        model = COMFORMER_CONV_UNET(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES, activation=ACTIVATION) 
    elif(approachName == "COMFORMER_CONV_UNET_CBAM"):    
        model = CONFORMER_CONV_UNET_CBAM(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES, activation=ACTIVATION) 
    elif(approachName == "CONFORMER_CONV_UNET_PPM_MARCUS"):    
        model = CONFORMER_CONV_UNET_PPM(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES, activation=ACTIVATION) 
   
    

    return model



import tensorflow as tf
import numpy as np
import cv2
from glob import glob
import os
class PatientSliceGenerator:
    """
        Generate ordered slices for a single patient.
    """
    def __init__(self, patient_path,image_size):
        self.patient = patient_path
        # self.pathDataset = "/home/viplab/Dataset/kits21_Dataset/kits21/kits21/data/"
        self.pathDataset = getDatabasePath()
        self.HEIGHT_IMG = self.WIDTH_IMG = image_size
        #self.class_values = [1,2,3]
        self.class_values = [0,1,2,3]
        #self.class_values = [1,2]
        
    
    def __call__(self) -> (tf.Tensor, tf.Tensor, tf.Tensor):
        pathPacient = os.path.join(self.pathDataset, self.patient+"/")
        pathImages = os.path.join(str(pathPacient),self.patient +"_Exam")
        pathAnnotations = os.path.join(str(pathPacient),self.patient +"_Segmentation")
        
        #print(pathPacient,pathImages,pathAnnotations)
       
        img =  glob(str(pathImages)+'/*.png') 
        annotations = glob(str(pathAnnotations)+'/*.png') 
        
        # annotations.sort(key = lambda p: int(Path(p).name.replace("GT_", "").replace(".png", "")))
        # img = [f.replace("GT_", "").replace(".png", ".pfm") for f in annotations]
        
        for i in range(len(annotations)):
            x = cv2.imread(img[i],  cv2.IMREAD_UNCHANGED)
            x = cv2.resize(x, (self.HEIGHT_IMG, self.WIDTH_IMG)) 
            y = cv2.imread(annotations[i], 0)
            y = cv2.resize(y, (self.HEIGHT_IMG, self.WIDTH_IMG))
        
            masks = [(y == v) for v in self.class_values]
            
            y = np.stack(masks, axis=-1).astype('float')
            # #add background if mask is not binary
            # if y.shape[-1] != 1:
            #     background = 1 - y.sum(axis=-1, keepdims=True)
            #     y = np.concatenate((y, background), axis=-1)
            
            yield x.astype(np.float32), y.astype(np.float32)



