#BackBone Lists
def WithoutBackbone():
    return ['WithoutBackbone']

def EfficientNet():
    return ['efficientnetb0', 'efficientnetb1', 'efficientnetb2',"efficientnetb3", 'efficientnetb4', 'efficientnetb5', 'efficientnetb6', 'efficientnetb7']

def MobileNet():
    return ['mobilenet', 'mobilenetv2']

def Inception():
    return ['inceptionv3', 'inceptionresnetv2']
    #return ['inceptionresnetv2']

def DenseNet():
    return ['densenet121', 'densenet169', 'densenet201']
    #return ['densenet169', 'densenet201']

def SENet154():
    return ['senet154']

def SE_ResNeXt():
    return ['seresnext50', 'seresnext101']

def ResNeXt():
    return ['resnext50', 'resnext101']

def SE_ResNet():
    return ['seresnet18', 'seresnet34', 'seresnet50' ,'seresnet101', 'seresnet152']

def ResNet():
    return ['resnet18' ,'resnet34', 'resnet50', 'resnet101', 'resnet152']

def VGG():
    return ['vgg16', 'vgg19']

def getBackbonesList(argument):
    
    switcher = {
        "WithoutBackbone": WithoutBackbone(),
        "EfficientNet": EfficientNet(),
        "MobileNet": MobileNet(),
        "Inception": Inception(),
        "DenseNet": DenseNet(),
        "SENet154": SENet154(),
        "SE_ResNeXt": SE_ResNeXt(),
        "ResNeXt": ResNeXt(),
        "SE_ResNet": SE_ResNet(),
        "ResNet": ResNet(),
        "VGG": VGG()
    }
    # Get the function from switcher dictionary
    func = switcher.get(argument, lambda: "Invalid month")
    # Execute the function
    return func