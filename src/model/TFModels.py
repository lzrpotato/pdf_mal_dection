from model_mobilenetV3 import MobileNetV3
from model_resnet101 import Resnet101
from model_squeezenet1_1 import SqueezeNet11
from model_vgg19 import VGG19


def get_tf_model(model, nclass):
    'TF_MBNET3','TF_RESNET101','TF_SZNET11','TF_VGG19',
    if model == 'MBNET3':
        model = MobileNetV3(nclass)
    elif model == 'TF_RESNET101':
        model = Resnet101(nclass)
    elif model == 'TF_SZNET11':
        model = SqueezeNet11(nclass)
    elif model == 'TF_VGG19':
        model = VGG19(nclass) 

    model.pretrained_weights()
    model.modify_model()
    return 