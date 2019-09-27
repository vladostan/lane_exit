from keras.layers import Conv2D, Activation, GlobalAveragePooling2D, Dense, Cropping2D
from keras.models import Model

from .blocks import DecoderBlock
from ..utils import get_layer_number, to_tuple


def build_linknet(backbone,
                  classes,
                  skip_connection_layers,
                  decoder_filters=(None, None, None, None, 16),
                  upsample_rates=(2, 2, 2, 2, 2),
                  n_upsample_blocks=5,
                  upsample_kernel_size=(3, 3),
                  upsample_layer='upsampling',
                  activation='sigmoid',
                  use_batchnorm=True):

    input = backbone.input
    x = backbone.output

    # convert layer names to indices
    skip_connection_idx = ([get_layer_number(backbone, l) if isinstance(l, str) else l
                            for l in skip_connection_layers])

    for i in range(n_upsample_blocks):

        # check if there is a skip connection
        skip_connection = None
        if i < len(skip_connection_idx):
            skip_connection = backbone.layers[skip_connection_idx[i]].output

        upsample_rate = to_tuple(upsample_rates[i])

        x = DecoderBlock(stage=i,
                         filters=decoder_filters[i],
                         kernel_size=upsample_kernel_size,
                         upsample_rate=upsample_rate,
                         use_batchnorm=use_batchnorm,
                         upsample_layer=upsample_layer,
                         skip=skip_connection)(x)

    x = Conv2D(classes, (3, 3), padding='same', name='final_conv')(x)
    x = Activation(activation, name=activation)(x)

    model = Model(input, x)

    return model

def build_linknet_notop(backbone,
                  classes,
                  skip_connection_layers,
                  decoder_filters=(None, None, None, None, 16),
                  upsample_rates=(2, 2, 2, 2, 2),
                  n_upsample_blocks=5,
                  upsample_kernel_size=(3, 3),
                  upsample_layer='upsampling',
                  activation='sigmoid',
                  output_name = "segmentation_output",
                  use_batchnorm=True):

    input = backbone.input
    x = backbone.output

    # convert layer names to indices
    skip_connection_idx = ([get_layer_number(backbone, l) if isinstance(l, str) else l
                            for l in skip_connection_layers])

    for i in range(n_upsample_blocks):

        # check if there is a skip connection
        skip_connection = None
        if i < len(skip_connection_idx):
            skip_connection = backbone.layers[skip_connection_idx[i]].output

        upsample_rate = to_tuple(upsample_rates[i])

        x = DecoderBlock(stage=i,
                         filters=decoder_filters[i],
                         kernel_size=upsample_kernel_size,
                         upsample_rate=upsample_rate,
                         use_batchnorm=use_batchnorm,
                         upsample_layer=upsample_layer,
                         skip=skip_connection)(x)

    x = Conv2D(classes, (3, 3), padding='same', name='final_conv')(x)
    x = Activation(activation, name=output_name)(x)

    return input, x

def build_linknet_bottleneck(backbone,
                  classes,
                  skip_connection_layers,
                  decoder_filters=(None, None, None, None, 16),
                  upsample_rates=(2, 2, 2, 2, 2),
                  n_upsample_blocks=5,
                  upsample_kernel_size=(3, 3),
                  upsample_layer='upsampling',
                  activation='sigmoid',
                  use_batchnorm=True):

    input = backbone.input
    
    x = backbone.output
    
    x2 = GlobalAveragePooling2D()(x)
    x2 = Dense(classes, activation='sigmoid', name="classification_output")(x2)

    # convert layer names to indices
    skip_connection_idx = ([get_layer_number(backbone, l) if isinstance(l, str) else l
                            for l in skip_connection_layers])

    for i in range(n_upsample_blocks):

        # check if there is a skip connection
        skip_connection = None
        if i < len(skip_connection_idx):
            skip_connection = backbone.layers[skip_connection_idx[i]].output

        upsample_rate = to_tuple(upsample_rates[i])

        x = DecoderBlock(stage=i,
                         filters=decoder_filters[i],
                         kernel_size=upsample_kernel_size,
                         upsample_rate=upsample_rate,
                         use_batchnorm=use_batchnorm,
                         upsample_layer=upsample_layer,
                         skip=skip_connection)(x)

    x = Conv2D(classes, (3, 3), padding='same', name='final_conv')(x)
    x = Activation(activation, name="segmentation_output")(x)
    
    model = Model(input, [x,x2])

    return model

def build_linknet_bottleneck_crop(backbone,
                  classes,
                  skip_connection_layers,
                  decoder_filters=(None, None, None, None, 16),
                  upsample_rates=(2, 2, 2, 2, 2),
                  n_upsample_blocks=5,
                  upsample_kernel_size=(3, 3),
                  upsample_layer='upsampling',
                  activation='sigmoid',
                  use_batchnorm=True):

    input = backbone.input
    
    x = backbone.output
        
    crop_w = int((x.shape[2]-x.shape[1])//2)
    
    assert 2*crop_w == int(x.shape[2]-x.shape[1])
    
    x2 = Cropping2D(cropping=(0, crop_w))(x)
    x2 = GlobalAveragePooling2D()(x2)
    x2 = Dense(classes, activation='sigmoid', name="classification_output")(x2)

    # convert layer names to indices
    skip_connection_idx = ([get_layer_number(backbone, l) if isinstance(l, str) else l
                            for l in skip_connection_layers])

    for i in range(n_upsample_blocks):

        # check if there is a skip connection
        skip_connection = None
        if i < len(skip_connection_idx):
            skip_connection = backbone.layers[skip_connection_idx[i]].output

        upsample_rate = to_tuple(upsample_rates[i])

        x = DecoderBlock(stage=i,
                         filters=decoder_filters[i],
                         kernel_size=upsample_kernel_size,
                         upsample_rate=upsample_rate,
                         use_batchnorm=use_batchnorm,
                         upsample_layer=upsample_layer,
                         skip=skip_connection)(x)

    x = Conv2D(classes, (3, 3), padding='same', name='final_conv')(x)
    x = Activation(activation, name="segmentation_output")(x)
    
    model = Model(input, [x,x2])

    return model