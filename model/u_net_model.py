import torch 
import torch.nn as nn

import torch.nn.functional as F

from dataset_utils.utils import calculate_soft_iou_loss, calculate_focal_loss, calculate_dice_loss

import torchvision.transforms as transform

class SimpleConvBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int=3, padding=1, add_dropout:bool=False, add_maxpool:bool=False):
        super(SimpleConvBlock, self).__init__()

        if add_dropout:
            self.convblock = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=True),
                        nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=True),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(), # in_place saves memory.
                        nn.Dropout()
                    )

        else:
            self.convblock = nn.Sequential(
                # nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=True),
                nn.BatchNorm2d(out_channels),
                nn.ReLU() # in_place memory
            )

        if add_maxpool:
            self.convblock.append(
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

    def forward(self, x):
        return self.convblock(x)
    
class SimpleDeConvBlock(nn.Module):

    def __init__(self, in_channels:int, out_channels:int) -> None:
        super(SimpleDeConvBlock, self).__init__()

        #out = (x - 1)s - 2p + d(k - 1) + op + 1
        self.transpose_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.convblock = SimpleConvBlock(in_channels, out_channels)
        # self.convblock = nn.Conv2d(in_channels, out_channels)

    def forward(self, input_x:torch.tensor, concat_x:torch.tensor):

        transpose_output = self.transpose_layer(input_x)
        transpose_output = nn.ReLU()(transpose_output)

        concatenated_output = torch.concat([transpose_output, concat_x], dim=1)
        deconv_output = self.convblock(concatenated_output)

        return deconv_output

class SimpleDeConvBlock_Revised(nn.Module):

    def __init__(self, in_channels:int, out_channels:int) -> None:
        super(SimpleDeConvBlock_Revised, self).__init__()

        #out = (x - 1)s - 2p + d(k - 1) + op + 1
        self.transpose_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.convblock = SimpleConvBlock(in_channels, out_channels)
        # self.convblock = nn.Conv2d(in_channels, out_channels, )

    def forward(self, input_x:torch.tensor, concat_x:torch.tensor):

        transpose_output = self.transpose_layer(input_x)
        croped = self.crop(concat_x, transpose_output)
        concatenated_output = torch.concat([transpose_output, croped], dim=1)
        deconv_output = self.convblock(concatenated_output)

        return deconv_output

    def crop(self,input_tensor,target_tensor):
        # For making the size of the encoder conv layer and the decoder Conv layer same
        _,_,H,W = target_tensor.shape
        return transform.CenterCrop([H,W])(input_tensor)

class AttentionGate(nn.Module):

    def __init__(self, g:int, x:int):
        super(AttentionGate, self).__init__()

        '''
        g - lower layer feature map (has more channels). (last layer of encoder/previous decoder layer)
        x - upper layer feature map (has lesser channels). (skip connection from parallel encoder layer)
        '''

        self.g_conv_layer = nn.Conv2d(g, x, kernel_size=1) #(n_c, h, w) -> (m_c, h', w')
        self.x_conv_layer = nn.Conv2d(x, x, kernel_size=3, stride=2, padding=1) #(n_c, h', w') -> (n_c, h, w)

        self.attention_layer = nn.Conv2d(x, 1, kernel_size=1)

    def forward(self, g:torch.tensor, x:torch.tensor):
        
        g_feature = self.g_conv_layer(g)
        x_feature = self.x_conv_layer(x)

        feature_sum = g_feature + x_feature

        feature_sum = F.relu(feature_sum, inplace=True)

        attention = self.attention_layer(feature_sum)
        attention = F.sigmoid(attention)

        upsampled_attention = F.interpolate(attention, scale_factor=2, mode='bilinear', align_corners=False) 

        return upsampled_attention * x     

class UNet(nn.Module):
    def __init__(self, image_channels:int=3, output_classes:int=33,device="cuda"):
        super(UNet, self).__init__()

        self.device =  device

        self.n_channel_layer_1 = 32
        self.n_channel_layer_2 = 2 * self.n_channel_layer_1 #64
        self.n_channel_layer_3 = 2 * self.n_channel_layer_2 #128
        self.n_channel_layer_4 = 2 * self.n_channel_layer_3 #256
        self.n_channel_layer_5 = 2 * self.n_channel_layer_4 #512 
        self.n_channel_layer_6 = 2 * self.n_channel_layer_5 #1024

        self.channels = [self.n_channel_layer_1, self.n_channel_layer_2, self.n_channel_layer_3, self.n_channel_layer_4, self.n_channel_layer_5, self.n_channel_layer_6]
        self.pooling_layers = [nn.AvgPool2d(2, 2)] * len(self.channels)

        self.encoder_module = nn.ModuleDict()
        input_channels = image_channels

        for i, n_channel in enumerate(self.channels):
            if n_channel == self.n_channel_layer_6:
                convblock = SimpleConvBlock(in_channels=input_channels, out_channels=n_channel, add_maxpool=False)
            else:             
                convblock = SimpleConvBlock(in_channels=input_channels, out_channels=n_channel, add_dropout=True)
            
            self.encoder_module.add_module(name=f'convblock_{i}', module=convblock)
            input_channels = n_channel

        self.decoder_module = nn.ModuleDict()
        self.attention_gate_module = nn.ModuleDict()
        input_channels = n_channel

        for i, n_channel in enumerate(self.channels[-2::-1]):
            attn_gate = AttentionGate(input_channels, n_channel)
            deconvblock = SimpleDeConvBlock(in_channels=input_channels, out_channels=n_channel)
            # deconvblock = SimpleDeConvBlock_Revised(in_channels=input_channels, out_channels=n_channel)
            self.decoder_module.add_module(name=f'deconvblock_{i}', module=deconvblock)
            self.attention_gate_module.add_module(name=f'attn_gate_{i}', module=attn_gate)
            input_channels = n_channel

        self.output_classes = output_classes
        
        self.pre_classification_layer = nn.Conv2d(n_channel, n_channel, kernel_size=1, stride=1)
        self.final_classification_layer = nn.Conv2d(n_channel, output_classes, kernel_size=1, stride=1)


    def forward(self, image_tensors:torch.tensor, label_tensors:torch.tensor):

        encoder_features = []
        
        for idx, module in enumerate(list(self.encoder_module)[:-1]):
            output_features = self.encoder_module[module](image_tensors)
            pooled_features = self.pooling_layers[idx](output_features)
            image_tensors = pooled_features

            encoder_features.append(output_features.clone())

        last_encoder_module = list(self.encoder_module)[-1]
        output_features = self.encoder_module[last_encoder_module](image_tensors)
    
        for idx, module in enumerate(self.decoder_module):
            attention_feature = self.attention_gate_module[f'attn_gate_{idx}'](output_features, encoder_features[-1])
            # output_features = self.decoder_module[module](output_features, encoder_features[-1])
            output_features = self.decoder_module[module](output_features, attention_feature)
            
            del encoder_features[-1]
        
        
        predicted_segmentation_map = self.pre_classification_layer(output_features)
        predicted_segmentation_map = self.final_classification_layer(output_features)

        # if predicted_segmentation_map.shape[2:] != label_tensors.shape[2:]:
        #     predicted_segmentation_map = F.interpolate(predicted_segmentation_map, list(label_tensors.shape)[1:])

        # predicted_segmentation_map = nn.Softmax(dim=1)(predicted_segmentation_map)     

        # predicted_segmentation_map = torch.argmax(predicted_segmentation_map, dim=1)   
        
        # cross_loss = nn.CrossEntropyLoss(ignore_index=0)(
        #     predicted_segmentation_map, label_tensors.squeeze(1).long()
        # )

        # focal_loss = calculate_focal_loss(cross_loss)

        dice_loss = calculate_dice_loss(
            predicted_segmentation_map, label_tensors, self.output_classes
        )        

        return dice_loss, predicted_segmentation_map, label_tensors

        # iou_loss = calculate_soft_iou_loss(
        #     predicted_segmentation_map, label_tensors, self.output_classes
        # )

        # loss = calculate_focal_loss(
        #     predicted_segmentation_map, label_tensors
        # )

            


        
    