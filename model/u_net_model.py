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

class ResidualConvBlock(nn.Module):

    def __init__(self, in_channels:int, out_channels:int, kernel_size:int=3, padding=1, add_dropout:bool=False, add_maxpool:bool=False):
        super(ResidualConvBlock, self).__init__()

        self.add_maxpool = add_maxpool
        self.add_dropout = add_dropout

        self.convblock_1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=True)
        self.convblock_2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=True)
        
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout()

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
            
    def forward(self, x):

        conv1_out = self.convblock_1(x)
        out = self.convblock_2(conv1_out)

        out = self.batch_norm(out)
        out = self.relu(out)

        out = out + conv1_out

        if self.add_maxpool:
            out = self.max_pool(out) 

        return self.relu(out)

class SimpleDeConvBlock(nn.Module):

    def __init__(self, in_channels:int, out_channels:int) -> None:
        super(SimpleDeConvBlock, self).__init__()

        #out = (x - 1)s - 2p + d(k - 1) + op + 1
        self.transpose_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.convblock = SimpleConvBlock(in_channels, out_channels)

    def forward(self, input_x:torch.tensor, concat_x:torch.tensor):

        transpose_output = self.transpose_layer(input_x)
        transpose_output = nn.ReLU()(transpose_output)

        concatenated_output = torch.concat([transpose_output, concat_x], dim=1)
        deconv_output = self.convblock(concatenated_output)

        return deconv_output

class ResidualDeConvBlock(nn.Module):

    def __init__(self, in_channels:int, out_channels:int) -> None:
        super(ResidualDeConvBlock, self).__init__()

        #out = (x - 1)s - 2p + d(k - 1) + op + 1
        self.transpose_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.convblock = ResidualConvBlock(in_channels, out_channels)

    def forward(self, input_x:torch.tensor, concat_x:torch.tensor):

        transpose_output = self.transpose_layer(input_x)
        transpose_output = nn.ReLU()(transpose_output)

        concatenated_output = torch.concat([transpose_output, concat_x], dim=1)
        deconv_output = self.convblock(concatenated_output)

        return deconv_output

class AttentionGate(nn.Module):

    def __init__(self, g:int, x:int):
        super(AttentionGate, self).__init__()

        '''
        g - lower layer feature map (has more channels). (last layer of encoder/previous decoder layer)
        x - upper layer feature map (has lesser channels). (skip connection from parallel encoder layer)
        '''

        self.g_conv_layer = nn.Conv2d(g, x, kernel_size=1) #(n_c, h, w) -> (m_c, h, w)
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
                # convblock = SimpleConvBlock(in_channels=input_channels, out_channels=n_channel, add_maxpool=False)
                convblock = ResidualConvBlock(in_channels=input_channels, out_channels=n_channel, add_maxpool=False)
            else:             
                # convblock = SimpleConvBlock(in_channels=input_channels, out_channels=n_channel, add_dropout=True)
                convblock = ResidualConvBlock(in_channels=input_channels, out_channels=n_channel, add_dropout=True)
            
            self.encoder_module.add_module(name=f'convblock_{i}', module=convblock)
            input_channels = n_channel

        self.decoder_module = nn.ModuleDict()
        self.attention_gate_module = nn.ModuleDict()
        input_channels = n_channel

        for i, n_channel in enumerate(self.channels[-2::-1]):
            attn_gate = AttentionGate(input_channels, n_channel)
            # deconvblock = SimpleDeConvBlock(in_channels=input_channels, out_channels=n_channel)
            deconvblock = ResidualDeConvBlock(in_channels=input_channels, out_channels=n_channel)
            self.decoder_module.add_module(name=f'deconvblock_{i}', module=deconvblock)
            self.attention_gate_module.add_module(name=f'attn_gate_{i}', module=attn_gate)
            input_channels = n_channel

        self.output_classes = output_classes
        
        self.pre_classification_layer = nn.Conv2d(n_channel, n_channel, kernel_size=1, stride=1)
        self.final_classification_layer = nn.Conv2d(n_channel, output_classes, kernel_size=1, stride=1)


    def forward(self, image_tensors:torch.tensor, label_tensors:torch.tensor, region_masks:torch.tensor=None):

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
        
        cross_loss = nn.CrossEntropyLoss(ignore_index=0)(
            predicted_segmentation_map, label_tensors.squeeze(1).long()
        )

        if region_masks is not None:
            ''' 
            TODO- Recheck how to formulate masked loss. 
            '''
            per_pixel_loss = F.cross_entropy(predicted_segmentation_map, label_tensors.squeeze(1).long(), reduction="none")
            masked_loss = per_pixel_loss * region_masks
            # print(masked_loss.size())
            # print(masked_loss.mean())
            
            if masked_loss.numel() > 0:  # Check if there are any elements to avoid division by zero
                #only compute mean for elements with 1s
                masked_loss = torch.masked_select(per_pixel_loss, region_masks.bool()).mean()
            else:
                masked_loss = torch.tensor(0.0, device=predicted_segmentation_map.device, requires_grad=True)            

            #alpha = 0.5
            #alpha*(1-alpha)
            total_loss = cross_loss + 0.5 * masked_loss

        else:
            total_loss = cross_loss

        return total_loss, predicted_segmentation_map, label_tensors


            


        
    