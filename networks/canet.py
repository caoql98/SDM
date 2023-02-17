#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 16:35:19 2021

@author: caoqinglong
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone import resnet_dialated as resnet
from models import ASPP
from models.backbone import vgg
from torchvision import transforms
import numpy as np
# The Code of baseline network is referenced from https://github.com/icoz69/CaNet
# The code of training & testing is referenced from https://github.com/xiaomengyc/SG-One

class OneModel(nn.Module):
    def __init__(self, args):


        super(OneModel, self).__init__()
        # self.model_backbone = vgg.vgg16(pretrained=True)
        self.model_res = resnet.Res50_Deeplab(pretrained=True)
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=2, dilation=2, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(256 , 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=1536, out_channels=256, kernel_size=3, stride=1, padding=2, dilation=2, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.residule1=nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(256+2,256,kernel_size=3,stride=1,padding=1,bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        )

        self.residule2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(256 , 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        )

        self.residule3 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.layer6 = ASPP.PSPnet()
        self.layer7 = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU()

        )
        self.layer9 = nn.Conv2d(256, 2, kernel_size=1, stride=1, bias=True)
        self.batch_size = args.batch_size

    def forward(self, query_rgb, support_rgb, support_mask, query_mask):
        # extract support_ feature
        support_feature = self.extract_feature_res(support_rgb)
        
        # support_feature = self.layer5(support_feature)
        # extract query feature
        query_feature = self.extract_feature_res(query_rgb)
        # query_feature = self.layer5(query_feature)
        b,c,h,w = query_feature.shape
        vec_pos = self.Weighted_GAP(support_feature, support_mask)
        fea_pos = vec_pos.expand(-1, -1, h, w)  # tile for cat
        exit_feat_in = torch.cat([query_feature, fea_pos], dim=1)
        feature_query = self.layer1(exit_feat_in)

        historty_mask = torch.zeros(b,2,h,w).fill_(0.0).cuda()


        for i in range(4):

            out = torch.cat([feature_query,historty_mask],dim=1)
            out = feature_query + self.residule1(out)
            out = out + self.residule2(out)
            out = out + self.residule3(out)
            out = self.layer6(out)
            out = self.layer7(out)
            historty_mask = self.layer9(out)

        # b,_,_,_ = out.shape
        # unloader = transforms.ToPILImage()
        # for i in range(b):
        #     out1_1 = F.interpolate(historty_mask, query_rgb.shape[-2:], mode='bilinear', align_corners=True)
        #     b,c,h,w = query_rgb.shape
        #     out1_1 = F.softmax(out1_1,dim=1)
        #     out1_1  = out1_1[:,1,:,:].view(b,1,h,w)
        #     out1_1 = out1_1>0.5
        #     image1 = out1_1[i,:,:,:].cpu().clone()  # clone the tensor
        #         # image = image.squeeze(0)  # remove the fake batch dimension
        #     # image1 = (image1*1).astype(np.uint8)
        #     image1 = np.array(image1).astype(np.uint8)
        #     image1 = torch.Tensor(image1)
        #     image1 = unloader(image1)
        #     image1.save('/disk2/caoqinglong/remote_sensing/vi_canet_3/'+query_name[0]+'_'+str(i)+'pred.jpg')
        #
        #     out1_1 = F.interpolate(query_mask, query_rgb.shape[-2:], mode='bilinear', align_corners=True)
        #     b,c,h,w = query_rgb.shape
        #     out1_1  = out1_1.view(b,1,h,w)
        #
        #     image1 = out1_1[i,:,:,:].cpu().clone()  # clone the tensor
        #         # image = image.squeeze(0)  # remove the fake batch dimension
        #     image1 = unloader(image1)
        #     image1.save('/disk2/caoqinglong/remote_sensing/vi_canet_3/'+query_name[0]+'_'+str(i)+'GT.jpg')
        return historty_mask



    def forward_5shot(self, query_rgb, support_rgb, support_mask,query_mask):

        B,K,C,H,W = support_rgb.shape
        support_rgb = support_rgb.contiguous().view(B*K,C,H,W)
        support_mask = support_mask.contiguous().view(B*K,1,H,W)
        # extract support_ feature
        support_feature = self.extract_feature_res(support_rgb)

        query_feature = self.extract_feature_res(query_rgb)
        b, c, h, w = query_feature.shape
        
        f_prototypes = self.Weighted_GAP(support_feature, support_mask)
        s_f = f_prototypes.contiguous().view(B,K,c,1,1)

        f_query = []
        f_weight = []
        for i in range(K):
            support_prototype = s_f[:,i,:,:,:]
            fea_pos = support_prototype.expand(-1, -1, h, w)  # tile for cat
            exit_feat_in = torch.cat([query_feature, fea_pos], dim=1)
            feature_query = self.layer1(exit_feat_in)
            weight_query = self.layer2(feature_query)
            weight_query = weight_query.mean(dim=[2,3],keepdim=True)
            f_query.append(feature_query)
            f_weight.append(weight_query.contiguous().unsqueeze(-1))

        f_weight = torch.cat(f_weight, dim =-1)
        f_weight = F.softmax(f_weight,dim=-1)
        f_weight = f_weight.contiguous()
        weights_list = []
        for i in range(K):
            weights_list.append(f_weight[:,:,:,:,i])

        feature_query = f_query[0]*weights_list[0]
        for i in range(1,K):
            feature_query = feature_query+(f_query[i]*weights_list[i])
        feature_query = feature_query.contiguous()



        historty_mask = torch.zeros(b,2, h, w).fill_(0.0).cuda()

        for i in range(4):

            out = torch.cat([feature_query, historty_mask], dim=1)
            out = feature_query + self.residule1(out)
            out = out + self.residule2(out)
            out = out + self.residule3(out)
            out = self.layer6(out)
            out = self.layer7(out)
            historty_mask = self.layer9(out)


        return historty_mask
    def extract_feature_res(self, rgb):
        out_resnet = self.model_res(rgb)
        stage2_out = out_resnet[1]
        stage3_out = out_resnet[2]
        out_23 = torch.cat([stage2_out, stage3_out], dim=1)
        feature = self.layer5(out_23)

        return feature
    
    # def extract_feature_res(self, rgb):
    #     feature = self.model_backbone(rgb)
    #     feature = self.layer5(feature)
    #     return feature
    
    def f_v_concate(self, feature, vec_pos, feature_size):
        fea_pos = vec_pos.expand(-1, -1, feature_size[0], feature_size[1])  # tile for cat
        exit_feat_in = torch.cat([feature, fea_pos], dim=1)

        return exit_feat_in

    def get_loss(self, logits, query_label, idx):
        bce_logits_func = nn.CrossEntropyLoss()
        query_side  = logits

        b, c, w, h = query_label.size()
        query_side = F.upsample(query_side, size=(w, h), mode='bilinear')

        query_label = query_label.view(b, -1)
        bb, cc, _, _ = query_side.size()
        query_side = query_side.view(b, cc, w * h)

        loss_query = bce_logits_func(query_side, query_label.long())



        loss = loss_query

        return loss, loss_query, loss_query

    def get_pred(self, logits, query_image):
        outB = logits
        w, h = query_image.size()[-2:]
        outB_side1 = F.upsample(outB, size=(w, h), mode='bilinear')
        out_softmax = F.softmax(outB_side1, dim=1)
        values, pred = torch.max(out_softmax, dim=1)
        return out_softmax, pred
    
    def Weighted_GAP(self,supp_feat, mask):

        supp_feat = F.interpolate(supp_feat, mask.shape[-2:], mode='bilinear', align_corners=True)
        supp_feat = supp_feat * mask
        feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
        area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
        supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area  
        return supp_feat