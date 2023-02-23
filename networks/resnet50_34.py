import torch
import torch.nn as nn
import torch.nn.functional as F
# from models.backbone import resnet_dialated as models1
import numpy as np
from models.backbone import resnet as models1
from models.backbone import vgg as vgg_models
from models import ASPP
# from models.GRN import GRN
# from models.mRN import mRN
# from models.PMMs import PMMs
from torchvision import transforms
from models.PMMs_single import PMMs
import math
# The Code of baseline network is referenced from https://github.com/icoz69/CaNet
# The code of training & testing is referenced from https://github.com/xiaomengyc/SG-One
def get_vgg16_layer(model):
    layer0_idx = range(0,7)
    layer1_idx = range(7,14)
    layer2_idx = range(14,24)
    layer3_idx = range(24,34)
    layer4_idx = range(34,43)
    layers_0 = []
    layers_1 = []
    layers_2 = []
    layers_3 = []
    layers_4 = []
    for idx in layer0_idx:
        layers_0 += [model.features[idx]]
    for idx in layer1_idx:
        layers_1 += [model.features[idx]]
    for idx in layer2_idx:
        layers_2 += [model.features[idx]]
    for idx in layer3_idx:
        layers_3 += [model.features[idx]]
    for idx in layer4_idx:
        layers_4 += [model.features[idx]]
    layer0 = nn.Sequential(*layers_0)
    layer1 = nn.Sequential(*layers_1)
    layer2 = nn.Sequential(*layers_2)
    layer3 = nn.Sequential(*layers_3)
    layer4 = nn.Sequential(*layers_4)
    return layer0,layer1,layer2,layer3,layer4

class OneModel(nn.Module):
    def __init__(self, args):
        self.inplanes = 64
        self.num_pro4 = 3
        self.num_pro3 = 3
        self.num_pro2 = 3
        self.num_pro1 = 3
        super(OneModel, self).__init__()
        # self.layer1 = nn.Sequential(
        #     nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=2, dilation=2, bias=True),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU())
        # self.layer2 = nn.Sequential(
        #     nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=2, dilation=2, bias=True),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU())
        # self.layer3 = nn.Sequential(
        #     nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=2, dilation=2, bias=True),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU())
        # self.layer4 = nn.Sequential(
        #     nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=2, dilation=2, bias=True),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU())
        self.reduce_dim4 = 512
        self.reduce_dim3 = 256
        self.reduce_dim2 = 256
        self.reduce_dim1 = 128
        # self.reduce_dim = 256
        self.layer45 = nn.Sequential(
            nn.Conv2d(in_channels=self.reduce_dim4* 2, out_channels=512, kernel_size=3, stride=1, padding=2, dilation=2,
                      bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.layer46 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1,
                      bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        
        self.layer35 = nn.Sequential(
            nn.Conv2d(in_channels=self.reduce_dim3 * 2, out_channels=256, kernel_size=3, stride=1, padding=2, dilation=2,
                      bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.layer36 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1,
                      bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        self.layer25 = nn.Sequential(
            nn.Conv2d(in_channels=self.reduce_dim2 * 2, out_channels=256, kernel_size=3, stride=1, padding=2, dilation=2,
                      bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.layer26 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1,
                      bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        self.layer15 = nn.Sequential(
            nn.Conv2d(in_channels=self.reduce_dim1* 2, out_channels=128, kernel_size=3, stride=1, padding=2, dilation=2,
                      bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.layer16 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1,
                      bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        # self.conv1_dsn6 = nn.Conv2d(1024, 256, kernel_size=1)
        # self.conv2_dsn6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # self.conv3_dsn6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # self.conv4_dsn6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # self.conv5_dsn6 = nn.Conv2d(256, 2, kernel_size=1)
        # self.channel_1 =int(128/2)
        # self.channel_2 = int(256/2)
        # self.channel_3 = int(256/2)
        # self.channel_4 = int(512/2)
        self.channel_1 = 128
        self.channel_2 = 256
        self.channel_3 = 256
        self.channel_4 = 512
        self.residule1_1 = nn.Sequential(
            nn.Conv2d(self.channel_1+2, self.channel_1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(self.channel_1, self.channel_1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
        )
        self.residule1_2 = nn.Sequential(
            nn.Conv2d(self.channel_1,self.channel_1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(self.channel_1, self.channel_1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
        )
        self.layer1_9 = nn.Conv2d(self.channel_1, 2, kernel_size=1, stride=1, bias=True) 
        
        
        
        
        self.residule2_1 = nn.Sequential(
            nn.Conv2d(self.channel_2+2, self.channel_2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(self.channel_2, self.channel_2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
        )
        self.residule2_2 = nn.Sequential(
            nn.Conv2d(self.channel_2, self.channel_2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(self.channel_2, self.channel_2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
        )
        self.layer2_9 = nn.Conv2d(self.channel_2, 2, kernel_size=1, stride=1, bias=True) 
        
        
        
        self.residule3_1 = nn.Sequential(
            nn.Conv2d(self.channel_3+2, self.channel_3, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(self.channel_3, self.channel_3, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
        )
        self.residule3_2 = nn.Sequential(
            nn.Conv2d(self.channel_3,self.channel_3, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(self.channel_3, self.channel_3, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
        )
        self.layer3_9 = nn.Conv2d(self.channel_2, 2, kernel_size=1, stride=1, bias=True) 
        
        
                
        self.residule4_1 = nn.Sequential(
            nn.Conv2d(self.channel_4+2,self.channel_4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(self.channel_4, self.channel_4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
        )
        self.residule4_2 = nn.Sequential(
            nn.Conv2d(self.channel_4, self.channel_4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(self.channel_4, self.channel_4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
        )
        self.layer4_9 = nn.Conv2d(self.channel_4, 2, kernel_size=1, stride=1, bias=True) 
        
        self.dsn4_3 = nn.ConvTranspose2d(2, 2, kernel_size=2, stride=2)
        self.dsn3_2 = nn.ConvTranspose2d(2, 2, kernel_size=2, stride=2)
        self.dsn2_1 = nn.ConvTranspose2d(2, 2, kernel_size=2, stride=2)
        # self.batchnorm1=nn.BatchNorm2d(256)
        # self.batchnorm2=nn.BatchNorm2d(64)
        # self.batch_size = args.batch_size
        # self.layer6 = ASPP.PSPnet()
        
        models1.BatchNorm = nn.BatchNorm2d
        self.ppm_scales=[80, 40, 20, 10]
        self.modeln = modeln.NNet(pretrained = True)
        # resnet = models1.Res50_Deeplab(pretrained = True)

        resnet = models1.resnet50(pretrained = True)
        # vgg_models.BatchNorm = nn.BatchNorm2d
        # vgg16 = vgg_models.vgg16_bn(pretrained=True)
        # self.layers0, self.layers1, self.layers2, self.layers3, self.layers4 = get_vgg16_layer(vgg16)

        self.avgpool_list = []
        for bin in self.ppm_scales:
            if bin > 1:
                self.avgpool_list.append(
                    nn.AdaptiveAvgPool2d(bin)
                )
        self.layers0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1, resnet.conv2, resnet.bn2, resnet.relu2, resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
        self.layers1, self.layers2, self.layers3,self.layers4 = resnet.layer1, resnet.layer2, resnet.layer3,resnet.layer4
        # # self.model_res = resnet.Res50_Deeplab(pretrained=True)

        fea_dim = 1024 + 512
        # fea_dim = 512 + 256
        self.reduce_dim = 256
        # self.down_query = nn.Sequential(
        #     nn.Conv2d(fea_dim, self.reduce_dim, kernel_size=1, padding=0, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout2d(p=0.5)                  
        # )
        # self.down_supp = nn.Sequential(
        #     nn.Conv2d(fea_dim, self.reduce_dim, kernel_size=1, padding=0, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout2d(p=0.5)                   
        # ) 
        
        self.down_query4 = nn.Sequential(
            nn.Conv2d(fea_dim, self.reduce_dim4, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)                  
        )
        self.down_supp4 = nn.Sequential(
            nn.Conv2d(fea_dim, self.reduce_dim4, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)                   
        ) 
        
        self.down_query3 = nn.Sequential(
            nn.Conv2d(fea_dim, self.reduce_dim3, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)                  
        )
        self.down_supp3 = nn.Sequential(
            nn.Conv2d(fea_dim, self.reduce_dim3, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)                   
        ) 
        
        self.down_query2 = nn.Sequential(
            nn.Conv2d(fea_dim, self.reduce_dim2, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)                  
        )
        self.down_supp2 = nn.Sequential(
            nn.Conv2d(fea_dim, self.reduce_dim2, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)                   
        ) 
        
        self.down_query1 = nn.Sequential(
            nn.Conv2d(fea_dim, self.reduce_dim1, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)                  
        )
        self.down_supp1 = nn.Sequential(
            nn.Conv2d(fea_dim, self.reduce_dim1, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)                   
        ) 
        # self.drop01 = nn.Dropout2d(p=0.1)
        
        # self.nns1 = nn.Sequential(
        #     nn.Conv2d(192,32, kernel_size=3, stride=1, padding=1, bias=True),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True),
        #     nn.ReLU(),
        # )
        # self.nns2 = nn.Sequential(
        #     nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True),
        #     nn.ReLU(),
        # )
        # self.nns3 = nn.Sequential(
        #     nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True),
        #     nn.ReLU(),
        # )
        # self.am = nn.Sequential(
        #     nn.Conv2d(32,32, kernel_size=3, stride=1, padding=1, bias=True),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1, bias=True),
        # )
        
    def forward(self, query_rgb,support_rgb, support_mask):
        # resout_query = self.model_res(query_rgb)
        # resout_support = self.model_res(support_rgb)
        
        # with torch.no_grad():
        #     out_nn =  self.modeln(query_rgb, out_feat_keys=self.modeln.all_feat_names)
        #     out_nns = self.modeln(support_rgb, out_feat_keys=self.modeln.all_feat_names)
        #     out_f = out_nn[1]
        #     out_fs = out_nns[1]
        #     # out_f = out_f.sum(dim=1 , keepdim= True)
        #     # b,c,h,w = out_f.shape
        #     # max1 = out_f.view(b,c,h*w).max(dim=2)[0]
        #     # max1 = max1.view(b,c,1,1)
        #     # out_fq = out_f/max1
        # mask_nn = F.interpolate(support_mask , out_fs.shape[-2:], mode='bilinear', align_corners=True)
        # out_fs = out_fs*mask_nn
        # out_fs = out_fs.mean(dim=0,keepdim=True)
        # out_fs = out_fs.mean(dim=[2, 3], keepdim=True)
        # out_f = F.conv2d(out_f, out_fs.permute(1,0,2,3), groups=192)
        # out_f = self.nns1(out_f)
        # out_f = out_f + self.nns2(out_f)
        # out_f = out_f + self.nns3(out_f)
        # out_map = self.am(out_f)
        # out_map = torch.sigmoid(out_map)
        
        l = support_rgb.size()

        if len(l)==5:
            B, K, C, H, W = support_rgb.shape
            support_rgb = support_rgb.view(B*K,C,H,W)
            support_mask = support_mask.view(B*K,1,H,W)
         #   Query Feature
        with torch.no_grad():
            query_feat_0 = self.layers0(query_rgb)
            
            # out_map01 =  F.interpolate(out_map, query_feat_0.shape[-2:], mode='bilinear', align_corners=True)
            # query_feat_0 = out_map01*query_feat_0 +query_feat_0 
            
            query_feat_1 = self.layers1(query_feat_0)
            # out_map01 =  F.interpolate(out_map, query_feat_1.shape[-2:], mode='bilinear', align_corners=True)
            # query_feat_1 = out_map01*query_feat_1 +query_feat_1
            query_feat_2 = self.layers2(query_feat_1)
            
            # out_map23 =  F.interpolate(out_map, query_feat_2.shape[-2:], mode='bilinear', align_corners=True)
            # query_feat_2 = out_map23*query_feat_2 +query_feat_2
            
            query_feat_3 = self.layers3(query_feat_2)
            # out_map23 =  F.interpolate(out_map, query_feat_3.shape[-2:], mode='bilinear', align_corners=True)
            # query_feat_3 = out_map23*query_feat_3 +query_feat_3
            
            query_feat_4 = self.layers4(query_feat_3)
            query_feat_3 = F.interpolate(query_feat_3, size=(query_feat_2.size(2),query_feat_2.size(3)), mode='bilinear', align_corners=True)
        
        with torch.no_grad():
            supp_feat_0 = self.layers0(support_rgb)
            supp_feat_1 = self.layers1(supp_feat_0)
            supp_feat_2 = self.layers2(supp_feat_1)
            supp_feat_3 = self.layers3(supp_feat_2)  
            mask = F.interpolate(support_mask, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='bilinear', align_corners=True)
            supp_feat_4f = self.layers4(supp_feat_3*mask)
            supp_feat_4b = self.layers4(supp_feat_3*(1-mask))
            supp_feat_3 = F.interpolate(supp_feat_3, size=(supp_feat_2.size(2),query_feat_2.size(3)), mode='bilinear', align_corners=True)
            
 
            
        #     # out_nn =  self.modeln(support_rgb, out_feat_keys=self.modeln.all_feat_names)
        #     # out_f = out_nn[1]
        #     # out_f = out_f.sum(dim=1 , keepdim= True)
        #     # b,c,h,w = out_f.shape
        #     # max1 = out_f.view(b,c,h*w).max(dim=2)[0]
        #     # max1 = max1.view(b,c,1,1)
        #     # out_fs = out_f/max1
            
            # out_b = 1-out_f
            # out_q =  torch.cat([out_b, out_f], dim=1)
            
            # unloader = transforms.ToPILImage()
            # b,c,h,w = query_rgb.shape
            
            
        bf,cf,hf,wf = supp_feat_4f.shape
        mask = F.interpolate(support_mask,  supp_feat_4f.shape[-2:], mode='bilinear', align_corners=True)
        cosine_eps = 1e-7                   
        q =  query_feat_4
        s_f = supp_feat_4f * mask 
        s_b = supp_feat_4b * (1-mask)
        bsize, ch_sz, sp_sz, _ = q.size()[:]

        tmp_query = q
        tmp_query = tmp_query.contiguous().view(bsize, ch_sz, -1)
        tmp_query_norm = torch.norm(tmp_query, 2, 1, True) 

        if len(l)==5:
            for i in range(5):
                s_f = s_f.view(B,K,cf,hf,wf)
                tmp_supp = s_f[:,i,:,:,:]
                tmp_supp = tmp_supp.contiguous().view(bsize, ch_sz, -1)
                tmp_supp = tmp_supp.contiguous().permute(0, 2, 1)
                tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)

                similarity = torch.bmm(tmp_supp, tmp_query) / (torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
                similarity = similarity.max(1)[0].view(bsize, sp_sz * sp_sz)
                similarity = (similarity - similarity.min(1)[0].unsqueeze(1))/(similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
                if i == 0:
                    corr_query_f = similarity.view(B, 1, sp_sz, sp_sz)
                else:
                    corr_query_f = torch.cat([corr_query_f, similarity.view(B, 1, sp_sz, sp_sz)],dim=1)
            corr_query_f = corr_query_f.mean(dim=1,keepdim=True)
        else:
            tmp_supp = s_f
            tmp_supp = tmp_supp.contiguous().view(bsize, ch_sz, -1)
            tmp_supp = tmp_supp.contiguous().permute(0, 2, 1)
            tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)

            similarity = torch.bmm(tmp_supp, tmp_query)/(torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
            similarity = similarity.max(1)[0].view(bsize, sp_sz*sp_sz)
            similarity = (similarity - similarity.min(1)[0].unsqueeze(1))/(similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
            corr_query_f = similarity.view(bsize, 1, sp_sz, sp_sz)

        if len(l)==5:
            for i in range(5):
                s_b = s_b.view(B, K, cf, hf, wf)
                tmp_supp = s_b[:, i, :, :, :]
                tmp_supp = tmp_supp.contiguous().view(bsize, ch_sz, -1)
                tmp_supp = tmp_supp.contiguous().permute(0, 2, 1)
                tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)

                similarity = torch.bmm(tmp_supp, tmp_query) / (torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
                similarity = similarity.max(1)[0].view(bsize, sp_sz * sp_sz)
                similarity = (similarity - similarity.min(1)[0].unsqueeze(1))/(similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
                if i == 0:
                    corr_query_b = similarity.view(B, 1, sp_sz, sp_sz)
                else:
                    corr_query_b = torch.cat([corr_query_b, similarity.view(B, 1, sp_sz, sp_sz)],dim=1)
            corr_query_b = corr_query_b.mean(dim=1,keepdim=True)
        else:
            tmp_supp = s_b
            tmp_supp = tmp_supp.contiguous().view(bsize, ch_sz, -1)
            tmp_supp = tmp_supp.contiguous().permute(0, 2, 1)
            tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)

            similarity = torch.bmm(tmp_supp, tmp_query)/(torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
            similarity = similarity.max(1)[0].view(bsize, sp_sz*sp_sz)
            similarity = (similarity - similarity.min(1)[0].unsqueeze(1))/(similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
            corr_query_b = similarity.view(bsize, 1, sp_sz, sp_sz)

        
        Prob_map = torch.cat([corr_query_b,corr_query_f],1)
        
        
        
        
        resout_query = torch.cat([query_feat_3, query_feat_2], 1)
        resout_support = torch.cat([supp_feat_3, supp_feat_2], 1)

        feature_query = self.avgpool_list[3](self.down_query4(resout_query))
        feature_support = self.avgpool_list[3]( self.down_supp4(resout_support))
        
        b,c,h,w =  feature_query.shape

        # GRN_model4 = GRN(c).cuda()
        PMMs4 = PMMs(c, self.num_pro4).cuda()

        if len(l)==5:
            for i in range(5):
                feature_support = feature_support.view(B,K,c,h,w)
                support_mask = support_mask.view(B,K,1,H,W)
                if i == 0:
                    support_feature_all = feature_support[:,i,:,:,:]
                    support_mask_all = support_mask[:,i,:,:,:]
                else:
                    support_feature_all = torch.cat([support_feature_all,feature_support[:,i,:,:,:]], dim=2)
                    support_mask_all = torch.cat([support_mask_all, support_mask[:,i,:,:,:]], dim=2)
        else:
            support_feature_all = feature_support
            support_mask_all = support_mask
       
        vec_pos = PMMs4(support_feature_all, support_mask_all, feature_query)

        # feature concate
        feature_size = feature_query.shape[-2:]
        for i in range(self.num_pro4):
            vec = vec_pos[i]
            exit_feat_in_ = self.f_v_concate(feature_query, vec, feature_size)
            exit_feat_in_ = self.layer45(exit_feat_in_)
            if i == 0:
                exit_feat_in = exit_feat_in_
            else:
                exit_feat_in = exit_feat_in + exit_feat_in_
        exit_feat_in = self.layer46(exit_feat_in)
        feature_size = exit_feat_in.shape[-2:]
        Prob_map= F.interpolate(Prob_map, feature_size, mode='bilinear', align_corners=True)
        # out_fq_4= F.interpolate(out_fq, feature_size, mode='bilinear', align_corners=True)
        
        out = exit_feat_in
        # out_plus_history = exit_feat_in
        out_plus_history = torch.cat([out, Prob_map], dim=1)
        # out_plus_history = torch.cat([out_plus_history, out_fq_4], dim=1)
        out = out + self.residule4_1(out_plus_history)
        out = out + self.residule4_2(out)
        # out = out + self.residule3(out)
        # out = self.drop01(out)
        out4 = self.layer4_9(out)
        
        # out4_3 = self.dsn4_3(out4)

        
        feature_query = self.avgpool_list[2](self.down_query3(resout_query))
        feature_support =self.avgpool_list[2](self.down_supp3(resout_support))
        
        feature_size1 = feature_query.shape[-2:]
        out4_3 = F.interpolate(out4, feature_size1, mode='bilinear', align_corners=True)
        b,c,h,w = out4_3.shape
        out4_3_1 = F.softmax(out4_3,dim=1)
        out4_3_1  = out4_3_1[:,1,:,:].view(b,1,h,w)
        # out_fq3 = F.interpolate(out_fq, feature_size1 , mode='bilinear', align_corners=True)
        # out4_3_1 = out4_3_1+out_fq3*0.3
        x = -1*torch.sigmoid(out4_3_1)+1
        
        x = x.expand_as(feature_query)
        feature_query = x*feature_query
        b,c,h,w =  feature_query.shape

        PMMs3 = PMMs(c, self.num_pro3).cuda()

        if len(l)==5:
            for i in range(5):
                feature_support = feature_support.view(B,K,c,h,w)
                support_mask = support_mask.view(B,K,1,H,W)
                if i == 0:
                    support_feature_all = feature_support[:,i,:,:,:]
                    support_mask_all = support_mask[:,i,:,:,:]
                else:
                    support_feature_all = torch.cat([support_feature_all,feature_support[:,i,:,:,:]], dim=2)
                    support_mask_all = torch.cat([support_mask_all, support_mask[:,i,:,:,:]], dim=2)
        else:
            support_feature_all = feature_support
            support_mask_all = support_mask
        vec_pos = PMMs3(support_feature_all, support_mask_all, feature_query)

        # feature concate
        feature_size = feature_query.shape[-2:]
        for i in range(self.num_pro3):
            vec = vec_pos[i]
            exit_feat_in_ = self.f_v_concate(feature_query, vec, feature_size)
            exit_feat_in_ = self.layer35(exit_feat_in_)
            if i == 0:
                exit_feat_in = exit_feat_in_
            else:
                exit_feat_in = exit_feat_in + exit_feat_in_
        exit_feat_in = self.layer36(exit_feat_in)
        feature_size = exit_feat_in.shape[-2:]
        Prob_map = F.interpolate(Prob_map, feature_size, mode='bilinear', align_corners=True)
        # out_fq_3 = F.interpolate(out_fq, feature_size, mode='bilinear', align_corners=True)
        
        out = exit_feat_in
        # out_plus_history = exit_feat_in
        out_plus_history = torch.cat([out, Prob_map], dim=1)
        # out_plus_history = torch.cat([out_plus_history, out_fq_3], dim=1)
        
        out = out + self.residule3_1(out_plus_history)
        out = out + self.residule3_2(out)
        # out = out + self.residule3(out)
        # out = self.drop01(out)
        out3 = self.layer3_9(out)+out4_3
        # out_softmax3 = F.softmax(out3, dim=1)
        
        # out3_2 = self.dsn4_3(out4)
 
    
        feature_query = self.avgpool_list[1](self.down_query2(resout_query))
        feature_support = self.avgpool_list[1](self.down_supp2(resout_support))

        
        feature_size1 = feature_query.shape[-2:]
        out3_2 = F.interpolate(out3, feature_size1, mode='bilinear', align_corners=True)
        b,c,h,w = out3_2.shape
        out3_2_1 = F.softmax(out3_2,dim=1)
        out3_2_1  = out3_2_1[:,1,:,:].view(b,1,h,w)
        # out_fq2 = F.interpolate(out_fq, feature_size1 , mode='bilinear', align_corners=True)
        # out3_2_1 = out3_2_1+out_fq2*0.3
        x = -1*torch.sigmoid(out3_2_1)+1
        
        x = x.expand_as(feature_query)
        feature_query = x*feature_query
        b,c,h,w =  feature_query.shape

        PMMs2 = PMMs(c, self.num_pro2).cuda()

        if len(l)==5:
            for i in range(5):
                feature_support = feature_support.view(B,K,c,h,w)
                support_mask = support_mask.view(B,K,1,H,W)
                if i == 0:
                    support_feature_all = feature_support[:,i,:,:,:]
                    support_mask_all = support_mask[:,i,:,:,:]
                else:
                    support_feature_all = torch.cat([support_feature_all,feature_support[:,i,:,:,:]], dim=2)
                    support_mask_all = torch.cat([support_mask_all, support_mask[:,i,:,:,:]], dim=2)
        else:
            support_feature_all = feature_support
            support_mask_all = support_mask
        vec_pos = PMMs2(support_feature_all, support_mask_all, feature_query)

        # feature concate
        feature_size = feature_query.shape[-2:]
        for i in range(self.num_pro2):
            vec = vec_pos[i]
            exit_feat_in_ = self.f_v_concate(feature_query, vec, feature_size)
            exit_feat_in_ = self.layer25(exit_feat_in_)
            if i == 0:
                exit_feat_in = exit_feat_in_
            else:
                exit_feat_in = exit_feat_in + exit_feat_in_
        exit_feat_in = self.layer26(exit_feat_in)
        feature_size = exit_feat_in.shape[-2:]
        Prob_map= F.interpolate(Prob_map, feature_size, mode='bilinear', align_corners=True)
        # out_fq_2 = F.interpolate(out_fq, feature_size, mode='bilinear', align_corners=True)
        
        out = exit_feat_in
        # out_plus_history = exit_feat_in
        out_plus_history = torch.cat([out, Prob_map], dim=1)
        # out_plus_history = torch.cat([out_plus_history, out_fq_2], dim=1)
         
        out = out + self.residule2_1(out_plus_history)
        out = out + self.residule2_2(out)
        # out = out + self.residule3(out)
        # out = self.drop01(out)
        out2 = self.layer2_9(out)+out3_2
        # out_softmax2 = F.softmax(out2, dim=1)
        
        # out3_2 = self.dsn4_3(out4)

        
        feature_query = self.avgpool_list[0](self.down_query1(resout_query))
        feature_support = self.avgpool_list[0](self.down_supp1(resout_support))

        
        feature_size1 = feature_query.shape[-2:]
        out2_1 = F.interpolate(out2, feature_size1, mode='bilinear', align_corners=True)
        b,c,h,w = out2_1.shape
        out2_1_1 = F.softmax(out2_1,dim=1)
        out2_1_1  = out2_1_1[:,1,:,:].view(b,1,h,w)
        # out_fq1 = F.interpolate(out_fq, feature_size1 , mode='bilinear', align_corners=True)
        # out2_1_1 = out2_1_1+out_fq1*0.3
            
        x = -1*torch.sigmoid(out2_1_1)+1
        
        x = x.expand_as(feature_query)
        feature_query = x*feature_query
        b,c,h,w =  feature_query.shape

        PMMs1 = PMMs(c, self.num_pro1).cuda()

        if len(l)==5:
            for i in range(5):
                feature_support = feature_support.view(B,K,c,h,w)

                support_mask = support_mask.view(B,K,1,H,W)
                if i == 0:
                    support_feature_all = feature_support[:,i,:,:,:]
                    support_mask_all = support_mask[:,i,:,:,:]
                else:
                    support_feature_all = torch.cat([support_feature_all,feature_support[:,i,:,:,:]], dim=2)
                    support_mask_all = torch.cat([support_mask_all, support_mask[:,i,:,:,:]], dim=2)
        else:
            support_feature_all = feature_support
            support_mask_all = support_mask
        vec_pos = PMMs1(support_feature_all, support_mask_all, feature_query)

        # feature concate
        feature_size = feature_query.shape[-2:]
        for i in range(self.num_pro1):
            vec = vec_pos[i]
            exit_feat_in_ = self.f_v_concate(feature_query, vec, feature_size)
            exit_feat_in_ = self.layer15(exit_feat_in_)
            if i == 0:
                exit_feat_in = exit_feat_in_
            else:
                exit_feat_in = exit_feat_in + exit_feat_in_
        exit_feat_in = self.layer16(exit_feat_in)
        feature_size = exit_feat_in.shape[-2:]
        Prob_map= F.interpolate(Prob_map, feature_size, mode='bilinear', align_corners=True)
        # out_fq_1 = F.interpolate(out_fq, feature_size, mode='bilinear', align_corners=True)
        
        out = exit_feat_in
        # out_plus_history = exit_feat_in
        out_plus_history = torch.cat([out, Prob_map], dim=1)
        # out_plus_history = torch.cat([out_plus_history,  out_fq_1], dim=1)
        
        out = out + self.residule1_1(out_plus_history)
        out = out + self.residule1_2(out)
        # out = out + self.residule3(out)
        # out = self.drop01(out)
        out1 = self.layer1_9(out)+out2_1
        return out1,out2,out3, out4

    def get_loss(self, logits,query_label, idx):
        bce_logits_func = nn.CrossEntropyLoss()
        # bce_logits_func = nn.CrossEntropyLoss(ignore_index=255)
        # bce_logits_func = nn.MSELoss()
        out1, out2, out3, out4 = logits
        # out1, out2, out3, out4,out_map = logits
        b, c, w, h = query_label.size()
        # query_label = query_label.view(b,1,w,h)
        out1= F.upsample(out1, size=(w, h), mode='bilinear')
        out2= F.upsample(out2, size=(w, h), mode='bilinear')
        out3= F.upsample(out3, size=(w, h), mode='bilinear')
        out4= F.upsample(out4, size=(w, h), mode='bilinear')
     
        
        b, c, w, h = query_label.size()
        query_label = query_label.view(b, -1)
        
        out1 = out1.view(-1, 2, w * h)
        out2 = out2.view(-1, 2, w * h)
        out3 = out3.view(-1, 2, w * h)
        out4 = out4.view(-1, 2, w * h)

        loss1 = bce_logits_func(out1, query_label.long())
        loss2 = bce_logits_func(out2, query_label.long())
        loss3 = bce_logits_func(out3, query_label.long())
        loss4 = bce_logits_func(out4, query_label.long())

        loss = loss1+loss2+loss3+loss4
        return loss, loss2, loss3 

    def get_pred(self, logits, query_image):
        # out1, out2, out3, out4, out5, out6 = logits
        out1,out2,out3, out4 = logits
        # out1, out2, out3, out4,out_map = logits
        b, c, w, h = query_image.size()
        # out1 = out1.view(-1, 2, w , h)
        # out2 = out2.view(-1, 2, w , h)
        # out3 = out3.view(-1, 2, w , h)
        # out4 = out4.view(-1, 2, w , h)
        out1 = F.upsample(out1, size=(w, h), mode='bilinear')
        out2 = F.upsample(out2, size=(w, h), mode='bilinear')
        out3 = F.upsample(out3, size=(w, h), mode='bilinear')
        out4 = F.upsample(out4, size=(w, h), mode='bilinear')
        
        # out3 = (out1+0.2*out2+0.1*out3+0.08*out4)/4
        # out3 = (out1+out2+out3+out4)/4
        out_softmax = F.softmax(out1, dim=1)

        values, pred = torch.max(out_softmax, dim=1)
        return out_softmax, pred
    def f_v_concate(self, feature, vec_pos, feature_size):
        fea_pos = vec_pos.expand(-1, -1, feature_size[0], feature_size[1])  # tile for cat
        exit_feat_in = torch.cat([feature, fea_pos], dim=1)

        return exit_feat_in

    # def Segmentation(self, feature, history_mask):
    #     feature_size = feature.shape[-2:]

    #     history_mask = F.interpolate(history_mask, feature_size, mode='bilinear', align_corners=True)
    #     out = feature
    #     out_plus_history = torch.cat([out, history_mask], dim=1)
    #     out = out + self.residule1(out_plus_history)
    #     out = out + self.residule2(out)
    #     # out = out + self.residule3(out)

    #     # out = self.layer6(out)
    #     # out = self.layer7(out)
    #     out = self.layer9(out)

    #     out_softmax = F.softmax(out, dim=1)

    #     return out, out_softmax
