import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone import resnet_dialated as resnet
from models import ASPP
from models.PMMs import PMMs
import numpy as np
# The Code of baseline network is referenced from https://github.com/icoz69/CaNet
# The code of training & testing is referenced from https://github.com/xiaomengyc/SG-One

class OneModel(nn.Module):
    def __init__(self, args):

        self.inplanes = 64
        self.num_pro = 3
        super(OneModel, self).__init__()

        self.model_res = resnet.Res50_Deeplab(pretrained=True)
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=1536, out_channels=256, kernel_size=3, stride=1, padding=2, dilation=2, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU())
        
        # self.layer5_3 = nn.Sequential(
        #     nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, stride=1, padding=0, bias=True),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU())
        # self.layer5_2 = nn.Sequential(
        #     nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=True),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU())
        # self.layer5_1 = nn.Sequential(
        #     nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0, bias=True),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU())
        
        self.layer55 = nn.Sequential(
            nn.Conv2d(in_channels=256 * 2, out_channels=256, kernel_size=3, stride=1, padding=2, dilation=2,
                      bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.layer56 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1,
                      bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.layer6 = ASPP.PSPnet()

        self.layer7 = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU()

        )

        self.layer9 = nn.Conv2d(256, 2, kernel_size=1, stride=1, bias=True)  # numclass = 2

        self.residule1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(256+2, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        )

        self.residule2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        )

        self.residule3 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.PMMs = PMMs(256, self.num_pro).cuda()

        self.batch_size = args.batch_size
        
        # self.trans1 = nn.Sequential(
        #     nn.Conv2d(256+1, 256, kernel_size=1, stride=1, padding=0, bias=True),
        #     nn.ReLU(),
        #     nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=True),
        #     nn.ReLU(),
        # )
        # self.trans2 = nn.Sequential(
        #     nn.Conv2d(256+1, 256, kernel_size=1, stride=1, padding=0, bias=True),
        #     nn.ReLU(),
        #     nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=True),
        #     nn.ReLU(),
        # )
        self.fc_1 = nn.Linear(256, 64, bias=False)
        self.fc_2 = nn.Linear(64,256, bias=True)
        # torch.nn.init.normal_(self.fc_1.weight, mean=0, std=0.01)
        # self.s_conv = nn.Conv1d(1,1,kernel_size = 5, padding=int(4/2),bias=False)
        self.c_conv = nn.Conv1d(1,1,kernel_size = 5, padding=int(4/2),bias=False)
        self.fc_classify = nn.Linear(256,10)
    def forward(self, query_rgb, support_rgb, support_mask,label):
        
        # with torch.no_grad():
        # extract support_ feature
        support_feature = self.extract_feature_res(support_rgb)

        # extract query feature
        query_feature = self.extract_feature_res(query_rgb)
        
        support_mask_attentive= F.interpolate(support_mask, support_feature.shape[-2:], mode='bilinear',
                                              align_corners=True)
        
        class_vec = self.Weighted_GAP(support_feature,support_mask_attentive)
        back_vec = self.Weighted_GAP(support_feature,1-support_mask_attentive)
        
        
        feature_size = query_feature.shape[-2:]
        query_feature = self.narrow_gap(query_feature,class_vec)
        class_vector_i = query_feature.mean(dim=[2,3])
        exit_feat_in = self.f_v_concate(query_feature, class_vec, feature_size)
        exit_feat_in = self.layer55(exit_feat_in)
 
        b, c, h, w = query_feature.size()
        x = query_feature.view(b, c, h * w)  # b * c * n
        mu = torch.cat([class_vec.view(b,c,1),back_vec.view(b,c,1)],dim=2)# b * c * 2
        # with torch.no_grad():
        x_t = x.permute(0, 2, 1)  # b * n * c
        
        z = torch.bmm(x_t, mu)  # b * n * k

        z = F.softmax(z, dim=2)  # b * n * k
        P = z.permute(0, 2, 1)
        Prob_map = P.view(b, 2, h, w) #  b * k * w * h  probability map


        class_label = self.fc_classify(class_vector_i)

        # segmentation
        out, _ = self.Segmentation(exit_feat_in, Prob_map)
        
        return support_feature, query_feature, class_vec, out, class_label

    def forward_5shot(self, query_rgb, support_rgb_batch, support_mask_batch):
        # extract query feature
        query_feature = self.extract_feature_res(query_rgb)
        # feature concate
        feature_size = query_feature.shape[-2:]

        for i in range(support_rgb_batch.shape[1]):
            support_rgb = support_rgb_batch[:, i]
            support_mask = support_mask_batch[:, i]
            # extract support feature
            support_feature = self.extract_feature_res(support_rgb)
            support_mask_temp = F.interpolate(support_mask, support_feature.shape[-2:], mode='bilinear',
                                              align_corners=True)
            if i == 0:
                support_feature_all = support_feature
                support_mask_all = support_mask_temp
            else:
                support_feature_all = torch.cat([support_feature_all, support_feature], dim=2)
                support_mask_all = torch.cat([support_mask_all, support_mask_temp], dim=2)

        vec_pos, Prob_map = self.PMMs(support_feature_all, support_mask_all, query_feature)

        for i in range(self.num_pro):
            vec = vec_pos[i]
            exit_feat_in_ = self.f_v_concate(query_feature, vec, feature_size)
            exit_feat_in_ = self.layer55(exit_feat_in_)
            if i == 0:
                exit_feat_in = exit_feat_in_
            else:
                exit_feat_in = exit_feat_in + exit_feat_in_

        exit_feat_in = self.layer56(exit_feat_in)

        out, _ = self.Segmentation(exit_feat_in, Prob_map)

        return out, out, out, out

    def extract_feature_res(self, rgb):
        out_resnet = self.model_res(rgb)
        stage2_out = out_resnet[1]
        stage3_out = out_resnet[2]
        
        # stage1_out = out_resnet[0]
        # out3 = self.layer5_3(stage3_out)
        # out2 = self.layer5_2(stage2_out)+out3
        # b, c, w, h = stage1_out.size()
        # out2 = F.upsample(out2, size=(w, h), mode='bilinear')
        # feature = out2+self.layer5_1(stage1_out)
        out_23 = torch.cat([stage2_out, stage3_out], dim=1)
        feature = self.layer5(out_23)

        return feature

    def f_v_concate(self, feature, vec_pos, feature_size):
        fea_pos = vec_pos.expand(-1, -1, feature_size[0], feature_size[1])  # tile for cat
        exit_feat_in = torch.cat([feature, fea_pos], dim=1)

        return exit_feat_in

    def Segmentation(self, feature, history_mask):
        feature_size = feature.shape[-2:]

        history_mask = F.interpolate(history_mask, feature_size, mode='bilinear', align_corners=True)
        out = feature
        out_plus_history = torch.cat([out, history_mask], dim=1)
        out = out + self.residule1(out_plus_history)
        out = out + self.residule2(out)
        out = out + self.residule3(out)

        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer9(out)

        out_softmax = F.softmax(out, dim=1)

        return out, out_softmax
    
    def Weighted_GAP(self,supp_feat, mask):
        supp_feat = supp_feat * mask
        feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
        area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
        supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area  
        return supp_feat
    
    def narrow_gap(self,feat,vec):
        vec_c = self.c_conv(vec.squeeze(-1).transpose(-1,-2))
        vec_c = vec_c.transpose(-1,-2).unsqueeze(-1)
        vec_c = F.sigmoid(vec_c)
       
        b,c,h,w = feat.shape
        vec_s = vec.view(b,c)
        vec_s = self.fc_2(F.relu(self.fc_1(vec_s)))
        
        vec_c = vec_c.view(b,c,1,1)
        vec_s = vec_s.view(b,c,1,1)
        meta_query = (feat*vec_s.expand_as(feat)).sum(dim=1,keepdim=True)
        meta_query = meta_query.view(b,1,-1)
        meta_query = F.softmax(meta_query,dim=2)
        meta_query = meta_query.view(b,1,h,w)
        query_s = feat*meta_query+feat
        
        query_c = feat*vec_c.expand_as(feat)
        return query_s + query_c
        
    def focal_loss(self,x, p = 1, c = 0.1):
        return torch.pow(1 - x, p) * torch.log(c + x)

    def get_loss(self, logits, query_label, idx):
        bce_logits_func = nn.CrossEntropyLoss()
        outB, outA_pos, vec, outB_side,label_predict = logits

        b, c, w, h = query_label.size()
        outB_side = F.upsample(outB_side, size=(w, h), mode='bilinear')

        # add
        query_label = query_label.view(b, -1)
        bb, cc, _, _ = outB_side.size()
        outB_side = outB_side.view(b, cc, w * h)
        #

        loss_bce_seg = bce_logits_func(outB_side, query_label.long())
        loss_class = bce_logits_func(label_predict,idx)
        # loss_class1 = bce_logits_func(label_predict1,idx)
        # loss_class = (loss_class+loss_class1)*0.1
        loss = loss_bce_seg+loss_class*0.5

        return loss, loss_bce_seg, loss_class*0.5

    def get_pred(self, logits, query_image):
        outB, outA_pos, outB_side1, outB_side,label_predict= logits
        w, h = query_image.size()[-2:]
        outB_side = F.upsample(outB_side, size=(w, h), mode='bilinear')
        out_softmax = F.softmax(outB_side, dim=1)
        values, pred = torch.max(out_softmax, dim=1)
        return out_softmax, pred
