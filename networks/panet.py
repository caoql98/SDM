import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone import resnet_dialated as resnet
from models import ASPP
from models.PMMs import PMMs
from models.backbone import vgg
# The Code of baseline network is referenced from https://github.com/icoz69/CaNet
# The code of training & testing is referenced from https://github.com/xiaomengyc/SG-One

class OneModel(nn.Module):
    def __init__(self, args):

        self.inplanes = 64
        self.num_pro = 3
        super(OneModel, self).__init__()
        # self.model_backbone = vgg.vgg16(pretrained=True)
        self.model_res = resnet.Res50_Deeplab(pretrained=True)
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=1536, out_channels=256, kernel_size=3, stride=1, padding=2, dilation=2, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU())

        self.batch_size = args.batch_size

    def forward(self, query_rgb, support_rgb, support_mask,query_mask):
        # extract support_ feature
        support_feature = self.extract_feature_res(support_rgb)
        
        # support_feature = self.layer5(support_feature)
        # extract query feature
        query_feature = self.extract_feature_res(query_rgb)
        # query_feature = self.layer5(query_feature)
        
        f_prototypes = self.Weighted_GAP(support_feature, support_mask)
        b_prototypes = self.Weighted_GAP(support_feature, 1-support_mask)
        
       
        bf,cf,hf,wf = f_prototypes.shape
        bsize, ch_sz, sp_sz, _ = query_feature.shape
        cosine_eps = 1e-7  
        tmp_query = query_feature.contiguous().view(bsize, ch_sz, -1)
        tmp_query_norm = torch.norm(tmp_query, 2, 1, True) 
        
        tmp_supp = f_prototypes
        tmp_supp = tmp_supp.contiguous().view(bsize, ch_sz, -1)
        tmp_supp = tmp_supp.contiguous().permute(0, 2, 1)
        tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)
        # print(tmp_query_norm.shape)
        similarity = torch.bmm(tmp_supp, tmp_query)/(torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
        similarity = -20*similarity
        # similarity = (similarity - similarity.min(1)[0].unsqueeze(1))/(similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
        corr_query_f = similarity.view(bsize, 1, sp_sz, sp_sz)
 
        
        tmp_supp = b_prototypes
        tmp_supp = tmp_supp.contiguous().view(bsize, ch_sz, -1)
        tmp_supp = tmp_supp.contiguous().permute(0, 2, 1)
        tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)

        similarity = torch.bmm(tmp_supp, tmp_query)/(torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
        similarity = -20*similarity
        # similarity = (similarity - similarity.min(1)[0].unsqueeze(1))/(similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
        corr_query_b = similarity.view(bsize, 1, sp_sz, sp_sz)
        # segmentation

        probability_query = torch.cat([corr_query_b,corr_query_f],dim=1)
        
        # print(probability_query[0,:,0,:])
        
        out_softmax = F.softmax(probability_query, dim=1)
        values, pred_query = torch.max(out_softmax, dim=1)
        
        pred_query1 = pred_query.view(bsize, 1, sp_sz, sp_sz).float()
        fq_prototypes = self.Weighted_GAP(query_feature, pred_query1)
        bq_prototypes = self.Weighted_GAP(query_feature, 1-pred_query1)
        
        bf,cf,hf,wf = fq_prototypes.shape
        bsize, ch_sz, sp_sz, _ = support_feature.shape
        cosine_eps = 1e-7  
        tmp_query = support_feature.contiguous().view(bsize, ch_sz, -1)
        tmp_query_norm = torch.norm(tmp_query, 2, 1, True) 
        
        tmp_supp = fq_prototypes
        tmp_supp = tmp_supp.contiguous().view(bsize, ch_sz, -1)
        tmp_supp = tmp_supp.contiguous().permute(0, 2, 1)
        tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)

        similarity = torch.bmm(tmp_supp, tmp_query) /(torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
        similarity = -20*similarity
        # similarity = (similarity - similarity.min(1)[0].unsqueeze(1))/(similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
        corr_query_fq = similarity.view(bsize, 1, sp_sz, sp_sz)
        
        
        tmp_supp = bq_prototypes
        tmp_supp = tmp_supp.contiguous().view(bsize, ch_sz, -1)
        tmp_supp = tmp_supp.contiguous().permute(0, 2, 1)
        tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)

        similarity = torch.bmm(tmp_supp, tmp_query)/(torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
        similarity = -20*similarity
        # similarity = (similarity - similarity.min(1)[0].unsqueeze(1))/(similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
        corr_query_bq = similarity.view(bsize, 1, sp_sz, sp_sz)
        # segmentation
        probability_support = torch.cat([corr_query_bq,corr_query_fq],dim=1)
        
        return support_feature, query_feature, probability_query, probability_support

    def test_forward(self, query_rgb, support_rgb, support_mask,query_mask):
        # extract support_ feature
        support_feature = self.extract_feature_res(support_rgb)
        
        # support_feature = self.layer5(support_feature)
        # extract query feature
        query_feature = self.extract_feature_res(query_rgb)
        # query_feature = self.layer5(query_feature)
        
        f_prototypes = self.Weighted_GAP(support_feature, support_mask)
        b_prototypes = self.Weighted_GAP(support_feature, 1-support_mask)
        
       
        bf,cf,hf,wf = f_prototypes.shape
        bsize, ch_sz, sp_sz, _ = query_feature.shape
        cosine_eps = 1e-7  
        tmp_query = query_feature.contiguous().view(bsize, ch_sz, -1)
        tmp_query_norm = torch.norm(tmp_query, 2, 1, True) 
        
        tmp_supp = f_prototypes
        tmp_supp = tmp_supp.contiguous().view(bsize, ch_sz, -1)
        tmp_supp = tmp_supp.contiguous().permute(0, 2, 1)
        tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)
        # print(tmp_query_norm.shape)
        similarity = torch.bmm(tmp_supp, tmp_query)/(torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
        similarity = -20*similarity
        # similarity = (similarity - similarity.min(1)[0].unsqueeze(1))/(similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
        corr_query_f = similarity.view(bsize, 1, sp_sz, sp_sz)
 
        
        tmp_supp = b_prototypes
        tmp_supp = tmp_supp.contiguous().view(bsize, ch_sz, -1)
        tmp_supp = tmp_supp.contiguous().permute(0, 2, 1)
        tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)

        similarity = torch.bmm(tmp_supp, tmp_query)/(torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
        similarity = -20*similarity
        # similarity = (similarity - similarity.min(1)[0].unsqueeze(1))/(similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
        corr_query_b = similarity.view(bsize, 1, sp_sz, sp_sz)
        # segmentation

        probability_query = torch.cat([corr_query_b,corr_query_f],dim=1)
        
        return support_feature, query_feature, probability_query, probability_query

    def forward_5shot(self, query_rgb, support_rgb, support_mask,query_mask):

        B,K,C,H,W = support_rgb.shape
        support_rgb = support_rgb.view(B*K,C,H,W)
        support_mask = support_mask.view(B*K,1,H,W)
        # extract support_ feature
        support_feature = self.extract_feature_res(support_rgb)
        
        # support_feature = self.layer5(support_feature)
        # extract query feature
        query_feature = self.extract_feature_res(query_rgb)
        # query_feature = self.layer5(query_feature)
        
        f_prototypes = self.Weighted_GAP(support_feature, support_mask)
        b_prototypes = self.Weighted_GAP(support_feature, 1-support_mask)
        
       
        bf,cf,hf,wf = f_prototypes.shape
        bsize, ch_sz, sp_sz, _ = query_feature.shape
        cosine_eps = 1e-7  
        tmp_query = query_feature.contiguous().view(bsize, ch_sz, -1)
        tmp_query_norm = torch.norm(tmp_query, 2, 1, True) 
        
        tmp_supp = f_prototypes
        s_f = tmp_supp.view(B,K,cf,hf,wf)
        # s_f = s_f.mean(dim=1,keepdim=True)
        # s_f = s_f.view(B,cf,hf,wf)
        # tmp_supp = s_f.contiguous().view(bsize, ch_sz, -1)
        # tmp_supp = tmp_supp.contiguous().permute(0, 2, 1)
        # tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)

        # similarity = torch.bmm(tmp_supp, tmp_query) / (torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
        # similarity = -20*similarity
        # corr_query_f = similarity.view(B, 1, sp_sz, sp_sz)
        for i in range(5):
              tmp_supp = s_f[:,i,:,:,:]
              tmp_supp = tmp_supp.contiguous().view(bsize, ch_sz, -1)
              tmp_supp = tmp_supp.contiguous().permute(0, 2, 1)
              tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)

              similarity = torch.bmm(tmp_supp, tmp_query) / (torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
              similarity = -20*similarity
              if i == 0:
                  corr_query_f = similarity.view(B, 1, sp_sz, sp_sz)
              else:
                corr_query_f = torch.cat([corr_query_f, similarity.view(B, 1, sp_sz, sp_sz)],dim=1)
              corr_query_f = corr_query_f.mean(dim=1,keepdim=True)
 
        
        tmp_supp = b_prototypes
        s_b = tmp_supp.view(B,K,cf,hf,wf)
        # s_b = s_b.mean(dim=1,keepdim=True)
        # s_b = s_b.view(B,cf,hf,wf)
        # tmp_supp = s_b.contiguous().view(bsize, ch_sz, -1)
        # tmp_supp = tmp_supp.contiguous().permute(0, 2, 1)
        # tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)

        # similarity = torch.bmm(tmp_supp, tmp_query) / (torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
        # similarity = -20*similarity
        # corr_query_b = similarity.view(B, 1, sp_sz, sp_sz)
        
        for i in range(5):
              tmp_supp = s_b[:,i,:,:,:]
              tmp_supp = tmp_supp.contiguous().view(bsize, ch_sz, -1)
              tmp_supp = tmp_supp.contiguous().permute(0, 2, 1)
              tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)

              similarity = torch.bmm(tmp_supp, tmp_query) / (torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
              similarity = -20*similarity
              if i == 0:
                  corr_query_b = similarity.view(B, 1, sp_sz, sp_sz)
              else:
                corr_query_b = torch.cat([corr_query_b, similarity.view(B, 1, sp_sz, sp_sz)],dim=1)
              corr_query_b = corr_query_b.mean(dim=1,keepdim=True)
        # segmentation

        probability_query = torch.cat([corr_query_b,corr_query_f],dim=1)
        
        # print(probability_query[0,:,0,:])
    
        
        return support_feature, query_feature, probability_query, probability_query

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

    def get_loss(self, logits, query_label, support_label, idx):
        bce_logits_func = nn.CrossEntropyLoss()
        outB, outA_pos, query_side, support_side = logits

        b, c, w, h = query_label.size()
        query_side = F.upsample(query_side, size=(w, h), mode='bilinear')

        query_label = query_label.view(b, -1)
        bb, cc, _, _ = query_side.size()
        query_side = query_side.view(b, cc, w * h)

        loss_query = bce_logits_func(query_side, query_label.long())
        
        
        b, c, w, h = support_label.size()
        support_side = F.upsample(support_side, size=(w, h), mode='bilinear')

        support_label = support_label.view(b, -1)
        bb, cc, _, _ = support_side.size()
        support_side = support_side.view(b, cc, w * h)

        loss_support = bce_logits_func(support_side, support_label.long())


        loss = loss_query+loss_support

        return loss, loss_query, loss_support

    def get_pred(self, logits, query_image):
        outB, outA_pos, outB_side1, outB_side = logits
        w, h = query_image.size()[-2:]
        outB_side1 = F.upsample(outB_side1, size=(w, h), mode='bilinear')
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