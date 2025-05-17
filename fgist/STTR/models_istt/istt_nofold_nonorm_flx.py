# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone_50,ResBlock_nonorm,ResBlock
from .position_encoding import build_position_encoding_ours
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)

from .transformer_nonorm_flx import build_transformer

from torchvision.models import vgg19

    
class ISTT_NOFOLD(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbonec,backbones, position_embedding,transformer, num_classes, num_queries, fold_k,tail_norm,aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        
        self.backbone_content = backbonec
        self.backbone_style = backbones
        self.position_embedding=position_embedding
        self.aux_loss = aux_loss
        
        self.input_proj_c = nn.Conv2d(self.backbone_content.num_channels, hidden_dim, kernel_size=1)
        self.input_proj_s = nn.Conv2d(self.backbone_style.num_channels, hidden_dim, kernel_size=1)
        self.output_proj = nn.Conv2d(hidden_dim, self.backbone_content.num_channels, kernel_size=1)
        
        
        tail_layers = []
        res_block=ResBlock if tail_norm else ResBlock_nonorm
        for ri in range(self.backbone_content.reduce_times):
            times=2**ri
            content_c=self.backbone_content.num_channels
            out_c=3 if ri==self.backbone_content.reduce_times-1 else int(content_c/(times*2))
            tail_layers.extend([
                res_block(int(content_c/times), int(content_c/(times*2))),
                nn.Upsample(scale_factor = 2, mode='bilinear'),
                nn.ReflectionPad2d(1),
                nn.Conv2d(int(content_c/times),out_c,
                          kernel_size=3, stride=1, padding=0),
            ])
        self.tail = nn.Sequential(*tail_layers)
        
        
        
    
    def forward(self, samples: NestedTensor,style_images: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)  
            style_images = nested_tensor_from_tensor_list(style_images)
            
        B,C,out_h,out_w=samples.tensors.shape  
        
        src_features = self.backbone_content(samples)  # feature: [N,B,2048,H/32,W/32] ;  pos: [N,B,256,H/32,W/32] 
        style_features = self.backbone_style(style_images)  # feature: [N,B,2048,H/32,W/32] ;  pos: [N,B,256,H/32,W/32] 
        
        
        src_features, mask = src_features["0"].decompose()
        style_features, style_mask = style_features["0"].decompose()
        B,C,f_h,f_w=src_features.shape  
        
        
        pos = self.position_embedding(NestedTensor(src_features, mask)).to(src_features.dtype)
        style_pos = self.position_embedding(NestedTensor(style_features, style_mask)).to(style_features.dtype)
        
        assert mask is not None
        
        hs,mem = self.transformer(self.input_proj_s(style_features), style_mask, self.input_proj_c(src_features),pos,style_pos) # hs: [6, 2, 100, 
    
        
        B,h_w,C=hs[-1].shape        #[B, h*w=L, C]
        hs = hs[-1].permute(0,2,1).reshape(B,C,f_h,f_w)    # [B,C,h,w]

        res = self.output_proj(hs)   # [B,256*k*k,h*w=L]   L=[(H − k + 2P )/S+1] * [(W − k + 2P )/S+1]  k=16,P=2,S=32

        
        res = self.tail(res)# [B,3,H,W] 
        
        return res
    

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

class VGGEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(pretrained=True).features
        # print(vgg.get_layer('block4_conv2'))
        # print('-----')
        # print(vgg[: 2])
        # print('-----')
        # print(vgg[2: 7])
        # print('-----')
        # print(vgg[7: 12])
        # print('-----')
        # print(vgg[12: 21])
        # exit()
        self.block4_conv2 = vgg[:23]
        
        self.block1_conv1 = vgg[:2]
        self.block2_conv1 = vgg[2:7]
        self.block3_conv1 = vgg[7:12]
        self.block4_conv1 = vgg[12:21]
        self.block5_conv1 = vgg[21:30]
        
        self.slice1 = vgg[: 2]
        self.slice2 = vgg[2: 7]
        self.slice3 = vgg[7: 12]
        self.slice4 = vgg[12: 21]
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, images, output_last_feature=False, gatys=False):
        if gatys:
            if output_last_feature:
                return self.block4_conv2(images)
            else:
                h1 = self.block1_conv1(images)
                h2 = self.block2_conv1(h1)
                h3 = self.block3_conv1(h2)
                h4 = self.block4_conv1(h3)
                h5 = self.block5_conv1(h4)

                return [h1, h2, h3, h4, h5]
        else:
            h1 = self.slice1(images)
            h2 = self.slice2(h1)
            h3 = self.slice3(h2)
            h4 = self.slice4(h3)
            if output_last_feature:
                return h4
            else:
                return [h1, h2, h3, h4]

class SetCriterion(nn.Module):

    def __init__(self, weight_dict, distill_loss_type='MSE'):

        super().__init__()
        self.vgg_encoder = VGGEncoder()
   
        self.weight_dict = weight_dict

        if distill_loss_type == 'MSE':
            self.distill_loss = nn.MSELoss(reduction='mean')
        if distill_loss_type == 'MAE':
            self.distill_loss = nn.L1Loss(reduction='mean')
        if distill_loss_type == 'Gatys':
            self.distill_loss = 'Gatys'
        if distill_loss_type == 'Gatys_2':
            self.distill_loss = 'Gatys_2'
        if distill_loss_type == 'author':
            self.distill_loss = 'author'


    
    def calc_mean_std(self,features):
        batch_size, c = features.size()[:2]
        features_mean = features.reshape(batch_size, c, -1).mean(dim=2).reshape(batch_size, c, 1, 1)
        features_std = features.reshape(batch_size, c, -1).std(dim=2).reshape(batch_size, c, 1, 1) + 1e-6
        return features_mean, features_std

    def loss_content(self,out_features, t):
        
        loss=0
        for out_i,target_i in zip(out_features, t):
#             print("out_i.shape,target_i.shape:",out_i.shape,target_i.shape)
            loss+=F.mse_loss(out_i,target_i)
        return loss
    
    def loss_content_last_distill(self, out_features, t):
        return self.distill_loss(out_features, t)
    def loss_content_last(self,out_features, t):
        return F.mse_loss(out_features, t)
        
    def gram_matrix(self,input):
        a, b, c, d = input.size()
        # print(a,b,c,d)
        features = input.view(a, b, c * d)  

        G = torch.matmul(features, features.permute(0, 2, 1)) 
        return G
    
    
    
    def tv_loss(self,img):
        N,C,H,W = img.shape
        x1 = img[:,:,0:H-1,:]
        x2 = img[:,:,1:H,:]
        y1 = img[:,:,:,0:W-1]
        y2 = img[:,:,:,1:W]
        loss = ((x2-x1).pow(2).sum() + (y2-y1).pow(2).sum()) 
        return loss

    
    
    def loss_style_gram(self,output_middle_features, style_middle_features):
        target_gram = self.gram_matrix(style_middle_features)
        output_gram = self.gram_matrix(output_middle_features)
        return F.mse_loss(output_gram, target_gram)
    
    def loss_style_gram_multiple(self,content_middle_features, style_middle_features):
        loss = 0
#         print("content_middle_features.shape, style_middle_features.shape:",content_middle_features.shape, style_middle_features.shape)
        for c, s in zip(content_middle_features, style_middle_features):
            target_gram = self.gram_matrix(c)
            output_gram = self.gram_matrix(s)
            loss += (0.2 * (1. / (4. * (target_gram.shape[1] ** 2) * ((c.shape[2] * c.shape[3]) ** 2))) * ((output_gram - target_gram) ** 2).sum(dim=(1,2))).mean()
        return loss
    
    def loss_style_adain(self,content_middle_features, style_middle_features):
        loss = 0
#         print("content_middle_features.shape, style_middle_features.shape:",content_middle_features.shape, style_middle_features.shape)
        for c, s in zip(content_middle_features, style_middle_features):
#             print("c.shape,s.shape:",c.shape,s.shape)
            c_mean, c_std = self.calc_mean_std(c)
            s_mean, s_std = self.calc_mean_std(s)
            loss += F.mse_loss(c_mean, s_mean) + F.mse_loss(c_std, s_std)
        return loss
    
    def forward(self, outputs, targets_content, contents, styles): #_hybrid
        
        if self.distill_loss == 'Gatys':
            content_features = self.vgg_encoder(targets_content.tensors, output_last_feature=True)
            output_features = self.vgg_encoder(outputs, output_last_feature=True)
            loss_c = self.loss_content_last(output_features, content_features)

            style_middle_features = self.vgg_encoder(targets_content.tensors, output_last_feature=False)
            output_middle_features = self.vgg_encoder(outputs, output_last_feature=False)
            loss_s = self.loss_style_adain(output_middle_features, style_middle_features)

            loss_tv = self.tv_loss(outputs)

            losses = {
                'loss_content':loss_c,
                'loss_style':loss_s,
                'loss_tv':loss_tv
            }
        
        elif self.distill_loss == 'Gatys_2':
            content_features = self.vgg_encoder(targets_content.tensors, output_last_feature=True, gatys=True)
            output_features = self.vgg_encoder(outputs, output_last_feature=True, gatys=True)
            # loss_c = self.loss_content_last(output_features, content_features)
            # print(content_features.shape, output_features.shape)
            loss_c = (0.5 * ((output_features - content_features) ** 2).sum(dim=(1,2,3))).mean()
            # exit()

            style_middle_features = self.vgg_encoder(targets_content.tensors, output_last_feature=False, gatys=True)
            output_middle_features = self.vgg_encoder(outputs, output_last_feature=False, gatys=True)
            loss_s = self.loss_style_gram_multiple(output_middle_features, style_middle_features)
            
            # loss_s = self.loss_style_adain(output_middle_features, style_middle_features)

            loss_tv = self.tv_loss(outputs)

            losses = {
                'loss_content':loss_c,
                'loss_style':loss_s,
                'loss_tv':loss_tv
            }
        elif self.distill_loss == 'author':
            content_features = self.vgg_encoder(contents.tensors, output_last_feature=True)
            output_features = self.vgg_encoder(outputs, output_last_feature=True)
            loss_c = self.loss_content_last(output_features, content_features)

            style_middle_features = self.vgg_encoder(styles.tensors, output_last_feature=False)
            output_middle_features = self.vgg_encoder(outputs, output_last_feature=False)
            loss_s = self.loss_style_adain(output_middle_features, style_middle_features)

            loss_tv = self.tv_loss(outputs)

            losses = {
                'loss_content':loss_c,
                'loss_style':loss_s,
                'loss_tv':loss_tv
            }
        else:
            loss_c = self.loss_content_last_distill(outputs, targets_content.tensors)
            losses = {
                'loss_content':loss_c,
                # 'loss_style':loss_s,
                # 'loss_tv':loss_tv
        
            }

        return losses



class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = 20 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        # for panoptic, we just add a num_classes that is large enough to hold
        # max_obj_id + 1, but the exact value doesn't really matter
        num_classes = 250
    device = torch.device(args.device)

    backbonec = build_backbone_50(args, args.cbackbone_layer)
    backbones = build_backbone_50(args, args.sbackbone_layer)

    position_embedding = build_position_encoding_ours(args)
    
    transformer = build_transformer(args)

    model = ISTT_NOFOLD(
        backbonec,
        backbones,
        position_embedding,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        fold_k=args.fold_k,
        tail_norm=args.tnorm,
        aux_loss=args.aux_loss,
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    
    weight_dict = {'loss_content': args.content_loss_coef, 'loss_style': args.style_loss_coef, 'loss_tv':args.tv_loss_coef}
    
    
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
        # print(weight_dict)

#     losses = ['labels', 'boxes', 'cardinality']
    
    losses = ['content']
    if args.masks:
        losses += ["masks"]
    
    criterion = SetCriterion(weight_dict=weight_dict,
                             distill_loss_type=args.distill_loss_type)
    criterion.to(device)
    
   

    return model, criterion, None
