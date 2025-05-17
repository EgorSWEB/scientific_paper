class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)







from torchvision.models._utils import IntermediateLayerGetter

device = torch.device(args.device)

backbonec = build_backbone_50(args, args.cbackbone_layer)
backbones = build_backbone_50(args, args.sbackbone_layer)

# - build_backbone_50
train_backbone = args.lr_backbone > 0
return_interm_layers = args.masks # False
backbone = Backbone_50(args.backbone, backbone_layer, train_backbone, return_interm_layers, args.dilation) # dilation - False


# -- Backbone_50
class Backbone_50(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str, backbone_layer, # backbone_layer - resnet50
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(torchvision.models, name)(
            pretrained=True, norm_layer=FrozenBatchNorm2d) # is_main_process() - True
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048 # 2048, так как используется resnet50
        super().__init__(backbone, backbone_layer, train_backbone, num_channels, return_interm_layers)

class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, backbone_layer,train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
#             return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            return_layer_idxs=list(range(1,backbone_layer+1))
            return_layers = {}
            for i,rly in enumerate(return_layer_idxs):
                return_layers["layer{}".format(rly)]=str(i)
        else:
#             return_layers = {'layer4': "0"}
            return_layers = {"layer{}".format(backbone_layer): "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        
        layer2dim_dict={
            1: 256, 2: 512, 3: 1024, 4: 2048
        }
        layer2reduce_dict={ # 2**
            1: 2, 
            2: 3, 
            3: 4, 
            4: 5
        }
        self.num_channels =layer2dim_dict[backbone_layer]
        self.reduce_times=layer2reduce_dict[backbone_layer]
        

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class FrozenBatchNorm2d(torch.nn.Module):

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias






position_embedding = build_position_encoding_ours(args)

def build_position_encoding_ours(args):
    N_steps = args.hidden_dim // 2
    if args.position_embedding in ('v2', 'sine'): # это
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding



class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None): # normalize - true, scale=None
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale # 2 * math.pi

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask

        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
       
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        
        return pos











transformer = build_transformer(args)

def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim, # 256
        dropout=args.dropout,
        nhead=args.nheads, # 8
        dim_feedforward=args.dim_feedforward, # 2048
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm, # False
        return_intermediate_dec=True,
        enorm=args.enorm,
        dnorm=args.dnorm,
    )



class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False,enorm=False,dnorm=False):
        super().__init__()

        
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before,enorm)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before and enorm else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before,dnorm)
        decoder_norm = nn.LayerNorm(d_model) if dnorm else None
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, style_src, mask, src, query_pos_embed, pos_embed):

        src = src.flatten(2).permute(2, 0, 1)

        if len(style_src.shape)==4:
            bs, c, h, w = style_src.shape

        style_src = style_src.flatten(2).permute(2, 0, 1)
        
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1) if pos_embed is not None else None#[H*W,B,C]
        query_pos_embed = query_pos_embed.flatten(2).permute(2, 0, 1)  if query_pos_embed is not None else None  #[H*W,B,C]
        mask = mask.flatten(1)

        tgt = src
        memory = self.encoder(style_src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_pos_embed)
        if len(src.shape)==4:
            return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)
        else:
            return hs.transpose(1, 2), memory.permute(1, 2, 0)
            
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,enorm=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.enorm = enorm
        if enorm:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]

        src = src + self.dropout1(src2)
        if self.enorm:
            src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        if self.enorm:
            src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        if self.enorm:
            src2 = self.norm1(src)
        else:
            src2 = src
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        if self.enorm:
            src2 = self.norm2(src)
        else:
            src2 = src
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,dnorm=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.dnorm = dnorm # True
        if dnorm:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        if self.dnorm:
            tgt = self.norm1(tgt)
        tgt2_ = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt2=tgt2_[0]
        tgt2_att=tgt2_[1]
        
#         print("tgt2.shape",tgt2.shape)
        tgt = tgt + self.dropout2(tgt2)
        if self.dnorm:
            tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        if self.dnorm:
            tgt = self.norm3(tgt)
        return tgt,tgt2_att

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        if self.dnorm:
            tgt2 = self.norm1(tgt)
        else:
            tgt2 = tgt
        
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        if self.dnorm:
            tgt2 = self.norm2(tgt)
        else:
            tgt2 = tgt
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        if self.dnorm:
            tgt2 = self.norm3(tgt)
        else:
            tgt2 = tgt
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for li,layer in enumerate(self.layers):
            output,att = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                if self.norm is not None:
                    intermediate.append(self.norm(output))
                else:
                    intermediate.append(output)
            
#             show_feature_map(att,verbose="tgt2_att_{}".format(li),out_root="tmp_out",norm=True)
#             show_feature_map(output.transpose(0,1)[:,:,[0]].reshape,verbose="tgt2_output_{}".format(li),out_root="tmp_out")
            
#             assert(False)
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)





model = ISTT_NOFOLD(
    backbonec,
    backbones,
    position_embedding,
    transformer,
    num_classes=num_classes,
    num_queries=args.num_queries,
    fold_k=args.fold_k,
    tail_norm=args.tnorm, # True
    aux_loss=args.aux_loss,
)


class ResBlock(nn.Module):

    def __init__(self, outer_dim, inner_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(outer_dim),
            nn.LeakyReLU(),
            nn.Conv2d(outer_dim, inner_dim, 1),
            nn.BatchNorm2d(inner_dim),
            nn.LeakyReLU(),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.BatchNorm2d(inner_dim),
            nn.LeakyReLU(),
            nn.Conv2d(inner_dim, outer_dim, 1),
        )

    def forward(self, input):
        return input + self.net(input)


class ISTT_NOFOLD(nn.Module):
    def __init__(self, backbonec, backbones, position_embedding, transformer, num_classes, num_queries, fold_k,tail_norm,aux_loss=False):

        super().__init__()
        # self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        
        self.backbone_content = backbonec
        self.backbone_style = backbones
        self.position_embedding=position_embedding
        
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
        
        hs, mem = self.transformer(self.input_proj_s(style_features), style_mask, self.input_proj_c(src_features),pos,style_pos) # hs: [6, 2, 100, 
    
        
        B,h_w,C=hs[-1].shape        #[B, h*w=L, C]
        hs = hs[-1].permute(0,2,1).reshape(B,C,f_h,f_w)    # [B,C,h,w]

        res = self.output_proj(hs)   # [B,256*k*k,h*w=L]   L=[(H − k + 2P )/S+1] * [(W − k + 2P )/S+1]  k=16,P=2,S=32

        
        res = self.tail(res)# [B,3,H,W] 
        
        return res
    






matcher = build_matcher(args)

weight_dict = {'loss_content': args.content_loss_coef, 'loss_style': args.style_loss_coef, 'loss_tv':args.tv_loss_coef}


#     losses = ['labels', 'boxes', 'cardinality']

losses = ['content']







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
        self.block4_conv2 = vgg[:22]
        
        self.block1_conv1 = vgg[:1]
        self.block2_conv1 = vgg[1:6]
        self.block3_conv1 = vgg[6:11]
        self.block4_conv1 = vgg[11:20]
        self.block5_conv1 = vgg[20:29]
        
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


    
    def calc_mean_std(self,features):
        """

        :param features: shape of features -> [batch_size, c, h, w]
        :return: features_mean, feature_s: shape of mean/std ->[batch_size, c, 1, 1]
        """

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
        features = input.view(a * b, c * d)  

        G = torch.mm(features, features.t())  
        return G.div(a * b * c * d)
    
    
    
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
            loss += F.mse_loss(output_gram, target_gram)
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
    
    def forward(self, outputs, targets_content): #_hybrid
        
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
            loss_c = self.loss_content_last(output_features, content_features)

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
        else:
            loss_c = self.loss_content_last_distill(outputs, targets_content.tensors)
            losses = {
                'loss_content':loss_c,
                # 'loss_style':loss_s,
                # 'loss_tv':loss_tv
        
            }

        return losses




criterion = SetCriterion(weight_dict=weight_dict,
                         distill_loss_type=args.distill_loss_type)
criterion.to(device)

return model, criterion