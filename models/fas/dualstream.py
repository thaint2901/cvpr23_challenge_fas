import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import SelectAdaptivePool2d

from models.backbones import encoders



class CrossAttensionFusion2D(torch.nn.Module):
    def __init__(self, embed_size = 384, hidden_size = 512, n_head = 32, num_classes=2):
        """
            embed_size: channel of previous layer
            hidden_size: channel of output feature map
            n_head: number of attention head
        """
        assert(embed_size % n_head == 0), 'The size of head should be divided by the number of channels.'
        super().__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.n_head = 32
        self.attn_size = self.embed_size // self.n_head
        self.q = torch.nn.Conv2d(embed_size, embed_size, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(embed_size, embed_size, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(embed_size, embed_size, kernel_size=1, stride=1, padding=0)
        self.q_bpf = torch.nn.Conv2d(embed_size, embed_size, kernel_size=1, stride=1, padding=0)
        self.k_bpf = torch.nn.Conv2d(embed_size, embed_size, kernel_size=1, stride=1, padding=0)
        self.v_bpf = torch.nn.Conv2d(embed_size, embed_size, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv2d(2*embed_size, hidden_size, kernel_size=1, stride=1, padding=0)

        # head
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.flatten = nn.Flatten(1)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x, x_bpf):
        B, C, H, W = x.shape
        scale = int(self.attn_size)**(-0.5)

        # Normal branch
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        # BPF branch
        q_bpf = self.q_bpf(x_bpf)
        k_bpf = self.k_bpf(x_bpf)
        v_bpf = self.v_bpf(x_bpf)

        # Compute attention for normal branch
        q_bpf.mul_(scale)
        q_bpf = q_bpf.reshape((B, self.n_head, self.attn_size, H*W))
        q_bpf = q_bpf.permute(0, 3, 1, 2) # b, hw, head, att
        k     = k.reshape((B, self.n_head, self.attn_size, H*W))
        k     = k.permute(0, 3, 1, 2) # b, hw, head, att
        v     = v.reshape((B, self.n_head, self.attn_size, H*W))
        v     = v.permute(0, 3, 1, 2) # b, hw, head, att

        q_bpf = q_bpf.transpose(1, 2) # b, head, hw, att
        v     = v.transpose(1, 2) # b, head, hw, att
        k     = k.transpose(1, 2).transpose(2,3) # b, head, att, hw

        w = torch.matmul(q_bpf, k) # b, head, hw, hw
        w = F.softmax(w, dim=3)    # b, head, hw, hw
        f = w.matmul(v)            # b, head, hw, att
        f = f.transpose(1, 2).contiguous() # b, hw, head, att
        f = f.view(B, H, W, -1) # b, h, w, head*att
        f = f.permute(0, 3, 1, 2) # b, head*att, h, w
        f = f + x

        # Compute attention for bpf branch
        q.mul_(scale)
        q     = q.reshape((B, self.n_head, self.attn_size, H*W))
        q     = q.permute(0, 3, 1, 2) # b, hw, head, att
        k_bpf = k_bpf.reshape((B, self.n_head, self.attn_size, H*W))
        k_bpf = k_bpf.permute(0, 3, 1, 2) # b, hw, head, att
        v_bpf = v_bpf.reshape((B, self.n_head, self.attn_size, H*W))
        v_bpf = v_bpf.permute(0, 3, 1, 2) # b, hw, head, att

        q     = q.transpose(1, 2) # b, head, hw, att
        v_bpf = v_bpf.transpose(1, 2) # b, head, hw, att
        k_bpf = k_bpf.transpose(1, 2).transpose(2,3) # b, head, attn, hw

        w_bpf = torch.matmul(q, k_bpf) 
        w_bpf = F.softmax(w_bpf, dim=3)  
        f_bpf = w_bpf.matmul(v_bpf)          
        f_bpf = f_bpf.transpose(1, 2).contiguous() 
        f_bpf = f_bpf.view(B, H, W, -1) 
        f_bpf = f_bpf.permute(0, 3, 1, 2)
        f_bpf = f_bpf + x_bpf

        # Concat
        fused = torch.cat([f, f_bpf], 1)
        out = self.proj_out(fused)

        # head
        out = self.global_pool(out)
        out = self.flatten(out)
        out = self.classifier(out)

        return out
    

class DualStream(nn.Module):
    def __init__(self, encoder, feature_dim, test_cfg=None, train_cfg=None) -> None:
        super(DualStream, self).__init__()
        assert isinstance(encoder, dict)
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.return_label = self.test_cfg.pop('return_label', True)
        self.return_feature = self.test_cfg.pop('return_feature', False)
        self.encoder = encoders(encoder)
        self.encoder_bpf = encoders(encoder)

        num_classes = encoder.get("num_classes", 2)
        self.head = CrossAttensionFusion2D(embed_size=feature_dim, hidden_size=feature_dim, num_classes=num_classes)

        self.cls_loss = nn.CrossEntropyLoss()

    
    def _get_losses(self, feat, feat_bpf, feat_fused, label):
        """calculate training losses"""
        loss_cls = self.cls_loss(feat, label[:, 0]).unsqueeze(0) * self.train_cfg['w_cls']
        loss_bpf = self.cls_loss(feat_bpf, label[:, 0]).unsqueeze(0) * self.train_cfg['w_bpf']
        loss_fused = self.cls_loss(feat_fused, label[:, 0]).unsqueeze(0) * self.train_cfg['w_fused']
        loss = loss_cls + loss_bpf + loss_fused
        return dict(loss_cls=loss_cls, loss_bpf=loss_bpf, loss_fused=loss_fused, loss=loss)
    

    def forward(self, img, label=None, domain=None):
        """forward"""
        if self.training:
            img, img_bpf = img
            feat = self.encoder.forward_features(img)
            feat_bpf = self.encoder_bpf.forward_features(img_bpf)

            feat_fused = self.head(feat, feat_bpf)

            feat = self.encoder.forward_head(feat)
            feat_bpf = self.encoder_bpf.forward_head(feat_bpf)
            
            return self._get_losses(feat, feat_bpf, feat_fused, label)
        else:
            feat = self.encoder(img)
            pred = F.softmax(feat, dim=1)[:, 0]
            output = [pred]
            if self.return_label:
                output.append(label)
            if self.return_feature:
                output.append(feat)
            return output