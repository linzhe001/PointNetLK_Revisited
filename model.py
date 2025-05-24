""" part of source code from PointNetLK (https://github.com/hmgoforth/PointNetLK), modified. """

import numpy as np
import torch

import utils
from feature_extraction import FeatureExtractor


def mlp_layers(nch_input, nch_layers, b_shared=True, bn_momentum=0.1, dropout=0.0):
    """ [B, Cin, N] -> [B, Cout, N] or
        [B, Cin] -> [B, Cout]
    """
    layers = []
    last = nch_input
    for i, outp in enumerate(nch_layers):
        if b_shared:
            weights = torch.nn.Conv1d(last, outp, 1)
        else:
            weights = torch.nn.Linear(last, outp)
        layers.append(weights)
        layers.append(torch.nn.BatchNorm1d(outp, momentum=bn_momentum))
        layers.append(torch.nn.ReLU())
        if b_shared == False and dropout > 0.0:
            layers.append(torch.nn.Dropout(dropout))
        last = outp
    return layers


class MLPNet(torch.nn.Module):
    """ Multi-layer perception.
        [B, Cin, N] -> [B, Cout, N] or
        [B, Cin] -> [B, Cout]
    """
    def __init__(self, nch_input, nch_layers, b_shared=True, bn_momentum=0.1, dropout=0.0):
        super().__init__()
        list_layers = mlp_layers(nch_input, nch_layers, b_shared, bn_momentum, dropout)
        self.layers = torch.nn.Sequential(*list_layers)

    def forward(self, inp):
        out = self.layers(inp)
        return out


def symfn_max(x):
    # [B, K, N] -> [B, K, 1]
    a = torch.nn.functional.max_pool1d(x, x.size(-1))
    return a


class AnalyticalPointNetLK(torch.nn.Module):
    def __init__(self, ptnet, device):
        """
        初始化AnalyticalPointNetLK模型
        
        参数:
            ptnet: 特征提取器实例，必须继承自FeatureExtractor
            device: 计算设备
        """
        super().__init__()
        
        # 确保ptnet是FeatureExtractor的子类
        assert isinstance(ptnet, FeatureExtractor), "特征提取器必须继承自FeatureExtractor"
        
        self.ptnet = ptnet
        self.device = device
        self.inverse = utils.InvMatrix.apply
        self.exp = utils.ExpMap.apply  # [B, 6] -> [B, 4, 4]
        self.transform = utils.transform  # [B, 1, 4, 4] x [B, N, 3] -> [B, N, 3]

        self.step_train = 0
        self.step_test = 0

        # 识别特征提取器类型
        self.extractor_type = self.ptnet.extractor_type

        # results
        self.last_err = None
        self.prev_r = None
        self.g = None  # estimation result
        self.itr = 0

    @staticmethod
    def rsq(r):
        z = torch.zeros_like(r)
        
        return torch.nn.functional.mse_loss(r, z, reduction='sum')

    @staticmethod
    def comp(g, igt):
        """ |g*igt - I| """
        g = g.float()
        igt = igt.float()
        
        assert g.size(0) == igt.size(0)
        assert g.size(1) == igt.size(1) and g.size(1) == 4
        assert g.size(2) == igt.size(2) and g.size(2) == 4
        A = g.matmul(igt)
        I = torch.eye(4).to(A).view(1, 4, 4).expand(A.size(0), 4, 4)
        loss_pose = torch.nn.functional.mse_loss(A, I, reduction='mean') * 16
        
        return loss_pose

    @staticmethod
    def do_forward(net, p0, voxel_coords_p0, p1, voxel_coords_p1, maxiter=10, xtol=1.0e-7, p0_zero_mean=True, p1_zero_mean=True, mode='train', data_type='synthetic', num_random_points=100, use_numerical=False, delta=1.0e-3):
        voxel_coords_diff = None
        if mode != 'test' or data_type == 'synthetic':
            a0 = torch.eye(4).view(1, 4, 4).expand(
                p0.size(0), 4, 4).to(p0)  # [B, 4, 4]
            a1 = torch.eye(4).view(1, 4, 4).expand(
                p1.size(0), 4, 4).to(p1)  # [B, 4, 4]
        else:
            a0 = torch.eye(4).view(1, 4, 4).to(voxel_coords_p0)  # [1, 4, 4]
            a1 = torch.eye(4).view(1, 4, 4).to(voxel_coords_p1)  # [1, 4, 4]

        if p0_zero_mean:
            if data_type == 'synthetic':
                p0_m = p0.mean(dim=1)   # [B, N, 3] -> [B, 3]
                a0[:, 0:3, 3] = p0_m
                q0 = p0 - p0_m.unsqueeze(1)
            else:
                if mode != 'test':
                    p0_m = voxel_coords_p0
                    a0[:, 0:3, 3] = p0_m
                    q0 = p0 - p0_m.unsqueeze(1)
                else:
                    p0_m = voxel_coords_p0.mean(dim=0)
                    a0[:, 0:3, 3] = p0_m   # global frame
                    q0 = p0 - voxel_coords_p0.unsqueeze(1)   # local frame
                    voxel_coords_diff = voxel_coords_p0 - p0_m   
        else:
            q0 = p0

        if p1_zero_mean:
            if data_type == 'synthetic':
                p1_m = p1.mean(dim=1)   # [B, N, 3] -> [B, 3]
                a1[:, 0:3, 3] = -p1_m
                q1 = p1 - p1_m.unsqueeze(1)
            else:
                if mode != 'test':
                    p1_m = voxel_coords_p1
                    a1[:, 0:3, 3] = -p1_m
                    q1 = p1 - p1_m.unsqueeze(1)
                else:
                    p1_m = voxel_coords_p1.mean(dim=0)
                    a1[:, 0:3, 3] = -p1_m   # global frame
                    q1 = p1 - voxel_coords_p1.unsqueeze(1)   # local frame
        else:
            q1 = p1

        r = net(q0, q1, mode, maxiter=maxiter, xtol=xtol, voxel_coords_diff=voxel_coords_diff, data_type=data_type, num_random_points=num_random_points, use_numerical=use_numerical, delta=delta)

        if p0_zero_mean or p1_zero_mean:
            # output' = trans(p0_m) * output * trans(-p1_m)
            #         = [I, p0_m] * [R, t] * [I, -p1_m]
            #           [0,   1 ]   [0, 1]   [0,    1 ]
            est_g = net.g
            if p0_zero_mean:
                est_g = a0.to(est_g).bmm(est_g)
            if p1_zero_mean:
                est_g = est_g.bmm(a1.to(est_g))
            net.g = est_g

        return r

    def forward(self, p0, p1, mode, maxiter=10, xtol=1.0e-7, voxel_coords_diff=None, data_type='synthetic', num_random_points=100, use_numerical=False, delta=1.0e-3):
        if mode != 'test' or data_type == 'synthetic':
            g0 = torch.eye(4).to(p0).view(1, 4, 4).expand(
                p0.size(0), 4, 4).contiguous()
        else:
            g0 = torch.eye(4).to(p0).view(1, 4, 4)
        r, g, itr = self.iclk_new(g0, p0, p1, maxiter, xtol, mode, voxel_coords_diff=voxel_coords_diff, data_type=data_type, num_random_points=num_random_points, use_numerical=use_numerical, delta=delta)

        self.g = g
        self.itr = itr

        return r

    def update(self, g, dx):
        # [B, 4, 4] x [B, 6] -> [B, 4, 4]
        dg = self.exp(dx)
        return dg.matmul(g.float())

    def Cal_Jac(self, Mask_fn, A_fn, Ax_fn, BN_fn, max_idx, num_points, p0, mode, voxel_coords_diff=None, data_type='synthetic', mamba_in=None, mamba_out=None, extra_param_0=None, ssm_data=None, use_numerical=False, delta=1.0e-3):
        """
        调用特征提取器的get_jacobian方法计算雅可比矩阵
        
        参数:
            use_numerical: 是否使用数值方法计算雅可比矩阵
            delta: 数值方法的扰动参数
        """
        if self.extractor_type == "pointnet":
            # 原始PointNet特征提取器
            return self.ptnet.get_jacobian(
                p0=p0, 
                mask_fn=Mask_fn, 
                a_fn=A_fn, 
                ax_fn=Ax_fn, 
                bn_fn=BN_fn, 
                max_idx=max_idx, 
                mode=mode,
                voxel_coords_diff=voxel_coords_diff,
                data_type=data_type,
                num_points=num_points,
                use_numerical=use_numerical,
                delta=delta
            )
        elif self.extractor_type == "3dmamba_v1":
            # Mamba3D特征提取器需要额外的参数
            return self.ptnet.get_jacobian(
                p0=p0, 
                mask_fn=Mask_fn, 
                a_fn=A_fn, 
                ax_fn=Ax_fn, 
                bn_fn=BN_fn, 
                max_idx=max_idx, 
                mamba_in=mamba_in,
                mamba_out=mamba_out,
                mode=mode,
                voxel_coords_diff=voxel_coords_diff,
                data_type=data_type,
                num_points=num_points,
                use_numerical=use_numerical,
                delta=delta
            )
        elif self.extractor_type == "3dmamba_v2":
            # Mamba3D V2特征提取器需要额外的参数
            return self.ptnet.get_jacobian(
                p0=p0, 
                mask_fn=Mask_fn, 
                a_fn=A_fn, 
                ax_fn=Ax_fn, 
                bn_fn=BN_fn, 
                max_idx=max_idx, 
                bissm_in=mamba_in,
                bissm_out=mamba_out,
                mode=mode,
                voxel_coords_diff=voxel_coords_diff,
                data_type=data_type,
                num_points=num_points,
                use_numerical=use_numerical,
                delta=delta
            )
        elif self.extractor_type == "pointnet_attention_v1":
            # PointNet注意力特征提取器需要额外的参数
            return self.ptnet.get_jacobian(
                p0=p0, 
                mask_fn=Mask_fn, 
                a_fn=A_fn, 
                ax_fn=Ax_fn, 
                bn_fn=BN_fn, 
                max_idx=max_idx, 
                mode=mode,
                voxel_coords_diff=voxel_coords_diff,
                data_type=data_type,
                num_points=num_points,
                extra_param_0=extra_param_0,
                use_numerical=use_numerical,
                delta=delta
            )
        elif self.extractor_type == "fastpointtransformer_v1":
            # FastPointTransformer特征提取器
            return self.ptnet.get_jacobian(
                p0=p0, 
                mask_fn=Mask_fn, 
                a_fn=A_fn, 
                ax_fn=Ax_fn, 
                bn_fn=BN_fn, 
                max_idx=max_idx, 
                mode=mode,
                voxel_coords_diff=voxel_coords_diff,
                data_type=data_type,
                num_points=num_points,
                use_numerical=use_numerical,
                delta=delta
            )
        elif self.extractor_type == "fastpointtransformer_v2":
            # FastPointTransformer V2特征提取器
            return self.ptnet.get_jacobian(
                p0=p0, 
                mask_fn=Mask_fn, 
                a_fn=A_fn, 
                ax_fn=Ax_fn, 
                bn_fn=BN_fn, 
                max_idx=max_idx, 
                mode=mode,
                voxel_coords_diff=voxel_coords_diff,
                data_type=data_type,
                num_points=num_points,
                extra_param_0=extra_param_0,
                use_numerical=use_numerical,
                delta=delta
            )
        elif self.extractor_type == "swinattention_v1":
            # Swin Attention V1特征提取器
            return self.ptnet.get_jacobian(
                p0=p0, 
                mask_fn=Mask_fn, 
                a_fn=A_fn, 
                ax_fn=Ax_fn, 
                bn_fn=BN_fn, 
                max_idx=max_idx, 
                mode=mode,
                voxel_coords_diff=voxel_coords_diff,
                data_type=data_type,
                num_points=num_points,
                extra_param_0=extra_param_0,
                use_numerical=use_numerical,
                delta=delta
            )
        elif self.extractor_type == "swinattention_v2":
            # Swin Attention V2特征提取器
            return self.ptnet.get_jacobian(
                p0=p0, 
                mask_fn=Mask_fn, 
                a_fn=A_fn, 
                ax_fn=Ax_fn, 
                bn_fn=BN_fn, 
                max_idx=max_idx, 
                mode=mode,
                voxel_coords_diff=voxel_coords_diff,
                data_type=data_type,
                num_points=num_points,
                extra_param_0=extra_param_0,
                use_numerical=use_numerical,
                delta=delta
            )
        elif self.extractor_type == "ssm_v1":
            # SSM V1特征提取器
            return self.ptnet.get_jacobian(
                p0=p0, 
                mask_fn=Mask_fn, 
                a_fn=A_fn, 
                ax_fn=Ax_fn, 
                bn_fn=BN_fn, 
                max_idx=max_idx, 
                mode=mode,
                voxel_coords_diff=voxel_coords_diff,
                data_type=data_type,
                num_points=num_points,
                ssm_data=ssm_data,
                use_numerical=use_numerical,
                delta=delta
            )
        elif self.extractor_type == "ssm_v2":
            # SSM V2特征提取器
            return self.ptnet.get_jacobian(
                p0=p0, 
                mask_fn=Mask_fn, 
                a_fn=A_fn, 
                ax_fn=Ax_fn, 
                bn_fn=BN_fn, 
                max_idx=max_idx, 
                mode=mode,
                voxel_coords_diff=voxel_coords_diff,
                data_type=data_type,
                num_points=num_points,
                ssm_data=ssm_data,
                use_numerical=use_numerical,
                delta=delta
            )
        else:
            raise ValueError(f"不支持的特征提取器类型: {self.extractor_type}")

    def iclk_new(self, g0, p0, p1, maxiter, xtol, mode, voxel_coords_diff=None, data_type='synthetic', num_random_points=100, use_numerical=False, delta=1.0e-3):
        training = self.ptnet.training
        if training:
            self.step_train += 1
        else:
            self.step_test += 1
        batch_size = p0.size(0)
        num_points = p0.size(1)
        p0 = p0.float()   # T
        p1 = p1.float()   # S
        g0 = g0.float()   # T-->S

        g = g0
        
        # create a data sampler
        if mode != 'test':
            data_sampler = np.random.choice(num_points, (num_points//num_random_points, num_random_points), replace=False)
        # input through entire pointnet
        if training:
            # first, update BatchNorm modules
            f0 = self.ptnet(p0[:, data_sampler[0], :], 0)
            f1 = self.ptnet(p1[:, data_sampler[0], :], 0)
        self.ptnet.eval()

        if mode != 'test':
            for i in range(1, num_points//num_random_points-1):
                f0 = self.ptnet(p0[:, data_sampler[i], :], i)
                f1 = self.ptnet(p1[:, data_sampler[i], :], i)
                
        # 计算雅可比矩阵所需的中间输出
        if mode == 'test':
            if self.extractor_type == "pointnet":
                f0, Mask_fn, A_fn, Ax_fn, BN_fn, max_idx = self.ptnet(p0, -1)
                J = self.Cal_Jac(Mask_fn, A_fn, Ax_fn, BN_fn, max_idx,
                              num_points, p0, mode, voxel_coords_diff=voxel_coords_diff, 
                              data_type=data_type, use_numerical=use_numerical, delta=delta)
            elif self.extractor_type == "3dmamba_v1":
                f0, Mask_fn, A_fn, Ax_fn, BN_fn, max_idx, mamba_in, mamba_out = self.ptnet(p0, -1)
                J = self.Cal_Jac(Mask_fn, A_fn, Ax_fn, BN_fn, max_idx,
                              num_points, p0, mode, voxel_coords_diff=voxel_coords_diff, 
                              data_type=data_type, mamba_in=mamba_in, mamba_out=mamba_out,
                              use_numerical=use_numerical, delta=delta)
            elif self.extractor_type == "3dmamba_v2":
                f0, Mask_fn, A_fn, Ax_fn, BN_fn, max_idx, bissm_in, bissm_out = self.ptnet(p0, -1)
                J = self.Cal_Jac(Mask_fn, A_fn, Ax_fn, BN_fn, max_idx,
                              num_points, p0, mode, voxel_coords_diff=voxel_coords_diff, 
                              data_type=data_type, mamba_in=bissm_in, mamba_out=bissm_out,
                              use_numerical=use_numerical, delta=delta)
            elif self.extractor_type == "pointnet_attention_v1":
                f0, Mask_fn, A_fn, Ax_fn, BN_fn, max_idx, attn_data = self.ptnet(p0, -1)
                J = self.Cal_Jac(Mask_fn, A_fn, Ax_fn, BN_fn, max_idx,
                              num_points, p0, mode, voxel_coords_diff=voxel_coords_diff, 
                              data_type=data_type, extra_param_0=attn_data,
                              use_numerical=use_numerical, delta=delta)
            elif self.extractor_type == "fastpointtransformer_v1":
                f0, Mask_fn, A_fn, Ax_fn, BN_fn, max_idx = self.ptnet(p0, -1)
                J = self.Cal_Jac(Mask_fn, A_fn, Ax_fn, BN_fn, max_idx,
                              num_points, p0, mode, voxel_coords_diff=voxel_coords_diff, 
                              data_type=data_type, use_numerical=use_numerical, delta=delta)
            elif self.extractor_type == "fastpointtransformer_v2":
                f0, Mask_fn, A_fn, Ax_fn, BN_fn, max_idx, fpt_data = self.ptnet(p0, -1)
                J = self.Cal_Jac(Mask_fn, A_fn, Ax_fn, BN_fn, max_idx,
                              num_points, p0, mode, voxel_coords_diff=voxel_coords_diff, 
                              data_type=data_type, extra_param_0=fpt_data,
                              use_numerical=use_numerical, delta=delta)
            elif self.extractor_type == "swinattention_v1":
                f0, Mask_fn, A_fn, Ax_fn, BN_fn, max_idx, attn_info = self.ptnet(p0, -1)
                J = self.Cal_Jac(Mask_fn, A_fn, Ax_fn, BN_fn, max_idx,
                              num_points, p0, mode, voxel_coords_diff=voxel_coords_diff, 
                              data_type=data_type, extra_param_0=attn_info,
                              use_numerical=use_numerical, delta=delta)
            elif self.extractor_type == "swinattention_v2":
                f0, Mask_fn, A_fn, Ax_fn, BN_fn, max_idx, attn_info = self.ptnet(p0, -1)
                J = self.Cal_Jac(Mask_fn, A_fn, Ax_fn, BN_fn, max_idx,
                              num_points, p0, mode, voxel_coords_diff=voxel_coords_diff, 
                              data_type=data_type, extra_param_0=attn_info,
                              use_numerical=use_numerical, delta=delta)
            elif self.extractor_type == "ssm_v1":
                f0, Mask_fn, A_fn, Ax_fn, BN_fn, max_idx, ssm_data = self.ptnet(p0, -1)
                J = self.Cal_Jac(Mask_fn, A_fn, Ax_fn, BN_fn, max_idx,
                              num_points, p0, mode, voxel_coords_diff=voxel_coords_diff, 
                              data_type=data_type, ssm_data=ssm_data,
                              use_numerical=use_numerical, delta=delta)
            elif self.extractor_type == "ssm_v2":
                f0, Mask_fn, A_fn, Ax_fn, BN_fn, max_idx, ssm_data = self.ptnet(p0, -1)
                J = self.Cal_Jac(Mask_fn, A_fn, Ax_fn, BN_fn, max_idx,
                              num_points, p0, mode, voxel_coords_diff=voxel_coords_diff, 
                              data_type=data_type, ssm_data=ssm_data,
                              use_numerical=use_numerical, delta=delta)
        else:
            if num_points >= num_random_points:
                random_idx = np.random.choice(num_points, num_random_points, replace=False)
            else:
                random_idx = np.random.choice(num_points, num_random_points, replace=True)
            random_points = p0[:, random_idx]
            
            if self.extractor_type == "pointnet":
                f0, Mask_fn, A_fn, Ax_fn, BN_fn, max_idx = self.ptnet(random_points, -1)
                J = self.Cal_Jac(Mask_fn, A_fn, Ax_fn, BN_fn, max_idx, 
                             num_random_points, random_points, mode, 
                             voxel_coords_diff=None, data_type=data_type,
                             use_numerical=use_numerical, delta=delta)
            elif self.extractor_type == "3dmamba_v1":
                f0, Mask_fn, A_fn, Ax_fn, BN_fn, max_idx, mamba_in, mamba_out = self.ptnet(random_points, -1)
                J = self.Cal_Jac(Mask_fn, A_fn, Ax_fn, BN_fn, max_idx, 
                             num_random_points, random_points, mode, 
                             voxel_coords_diff=None, data_type=data_type,
                             mamba_in=mamba_in, mamba_out=mamba_out,
                             use_numerical=use_numerical, delta=delta)
            elif self.extractor_type == "3dmamba_v2":
                f0, Mask_fn, A_fn, Ax_fn, BN_fn, max_idx, bissm_in, bissm_out = self.ptnet(random_points, -1)
                J = self.Cal_Jac(Mask_fn, A_fn, Ax_fn, BN_fn, max_idx, 
                             num_random_points, random_points, mode, 
                             voxel_coords_diff=None, data_type=data_type,
                             mamba_in=bissm_in, mamba_out=bissm_out,
                             use_numerical=use_numerical, delta=delta)
            elif self.extractor_type == "pointnet_attention_v1":
                f0, Mask_fn, A_fn, Ax_fn, BN_fn, max_idx, attn_data = self.ptnet(random_points, -1)
                J = self.Cal_Jac(Mask_fn, A_fn, Ax_fn, BN_fn, max_idx, 
                             num_random_points, random_points, mode, 
                             voxel_coords_diff=None, data_type=data_type,
                             extra_param_0=attn_data, use_numerical=use_numerical, delta=delta)
            elif self.extractor_type == "fastpointtransformer_v1":
                f0, Mask_fn, A_fn, Ax_fn, BN_fn, max_idx = self.ptnet(random_points, -1)
                J = self.Cal_Jac(Mask_fn, A_fn, Ax_fn, BN_fn, max_idx, 
                             num_random_points, random_points, mode, 
                             voxel_coords_diff=None, data_type=data_type,
                             use_numerical=use_numerical, delta=delta)
            elif self.extractor_type == "fastpointtransformer_v2":
                f0, Mask_fn, A_fn, Ax_fn, BN_fn, max_idx, fpt_data = self.ptnet(random_points, -1)
                J = self.Cal_Jac(Mask_fn, A_fn, Ax_fn, BN_fn, max_idx, 
                             num_random_points, random_points, mode, 
                             voxel_coords_diff=None, data_type=data_type,
                             extra_param_0=fpt_data, use_numerical=use_numerical, delta=delta)
            elif self.extractor_type == "swinattention_v1":
                f0, Mask_fn, A_fn, Ax_fn, BN_fn, max_idx, attn_info = self.ptnet(random_points, -1)
                J = self.Cal_Jac(Mask_fn, A_fn, Ax_fn, BN_fn, max_idx,
                              num_random_points, random_points, mode, 
                              voxel_coords_diff=None, data_type=data_type,
                              extra_param_0=attn_info, use_numerical=use_numerical, delta=delta)
            elif self.extractor_type == "swinattention_v2":
                f0, Mask_fn, A_fn, Ax_fn, BN_fn, max_idx, attn_info = self.ptnet(random_points, -1)
                J = self.Cal_Jac(Mask_fn, A_fn, Ax_fn, BN_fn, max_idx,
                              num_random_points, random_points, mode, 
                              voxel_coords_diff=None, data_type=data_type,
                              extra_param_0=attn_info, use_numerical=use_numerical, delta=delta)
            elif self.extractor_type == "ssm_v1":
                f0, Mask_fn, A_fn, Ax_fn, BN_fn, max_idx, ssm_data = self.ptnet(random_points, -1)
                J = self.Cal_Jac(Mask_fn, A_fn, Ax_fn, BN_fn, max_idx,
                              num_random_points, random_points, mode, 
                              voxel_coords_diff=None, data_type=data_type,
                              ssm_data=ssm_data, use_numerical=use_numerical, delta=delta)
            elif self.extractor_type == "ssm_v2":
                f0, Mask_fn, A_fn, Ax_fn, BN_fn, max_idx, ssm_data = self.ptnet(random_points, -1)
                J = self.Cal_Jac(Mask_fn, A_fn, Ax_fn, BN_fn, max_idx,
                              num_random_points, random_points, mode, 
                              voxel_coords_diff=None, data_type=data_type,
                              ssm_data=ssm_data, use_numerical=use_numerical, delta=delta)

        # compute psuedo inverse of the Jacobian to solve delta(xi)
        Jt = J.transpose(1, 2)   # [B, 6, K]
        H = Jt.bmm(J)   # [B, 6, 6]
        # 添加一个小的正则化项到H矩阵的对角线上，确保其可逆
        H = H + torch.eye(H.size(-1), device=H.device) * 1e-4
        B = self.inverse(H)
        pinv = B.bmm(Jt)   # [B, 6, K]
        
        # iteratively solve for the pose
        itr = 0
        r = None
        for itr in range(maxiter):
            self.prev_r = r
            # [B, 1, 4, 4] x [B, N, 3] -> [B, N, 3]
            if mode == 'test':
                p = self.transform(g.unsqueeze(1), p1)   # in local frame
            else:
                p = self.transform(g.unsqueeze(1), p1[:, random_idx])
            
            if not training:
                with torch.no_grad():
                    f = self.ptnet(p.float(), 0)   # [B, N, 3] -> [B, K], in local frame / global frame
                    if mode == 'test' and data_type == 'real':
                        r = f.sum(0) - f0.sum(0)
                        r = r.unsqueeze(0)
                    else:
                        r = f - f0
            else:
                f = self.ptnet(p.float(), 0)   # [B, N, 3] -> [B, K]
                r = f - f0    # [B, K]

            if mode != 'test' or data_type == 'synthetic':
                dx = pinv.bmm(r.unsqueeze(-1)).view(batch_size, 6)
            else:   # during voxelization, testing
                dx = pinv.bmm(r.unsqueeze(-1)).view(1, 6)
            check = dx.norm(p=2, dim=1, keepdim=True).max()
            if float(check) < xtol:
                if itr == 0:
                    self.last_err = 0   # no update
                break
            g = self.update(g, dx)

        self.ptnet.train(training)

        return r, g, (itr+1)

