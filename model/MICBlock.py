import torch
import torch.nn as nn

from MICBlocl_local_global import series_decomp, series_decomp_multi
import torch.nn.functional as F


class MIC(nn.Module):
    """
    MIC layer to extract local and global features
    """

    def __init__(self, feature_size=512, n_heads=8, dropout=0.05, decomp_kernel=[32], conv_kernel=[24],
                 isometric_kernel=[18, 6], device='cuda'):
        super(MIC, self).__init__()
        self.conv_kernel = conv_kernel
        self.device = device

        # isometric convolution
        self.isometric_conv = nn.ModuleList([nn.Conv1d(in_channels=feature_size, out_channels=feature_size,
                                                       kernel_size=i, padding=0, stride=1)
                                             for i in isometric_kernel])

        # downsampling convolution: padding=i//2, stride=i
        self.conv = nn.ModuleList([nn.Conv1d(in_channels=feature_size, out_channels=feature_size,
                                             kernel_size=i, padding=i // 2, stride=i)
                                   for i in conv_kernel])

        # upsampling convolution
        self.conv_trans = nn.ModuleList([nn.ConvTranspose1d(in_channels=feature_size, out_channels=feature_size,
                                                            kernel_size=i, padding=0, stride=i)
                                         for i in conv_kernel])

        self.decomp = nn.ModuleList([series_decomp(k) for k in decomp_kernel])
        self.merge = torch.nn.Conv2d(in_channels=feature_size, out_channels=feature_size,
                                     kernel_size=(len(self.conv_kernel), 1))


        self.norm = torch.nn.LayerNorm(feature_size)
        self.act = torch.nn.Tanh()
        self.drop = torch.nn.Dropout(0.05)


    def conv_trans_conv(self, input, conv1d, conv1d_trans, isometric):
        batch, seq_len, channel = input.shape
        x = input.permute(0, 2, 1)

        # downsampling convolution
        x1 = self.drop(self.act(conv1d(x)))
        x = x1

        # isometric convolution
        zeros = torch.zeros((x.shape[0], x.shape[1], x.shape[2] - 1), device=self.device)
        x = torch.cat((zeros, x), dim=-1)
        x = self.drop(self.act(isometric(x)))
        x = self.norm((x + x1).permute(0, 2, 1)).permute(0, 2, 1)

        # upsampling convolution
        x = self.drop(self.act(conv1d_trans(x)))

        x = x[:, :, :seq_len]  # truncate

        x = x.permute(0, 2, 1)
        
        return x

    def forward(self, src):
        # multi-scale
        multi = []
        for i in range(len(self.conv_kernel)):
            src_out, trend1 = self.decomp[i](src)
            src_out = self.conv_trans_conv(src_out, self.conv[i], self.conv_trans[i], self.isometric_conv[i])
            multi.append(src_out)

            # merge
        mg = torch.tensor([], device=self.device)
        for i in range(len(self.conv_kernel)):
            mg = torch.cat((mg, multi[i].unsqueeze(1)), dim=1)
        mg = self.merge(mg.permute(0, 3, 1, 2)).squeeze(-2).permute(0, 2, 1)
    
        return mg



class SeasonalPrediction(nn.Module):
    def __init__(self, embedding_size=512, n_heads=8, dropout=0.05, d_layers=1, decomp_kernel=[32], c_out=1,
                 conv_kernel=[2, 4], isometric_kernel=[18, 6], device='cuda'):
        super(SeasonalPrediction, self).__init__()

        self.mic = nn.ModuleList([MIC(feature_size=embedding_size, n_heads=n_heads,
                                      decomp_kernel=decomp_kernel, conv_kernel=conv_kernel,
                                      isometric_kernel=isometric_kernel, device=device)
                                  for i in range(d_layers)])
        
        self.residual_convs = nn.Conv1d(in_channels=embedding_size,out_channels=embedding_size,
                                                 kernel_size=1)
        self.projection = nn.Linear(embedding_size, c_out)

    def forward(self, dec):


        for mic_layer in self.mic:
            
           
            dec = mic_layer(dec)
 

        return dec

class MICN_Model(nn.Module):
    def __init__(self, pred_len, seq_len, enc_in , end_in, d_model ,n_heads, dropout, d_layers, c_out,  conv_kernel=[12, 16]):
        super(MICN_Model, self).__init__()

        self.pred_len = pred_len
        self.seq_len = seq_len

        decomp_kernel = []  # kernel of decomposition operation
        isometric_kernel = []  # kernel of isometric convolution

        for ii in conv_kernel:
            if ii % 2 == 0:  # the kernel of decomposition operation must be odd
                decomp_kernel.append(ii + 1)
                isometric_kernel.append((self.seq_len + self.pred_len + ii) // ii)
            else:
                decomp_kernel.append(ii)
                isometric_kernel.append((self.seq_len + self.pred_len + ii - 1) // ii)

        


        # Multi-scale Hybrid Decomposition
        self.decomp_multi = series_decomp_multi(decomp_kernel)

        # # embedding
        # self.dec_embedding = DataEmbedding(enc_in, d_model, dropout)

        self.conv_trans = SeasonalPrediction(embedding_size=d_model, n_heads=n_heads,
                                             dropout=dropout,
                                             d_layers=d_layers, decomp_kernel=decomp_kernel,
                                             c_out=c_out, conv_kernel=conv_kernel,
                                             isometric_kernel=isometric_kernel, device=torch.device('cuda:0'))
    
        self.regression = nn.Linear(self.seq_len, self.seq_len)
        # self.regression.weight = nn.Parameter(
        #         (1 / self.pred_len) * torch.ones([self.pred_len, self.seq_len]),
        #         requires_grad=True)
       

    def forward(self, x_enc,  x_dec):

        # Multi-scale Hybrid Decomposition
        # seasonal_init_enc, trend = self.decomp_multi(x_enc)
       
        # mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        # seasonal_init_enc, trend = self.decomp_multi(x_enc)
        # trend = torch.cat([trend[:, -self.seq_len:, :], mean], dim=1)

        # embedding
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_init_dec = torch.cat([x_enc[:, -self.seq_len:, :], zeros], dim=1)
        
        dec_out = self.conv_trans(seasonal_init_dec)

        dec_out = dec_out[:, :-self.pred_len, :]  
        return dec_out

   



