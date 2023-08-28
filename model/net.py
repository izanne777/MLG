from layer import *
from MICBlocl_local_global import *
from MICBlock import *
from SCIBlock import *

class gtnet(nn.Module):

    def __init__(self, gcn_true, buildA_true, gcn_depth, num_nodes, device, predefined_A=None, static_feat=None, dropout=0.3, subgraph_size=20, node_dim=40, dilation_exponential=1, conv_channels=32, residual_channels=1, skip_channels=64, end_channels=128, seq_length=12, in_dim=2, out_dim=12, layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True, conv_kernel=[12, 16], hid_size =6):
        super(gtnet, self).__init__()
        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_A = predefined_A
        self.residual_channels = residual_channels

        self.seq_length = seq_length
        self.label_length = seq_length
        self.pred_length = out_dim

        self.idx = torch.arange(self.num_nodes).to(device)
        self.conv_kernel = conv_kernel
        self.hid_size = hid_size

        self.layers = layers
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        

        self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha, static_feat=static_feat)

        ## SCI-block
        self.SCIblock = SCINet(output_len = self.pred_length, input_len = self.seq_length, input_dim = self.num_nodes, groups =1, 
                                        hid_size = 6, num_levels = 1, concat_len = 0, kernel = 5, dropout = 0.0)

        # mixprop(self, c_in, c_out, gdep, dropout, alpha)
        self.gconv1 = mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha)
        self.gconv2 = mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha)       
        self.residual_convs =  nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.seq_length))
        self.skip_convs = nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.seq_length))
        self.norm = LayerNorm((residual_channels, num_nodes, self.seq_length),elementwise_affine=layer_norm_affline)

        
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                             out_channels=end_channels,
                                             kernel_size=(1,1),
                                             bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                             out_channels=out_dim,
                                             kernel_size=(1,1),
                                             bias=True)


        self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length), bias=True)
        self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1), bias=True)

        self.idx = torch.arange(self.num_nodes).to(device)
        
      
        ## MIC-block      
        self.micn = MICblock(pred_len = self.pred_length, seq_len = self.label_length, enc_in = self.num_nodes, end_in = 7, d_model=self.num_nodes, n_heads=8, dropout=0.0, d_layers=1, c_out=self.num_nodes,  conv_kernel=self.conv_kernel)
        self.conv1 = nn.Conv1d(in_channels = self.seq_length, out_channels=self.pred_length, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels = self.residual_channels*2, out_channels=self.residual_channels, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels = self.residual_channels, out_channels=self.pred_length, kernel_size=1)

    def forward(self, input, idx=None):
        
        if self.gcn_true:
            if self.buildA_true:
                if idx is None:
                    adp = self.gc(self.idx)
                else:
                    adp = self.gc(idx)
            else:
                adp = self.predefined_A

        # Seires Stationarizaion - normalization
        means = input.mean(3, keepdim=True).detach()
        x_enc = input - means
        stdev = torch.sqrt(
            torch.var(input, dim=3, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        input = x_enc

        x_enc = torch.squeeze(input ,dim=1)
        x_enc = x_enc.permute(0, 2, 1)
        dec_inp = torch.zeros([x_enc.shape[0], self.pred_length, x_enc.shape[-1]]).float().cuda(0)
        dec_inp = torch.cat([x_enc[:,:self.label_length,:], dec_inp], dim=1).float().cuda(0)

        micn = self.MICblock(x_enc, dec_inp)
        
        micn = micn.permute(0, 2, 1)
        micn = torch.unsqueeze(micn ,dim=1)

        # print('before 1x1 conv: ', (micn + input).shape)
        
        micn = F.dropout(micn, self.dropout, training=self.training)
        micn = micn+input
        x = self.start_conv( micn)
        skip = self.skip0(F.dropout(micn, self.dropout, training=self.training))   

      
        
        for i in range(self.layers):

            residual = x           
            x = torch.squeeze(x ,dim=1)

            x = x.reshape(-1, self.num_nodes, self.seq_length)

            x = x.permute(0, 2, 1)
            x = self.SCIblock(x)   
            x = x.permute(0, 2, 1)
            filter = torch.tanh(x)
            gate = torch.sigmoid(x)

            x = filter * gate 
            x = x.reshape(-1, self.residual_channels, self.num_nodes, self.seq_length)
          
            # x = F.dropout(x, self.dropout, training=self.training)

            s = x
            s = self.skip_convs(s)
            skip = s + skip

            if self.gcn_true:
              
                x = self.gconv1(x, adp)+self.gconv2(x, adp.transpose(1,0))
            else:
                x = self.residual_convs(x)

            x = x + residual

            if idx is None:
                x = self.norm(x,self.idx)
            else:
                x = self.norm(x,idx)
    
        x = x.reshape(-1, self.num_nodes, self.seq_length)

        x = x.permute(0, 2, 1)
        x = self.conv1(x)  
        # x = self.conv2(x)  
        # x = self.conv3(x)  
        x = x.permute(0, 2, 1)
      
        x = x.reshape(-1, self.residual_channels, self.num_nodes, self.pred_length)       
        
        skip = self.skipE(x) + skip

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)


        
        # Seires Stationarizaion - de normalization
        dec_out = x * \
                  (stdev[:, :, :, 0].unsqueeze(3).repeat(
                      1, 1, 1, self.pred_length))
        dec_out = dec_out + \
                  (means[:, :, :, 0].unsqueeze(3).repeat(
                      1, 1, 1, self.pred_length))

        return dec_out