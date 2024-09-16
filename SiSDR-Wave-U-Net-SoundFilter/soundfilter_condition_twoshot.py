## python train.py -C config/train/train.json
import torch
import torch.nn as nn
import torch.nn.functional as F

class DownSamplingLayer(nn.Module):
    def __init__(self, channel_in, channel_out,stride=2):
        super(DownSamplingLayer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(channel_in, channel_out, kernel_size=2*stride,
                      stride=stride),
            nn.GroupNorm(channel_out//16,channel_out),
            nn.ELU(alpha=1.0)
        )
        self.residual11 = nn.Sequential(nn.Conv1d(channel_in, channel_in,kernel_size= 3, padding='same', dilation=1),nn.GroupNorm(channel_in//16,channel_in), nn.ELU(alpha=1.0))
        self.residual12 = nn.Sequential(nn.Conv1d(channel_in, channel_in,kernel_size= 1),nn.GroupNorm(channel_in//16,channel_in),  nn.ELU(alpha=1.0))

        self.residual21 = nn.Sequential(nn.Conv1d(channel_in, channel_in,kernel_size= 3, padding='same', dilation=3), nn.GroupNorm(channel_in//16,channel_in),nn.ELU(alpha=1.0))
        self.residual22 =  nn.Sequential(nn.Conv1d(channel_in, channel_in,kernel_size= 1), nn.GroupNorm(channel_in//16,channel_in), nn.ELU(alpha=1.0))

        self.residual31 = nn.Sequential(nn.Conv1d(channel_in, channel_in,kernel_size= 3, padding='same', dilation=9),nn.GroupNorm(channel_in//16,channel_in), nn.ELU(alpha=1.0))
        self.residual32 =  nn.Sequential(nn.Conv1d(channel_in, channel_in,kernel_size= 1), nn.GroupNorm(channel_in//16,channel_in), nn.ELU(alpha=1.0))



    def forward(self, ipt):
        o1 = ipt

        o2 = self.residual11(o1)

        o3 = self.residual12(o2)
        o4 = self.residual21(o3 + o1) # TRY CONCAT ?
        o5 = self.residual22(o4)
        o6 = self.residual31(o1+o3+o5) # TRY CONCAT ?
        o7 = self.residual32(o6)
        o8 = self.main(o1 + o3 + o5 + o7)
        return o8 # TRY CONCAT ?

class FilmDownSamplingLayer(nn.Module):
    def __init__(self, channel_in, channel_out,stride=2):
        super(FilmDownSamplingLayer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(channel_in, channel_out, kernel_size=2*stride,
                      stride=stride),
            nn.GroupNorm(channel_out//16,channel_out),
            nn.ELU(alpha=1.0)
        )
        self.residual11 = nn.Sequential(nn.Conv1d(channel_in, channel_in,kernel_size= 3, padding='same', dilation=1),nn.GroupNorm(channel_in//16,channel_in), nn.ELU(alpha=1.0))
        self.residual12 = nn.Sequential(nn.Conv1d(channel_in, channel_in,kernel_size= 1),nn.GroupNorm(channel_in//16,channel_in),  nn.ELU(alpha=1.0))

        self.residual21 = nn.Sequential(nn.Conv1d(channel_in, channel_in,kernel_size= 3, padding='same', dilation=3), nn.GroupNorm(channel_in//16,channel_in),nn.ELU(alpha=1.0))
        self.residual22 =  nn.Sequential(nn.Conv1d(channel_in, channel_in,kernel_size= 1), nn.GroupNorm(channel_in//16,channel_in), nn.ELU(alpha=1.0))

        self.residual31 = nn.Sequential(nn.Conv1d(channel_in, channel_in,kernel_size= 3, padding='same', dilation=9),nn.GroupNorm(channel_in//16,channel_in), nn.ELU(alpha=1.0))
        self.residual32 =  nn.Sequential(nn.Conv1d(channel_in, channel_in,kernel_size= 1), nn.GroupNorm(channel_in//16,channel_in), nn.ELU(alpha=1.0))



    def forward(self, ipt):
        o1 = ipt
        o2 = self.residual11(o1)
        o3 = self.residual12(o2)
        o4 = self.residual21(o3 + o1) # TRY CONCAT ?
        o5 = self.residual22(o4)
        o6 = self.residual31(o1+o3+o5) # TRY CONCAT ?
        o7 = self.residual32(o6)
        o8 = self.main(o1 + o3 + o5 + o7)
        return o8 # TRY CONCAT ?
        

class UpSamplingLayer(nn.Module):
    def __init__(self, channel_in, channel_out, Stride):
        super(UpSamplingLayer, self).__init__()
        if(channel_in == 512): 
            output_padding = 0
        elif(channel_in == 256):
            output_padding = 5
        else:
            output_padding = 0

        self.upblock = nn.Sequential(
            nn.ConvTranspose1d(channel_in,channel_out, kernel_size=2*Stride,
                      stride=Stride, output_padding=output_padding),
            nn.GroupNorm(channel_out//16,channel_out),
            nn.ELU(alpha=1.0)
            #nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.residual11 = nn.Sequential(nn.Conv1d(channel_out, channel_out, kernel_size= 3, padding='same', dilation=1),nn.GroupNorm(channel_out//16,channel_out),  nn.ELU(alpha=1.0))
        self.residual12 = nn.Sequential(nn.Conv1d(channel_out, channel_out,kernel_size= 1), nn.GroupNorm(channel_out//16,channel_out),nn.ELU(alpha=1.0))
        self.residual21 = nn.Sequential(nn.Conv1d(channel_out, channel_out,kernel_size= 3, padding='same', dilation=3),nn.GroupNorm(channel_out//16,channel_out), nn.ELU(alpha=1.0))
        self.residual22 = nn.Sequential(nn.Conv1d(channel_out, channel_out,kernel_size= 1),nn.GroupNorm(channel_out//16,channel_out), nn.ELU(alpha=1.0))
        self.residual31 = nn.Sequential(nn.Conv1d(channel_out, channel_out,kernel_size= 3, padding='same', dilation=9),nn.GroupNorm(channel_out//16,channel_out), nn.ELU(alpha=1.0))
        self.residual32 = nn.Sequential(nn.Conv1d(channel_out, channel_out,kernel_size= 1),nn.GroupNorm(channel_out//16,channel_out), nn.ELU(alpha=1.0))

    def forward(self, ipt,gamma1,gamma2,beta1,beta2):
        bs = gamma1.shape[0]
        o1 = self.upblock(ipt)
        o1 = gamma1.view(bs,1,1) * o1 + beta1.view(bs,1,1)
        o2 = self.residual11(o1)
        o3 = self.residual12(o2)
        o4 = self.residual21(o3 + o1) # TRY CONCAT ?

        o5 = self.residual22(o4)

        o6 = self.residual31(o1+o3+o5) # TRY CONCAT ?

        o7 = self.residual32(o6)

        o8 = o1 + o3 + o5 + o7
        o8 = gamma2.view(bs,1,1) * o8 + beta2.view(bs,1,1)

        return o8 # TRY CONCAT ?
        
        

class Model(nn.Module):
    def __init__(self, n_layers=4):
        super(Model, self).__init__()

        #########################################################
        # ENCODER
        #########################################################
        self.encConv1 = nn.Sequential(nn.Conv1d(1, 32,kernel_size= 7), nn.GroupNorm(2,32), nn.ELU(alpha=1.0))
        self.n_layers = n_layers
        
        encoder_in_channels_list = [32,64,128,256]
        encoder_out_channels_list = [64,128,256,512]
        encoder_stride_list = [2,2,8,8]

        self.encoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.encoder.append(
                DownSamplingLayer(
                    channel_in=encoder_in_channels_list[i],
                    channel_out=encoder_out_channels_list[i],
                    stride = int(encoder_stride_list[i])
                )
            )
        self.encConv2 = nn.Sequential(nn.Conv1d(512, 256,kernel_size= 7,padding = 'same'), nn.GroupNorm(16,256), nn.ELU(alpha=1.0))

        #########################################################
        # FILM
        #########################################################
        film_in_channels_list = [32,64,128,256]
        film_out_channels_list = [64,128,256,512]
        modules = []
        for i in range(self.n_layers):
            modules.append(
                FilmDownSamplingLayer(
                    channel_in=film_in_channels_list[i],
                    channel_out=film_out_channels_list[i],
                    stride = int(encoder_stride_list[i])
                )
            )
        #modules.append(nn.Flatten())
        #modules.append(nn.Linear(31744,2*(2*n_layers +1)))
        #self.film_gen = nn.Sequential(nn.Conv1d(1, 32,kernel_size= 7), nn.BatchNorm1d(32), nn.ELU(alpha=1.0), *modules)

        self.first_conv = nn.Sequential(nn.Conv1d(1, 32,kernel_size= 7), nn.GroupNorm(2,32), nn.ELU(alpha=1.0))
        self.list = nn.Sequential(*modules,nn.Conv1d(512, 256,kernel_size= 7,padding = 'same'),
                                  nn.BatchNorm1d(256), nn.ELU(alpha=1.0),
                                  nn.AvgPool1d(kernel_size= 3, stride=3),
                                  nn.ELU(alpha=1.0))
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(2560,(6*n_layers +2))
        #self.film_gen = nn.Sequential(*modules)
        #########################################################
        # DECODER
        #########################################################

        self.decConv1 = nn.Sequential(nn.Conv1d(256, 512 ,kernel_size= 7,padding = 'same'), nn.GroupNorm(32,512), nn.ELU(alpha=1.0))
        decoder_in_channels_list = [512,256,128,64]
        decoder_out_channels_list = [256,128,64,32]
        decoder_stride_list = [8,8,2,2]
        self.decoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.decoder.append(
                UpSamplingLayer(
                    channel_in=decoder_in_channels_list[i],
                    channel_out=decoder_out_channels_list[i],
                    Stride = int(decoder_stride_list[i])
                )
            )
        self.decConv2 = nn.Sequential(nn.Conv1d(32, 1 ,padding = 6,kernel_size= 7),  nn.BatchNorm1d(1), nn.ELU(alpha=1.0))
        self.double()

    def forward_condition(self,conditioning1, conditioning2):
        x = self.first_conv(conditioning1)
        x = self.list(x)
        x = self.flatten(x)
        film_out1 = self.linear(x)

        x = self.first_conv(conditioning2)
        x = self.list(x)
        #x = torch.mean(x,2)
        x = self.flatten(x)
        film_out2 = self.linear(x)

        return film_out1, film_out2

    def forward(self, input, conditioning1, conditioning2):
        tmp = []
        o = input
        o = o.double()
        conditioning1 = conditioning1.double()
        conditioning2 = conditioning2.double()
        tmp.append(o)
        # Up Sampling
        o = self.encConv1(o)
        tmp.append(o)
        #film_out1 = self.film_gen(conditioning1)
        #film_out2 = self.film_gen(conditioning2)
        film_out1, film_out2 = self.forward_condition(conditioning1, conditioning2)
        gammas = (film_out1[:,:3*self.n_layers+1] + film_out2[:,:3*self.n_layers+1] )/2
        betas = (film_out1[:,3*self.n_layers+1:] + film_out2[:,3*self.n_layers+1:])/2
        for i in range(self.n_layers):
            o = self.encoder[i](o)
            o = o*(gammas[0,i]) + (betas[0,i]) #affine transform
            tmp.append(o)

        o = self.encConv2(o)
        o = o*gammas[0,self.n_layers] + betas[0,self.n_layers]
        o = self.decConv1(o)
        
        o = o + tmp[-1] # TRY CONCAT ?
        # Down Sampling
        for i in range(self.n_layers):
            o = self.decoder[i](o,(gammas[:,self.n_layers+i+1]),gammas[:,self.n_layers+i+2],
                                (betas[:,self.n_layers+i+1]),betas[:,self.n_layers+i+2])
            #o = o*gammas[self.n_layers+i+1] + betas[self.n_layers+i+1] #affine transform
            o = o + tmp[self.n_layers - i] # TRY CONCAT ?
        o = self.decConv2(o)

        o = o + tmp[0]  # TRY CONCAT ?
        
        return o
