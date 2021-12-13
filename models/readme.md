<details open>
<summary>ASSF</summary>

```
class ASFFV5(nn.Module):
    def __init__(self, level, multiplier=1, rfb=False, vis=False, act_cfg=True):
        """
        ASFF version for YoloV5 .
        different than YoloV3
        multiplier should be 1, 0.5
        which means, the channel of ASFF can be 
        512, 256, 128 -> multiplier=1
        256, 128, 64 -> multiplier=0.5
        For even smaller, you need change code manually.
        """
        super(ASFFV5, self).__init__()
        self.level = level
        self.dim = [1024, 512,256]
        # print(self.dim)
        
        self.inter_dim = self.dim[self.level]
        if level == 0:
            self.stride_level_1 = Conv(512, self.inter_dim, 3, 2)
                
            self.stride_level_2 = Conv(256, self.inter_dim, 3, 2)
                
            self.expand = Conv(self.inter_dim, 512, 3, 1)
        elif level == 1:
            self.compress_level_0 = Conv(1024, self.inter_dim, 1, 1)
            self.stride_level_2 = Conv(256, self.inter_dim, 3, 2)
            self.expand = Conv(self.inter_dim, 256, 3, 1)
        elif level == 2:
            self.compress_level_0 = Conv(
                1024, self.inter_dim, 1, 1)
            self.compress_level_1 = Conv(
                512, self.inter_dim, 1, 1)
            self.expand = Conv(self.inter_dim, 
                128, 3, 1)

        # when adding rfb, we use half number of channels to save memory
        compress_c = 8 if rfb else 16
        self.weight_level_0 = Conv(
            self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = Conv(
            self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(
            self.inter_dim, compress_c, 1, 1)

        self.weight_levels = Conv(
            compress_c*3, 3, 1, 1)
        self.vis = vis

    def forward(self, x): #l,m,s
        """
        # 128, 256, 512
        512, 256, 128
        from small -> large
        """
        x_level_0=x[2] #l
        x_level_1=x[1] #m
        x_level_2=x[0] #s
        # print('x_level_0: ', x_level_0.shape)
        # print('x_level_1: ', x_level_1.shape)
        # print('x_level_2: ', x_level_2.shape)
        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_downsampled_inter = F.max_pool2d(
                x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)
        elif self.level == 1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(
                level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)
        elif self.level == 2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(
                level_0_compressed, scale_factor=4, mode='nearest')
            x_level_1_compressed = self.compress_level_1(x_level_1)
            level_1_resized = F.interpolate(
                x_level_1_compressed, scale_factor=2, mode='nearest')
            level_2_resized = x_level_2

        # print('level: {}, l1_resized: {}, l2_resized: {}'.format(self.level,
        #      level_1_resized.shape, level_2_resized.shape))
        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        # print('level_0_weight_v: ', level_0_weight_v.shape)
        # print('level_1_weight_v: ', level_1_weight_v.shape)
        # print('level_2_weight_v: ', level_2_weight_v.shape)

        levels_weight_v = torch.cat(
            (level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] +\
            level_1_resized * levels_weight[:, 1:2, :, :] +\
            level_2_resized * levels_weight[:, 2:, :, :]

        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out 
```

</details>

<details open>
<summary>MHSA</summary>

```
class MHSA(nn.Module):
    def __init__(self, n_dims, width=14, height=14, heads=4,pos_emb=False):
        super(MHSA, self).__init__()

        self.heads = heads
        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.pos=pos_emb
        if self.pos :
            self.rel_h = nn.Parameter(torch.randn([1, heads, (n_dims ) // heads, 1, int(height)]), requires_grad=True)
            self.rel_w = nn.Parameter(torch.randn([1, heads, (n_dims )// heads, int(width), 1]), requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)
     
    def forward(self, x):
        n_batch, C, width, height = x.size() 
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)
        #print('q shape:{},k shape:{},v shape:{}'.format(q.shape,k.shape,v.shape))  #1,4,64,256
        content_content = torch.matmul(q.permute(0,1,3,2), k) #1,C,h*w,h*w
        # print("qkT=",content_content.shape)
        c1,c2,c3,c4=content_content.size()
        if self.pos:
       # print("old content_content shape",content_content.shape) #1,4,256,256
            content_position = (self.rel_h + self.rel_w).view(1, self.heads, C // self.heads, -1).permute(0,1,3,2)   #1,4,1024,64
           
            content_position = torch.matmul(content_position, q)# ([1, 4, 1024, 256])
            content_position=content_position if(content_content.shape==content_position.shape)else content_position[:,: , :c3,]
            assert(content_content.shape==content_position.shape)
        #print('new pos222-> shape:',content_position.shape)
       # print('new content222-> shape:',content_content.shape)
            energy = content_content + content_position
        else:
            energy=content_content
        attention = self.softmax(energy)
        out = torch.matmul(v, attention.permute(0,1,3,2)) #1,4,256,64
        out = out.view(n_batch, C, width, height)
        return out

class BottleneckTransformer(nn.Module):
    # Transformer bottleneck
    #expansion = 1

    def __init__(self, c1, c2, stride=1, heads=4, mhsa=True, resolution=None,expansion=1):
        super(BottleneckTransformer, self).__init__()
        c_=int(c2*expansion)
        self.cv1 = Conv(c1, c_, 1,1)
        #self.bn1 = nn.BatchNorm2d(c2)
        if not mhsa:
            self.cv2 = Conv(c_,c2, 3, 1)
        else:
            self.cv2 = nn.ModuleList()
            self.cv2.append(MHSA(c2, width=int(resolution[0]), height=int(resolution[1]), heads=heads))
            if stride == 2:
                self.cv2.append(nn.AvgPool2d(2, 2))
            self.cv2 = nn.Sequential(*self.cv2)
        #self.bn2 = nn.BatchNorm2d(planes)
        #self.cv3 = nn.Conv2d(planes, expansion * planes, kernel_size=1, bias=False)
        #self.bn3 = nn.BatchNorm2d(expansion * planes)
        #self.shortcut = nn.Sequential() 
        self.shortcut = c1==c2 
        if stride != 1 or c1 != expansion*c2:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c1, expansion*c2, kernel_size=1, stride=stride),
                nn.BatchNorm2d(expansion*c2)
            )
        self.fc1 = nn.Linear(c2, c2)     

    def forward(self, x):
        #print("transforme  input bottleck shape:",x.shape)
        # out = F.relu(self.bn1(self.conv1(x)))
        # out = F.relu(self.bn2(self.conv2(out)))
        # out = self.bn3(self.conv3(out))
        # out += self.shortcut(x)
        # out = F.relu(out)
        out=x + self.cv2(self.cv1(x)) if self.shortcut else self.cv2(self.cv1(x))
        return out

class CTR3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1,e=0.5,e2=1,w=20,h=20):  # ch_in, ch_out, number, , expansion,w,h
        super(CTR3, self).__init__()
        c_ = int(c2*e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[BottleneckTransformer(c_ ,c_, stride=1, heads=4,mhsa=True,resolution=(w,h),expansion=e2) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        #print("CTR3-INPUT:",x.shape)
       # return self.cv3
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))       

```

</details>

<details open>
<summary>CBAM(which performs worse in this model)</summary>

```
class ChannelAttentionModule(nn.Module):
    def __init__(self, c1, reduction=16):
        super(ChannelAttentionModule, self).__init__()
        mid_channel = c1 // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Linear(in_features=c1, out_features=mid_channel),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(in_features=mid_channel, out_features=c1)
        )
       # self.sigmoid = nn.Sigmoid()
        self.act=nn.SiLU()
    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x).view(x.size(0),-1)).unsqueeze(2).unsqueeze(3)
        maxout = self.shared_MLP(self.max_pool(x).view(x.size(0),-1)).unsqueeze(2).unsqueeze(3)
        return self.act(avgout + maxout)
        
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.act=nn.SiLU()
    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.act(self.conv2d(out))
        return out

class CBAM(nn.Module):
    def __init__(self, c1,c2):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(c1)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out
```

</details>

<details open>
<summary>BiFPN</summary>

```
class Concat_bifpn(nn.Module):   #pairwise add
    # Concatenate a list of tensors along dimension
    def __init__(self, c1, c2):
        super(Concat_bifpn, self).__init__()
        self.w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        self.conv = Conv(c1, c2, 1 ,1 , 0 )
        #self.act= nn.SiLU()  #这里原本用silu，但是用途应该是保证权重是0-1之间 所以改成relu  
        self.act=torch.nn.functional.relu
    def forward(self, x): # mutil-layer 1-3 layers
        #print("bifpn:",x.shape)
        if len(x) == 2:
            # w = self.relu(self.w1)
            w = self.w1
            weight = w / (torch.sum(w, dim=0) + self.epsilon)
            x = self.conv(self.act(weight[0] * x[0] + weight[1] * x[1]))
        elif len(x) == 3: 
            # w = self.relu(self.w2)
            w = self.w2
            weight = w / (torch.sum(w, dim=0) + self.epsilon)
            x = self.conv(self.act (weight[0] * x[0] + weight[1] * x[1] + weight[2] * x[2]))
        return x
```
