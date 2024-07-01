class E4e(Module):
    def __init__(self, num_layers=50, gan_size=256, w_mode='w+', pretrained=True, latent_avg_offset=True):
        super().__init__()
        assert gan_size in (256, 1024)
        assert w_mode in (None, 'w', 'w+')

        if w_mode == 'w':
            print('Warning! Setting w_mode="w" removes modules, required for w+.')
        self.w_mode = w_mode
        self.img_size = 256  # this is constant, unlike 'size', which depends on the GAN resolution!

        self.construct(num_layers, w_mode, gan_size)


        self.num_ws = int(np.log2(gan_size) - 1) * 2


        self.latent_avg_offset = latent_avg_offset
        if self.latent_avg_offset:
            self.register_buffer('latent_avg', torch.zeros(1, self.num_ws, 512))

        if pretrained:
            self.load(gan_size, w_mode)
        self.eval()
        self.requires_grad_(False)

            
    def load(self, gan_size, mode):
        if gan_size == 1024:
            state_file = os.path.join(thisdir, 'encoder_state_dict.pt')
            state_dict = torch.load(state_file, map_location='cpu')
            state_dict['latent_avg'] = torch.load(os.path.join(thisdir, 'latent_avg.pt'), map_location='cpu').unsqueeze(0)
        else:
            state_file = os.path.join(thisdir, 'outdir_ffhq256/checkpoints/best_model.pt')
            state_dict = torch.load(state_file, map_location='cpu')['state_dict']
            state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items()}

            unused_keys = []
            if mode == 'w':
                """ Remove keys not used in 'w' mode """
                for key in state_dict:
                    if 'styles' in key and not 'styles.0' in key:
                        unused_keys.append(key)
                    elif 'latlayer' in key:
                        unused_keys.append(key)

            """ Remove decoder keys"""
            for key in state_dict:
                if 'decoder' in key:
                    unused_keys.append(key)

            state_dict = {k: v for k, v in state_dict.items() if k not in unused_keys}

        

        self.load_state_dict(state_dict)

    def construct(self, num_layers, w_mode, gan_size):
        blocks = get_blocks(num_layers)
        unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        log_size = int(math.log(gan_size, 2))
        self.style_count = 2 * log_size - 2
        self.coarse_ind = 3
        self.middle_ind = 7

        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)

        self.progressive_stage = ProgressiveStage.Inference

        if w_mode == 'w':
            self.styles = self.styles[:1]
        elif w_mode == 'w+':
            self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
            self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

    def get_deltas_starting_dimensions(self):
        ''' Get a list of the initial dimension of every delta from which it is applied '''
        return list(range(self.style_count))  # Each dimension has a delta applied to it

    def set_progressive_stage(self, new_stage: ProgressiveStage):
        self.progressive_stage = new_stage
        print('Changed progressive stage to: ', new_stage)

    def get_fpn_feats(self, x):
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x
        return c1, c2, c3

    def downsample(self, image):
        return F.interpolate(image, (self.img_size, self.img_size), mode='area')

    def forward(self, x_01, w_mode=None):
        x = x_01 * 2 - 1
        if x.size(-1) != self.img_size:
            x = self.downsample(x)

        w_mode = w_mode or self.w_mode
        assert w_mode is not None
        
        x = self.input_layer(x)

        c1, c2, c3 = self.get_fpn_feats(x)
        # Infer main W and duplicate it
        w0 = self.styles[0](c3)
        if w_mode == 'w':
            if self.latent_avg_offset:
                w0 = w0 + self.latent_avg[:, 0]
            return w0
        
        w = w0.repeat(self.style_count, 1, 1).permute(1, 0, 2)
        stage = self.progressive_stage.value
        features = c3
        for i in range(1, min(stage + 1, self.style_count)):  # Infer additional deltas
            if i == self.coarse_ind:
                p2 = _upsample_add(c3, self.latlayer1(c2))  # FPN's middle features
                features = p2
            elif i == self.middle_ind:
                p1 = _upsample_add(p2, self.latlayer2(c1))  # FPN's fine features
                features = p1
            delta_i = self.styles[i](features)
            w[:, i] += delta_i

        if self.latent_avg_offset:
            w = w + self.latent_avg
        return w
def load_e4e():
    e4e = E4e(gan_size=256, pretrained=False, latent_avg_offset=True).eval().requires_grad_(False)
    state_dict = {}
    ckpt = torch.load('e4e.pt', map_location='cpu')
    state_dict['latent_avg'] = ckpt['latent_avg']
    state_dict.update(extract_from_statedict(ckpt['state_dict'], 'encoder'))
    e4e.load_state_dict(state_dict)
    return e4e
