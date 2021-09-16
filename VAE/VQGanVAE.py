
class VQGanVAE(nn.Module):
    def __init__(self, vqgan_model_path=None, vqgan_config_path=None):
        super().__init__()

        if vqgan_model_path is None:
            model_filename = 'vqgan.1024.model.ckpt'
            config_filename = 'vqgan.1024.config.yml'
            download(VQGAN_VAE_CONFIG_PATH, config_filename)
            download(VQGAN_VAE_PATH, model_filename)
            config_path = str(Path(CACHE_PATH) / config_filename)
            model_path = str(Path(CACHE_PATH) / model_filename)
        else:
            model_path = vqgan_model_path
            config_path = vqgan_config_path

        config = OmegaConf.load(config_path)

        model = instantiate_from_config(config["model"])

        state = torch.load(model_path, map_location = 'cpu')['state_dict']
        model.load_state_dict(state, strict = False)

        print(f"Loaded VQGAN from {model_path} and {config_path}")

        self.model = model

        # f as used in https://github.com/CompVis/taming-transformers#overview-of-pretrained-models
        f = config.model.params.ddconfig.resolution / config.model.params.ddconfig.attn_resolutions[0]
        self.num_layers = int(log(f)/log(2))
        self.image_size = 256
        self.num_tokens = config.model.params.n_embed
        self.is_gumbel = isinstance(self.model, GumbelVQ)

        self._register_external_parameters()

    def _register_external_parameters(self):
        """Register external parameters for DeepSpeed partitioning."""
        if (
                not distributed_utils.is_distributed
                or not distributed_utils.using_backend(
                    distributed_utils.DeepSpeedBackend)
        ):
            return

        deepspeed = distributed_utils.backend.backend_module
        deepspeed.zero.register_external_parameter(
            self, self.model.quantize.embed.weight if self.is_gumbel else self.model.quantize.embedding.weight)

    @torch.no_grad()
    def get_codebook_indices(self, img):
        b = img.shape[0]
        img = (2 * img) - 1
        _, _, [_, _, indices] = self.model.encode(img)
        if self.is_gumbel:
            return rearrange(indices, 'b h w -> b (h w)', b=b)
        return rearrange(indices, '(b n) -> b n', b = b)

    def decode(self, img_seq):
        b, n = img_seq.shape
        one_hot_indices = F.one_hot(img_seq, num_classes = self.num_tokens).float()
        z = one_hot_indices @ self.model.quantize.embed.weight if self.is_gumbel \
            else (one_hot_indices @ self.model.quantize.embedding.weight)

        z = rearrange(z, 'b (h w) c -> b c h w', h = int(sqrt(n)))
        img = self.model.decode(z)

        img = (img.clamp(-1., 1.) + 1) * 0.5
        return img

    def forward(self, img):
        raise NotImplemented
