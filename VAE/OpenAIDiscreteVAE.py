
# pretrained Discrete VAE from OpenAI

class OpenAIDiscreteVAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc = load_model(download(OPENAI_VAE_ENCODER_PATH))
        self.dec = load_model(download(OPENAI_VAE_DECODER_PATH))
        make_contiguous(self)

        self.num_layers = 3
        self.image_size = 256
        self.num_tokens = 8192

    @torch.no_grad()
    def get_codebook_indices(self, img):
        img = map_pixels(img)
        z_logits = self.enc.blocks(img)
        z = torch.argmax(z_logits, dim = 1)
        return rearrange(z, 'b h w -> b (h w)')

    def decode(self, img_seq):
        b, n = img_seq.shape
        img_seq = rearrange(img_seq, 'b (h w) -> b h w', h = int(sqrt(n)))

        z = F.one_hot(img_seq, num_classes = self.num_tokens)
        z = rearrange(z, 'b h w c -> b c h w').float()
        x_stats = self.dec(z).float()
        x_rec = unmap_pixels(torch.sigmoid(x_stats[:, :3]))
        return x_rec

    def forward(self, img):
        raise NotImplemented
