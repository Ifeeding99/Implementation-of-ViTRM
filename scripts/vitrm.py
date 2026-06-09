import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import math
import timm 

class SelfMHA(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout=0.2):
        super().__init__()
        assert embed_dim % n_heads == 0, f'Embedding dimension ({embed_dim}) must be divisible by n_heads {n_heads}!'
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.Q_weights = nn.Linear(embed_dim, embed_dim, bias=False)
        self.K_weights = nn.Linear(embed_dim, embed_dim, bias=False)
        self.V_weights = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_out_weights = nn.Linear(embed_dim, embed_dim, bias=False)
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x): # x has shape B, seq_len, embed_dim
        q = self.Q_weights(x)
        k = self.K_weights(x)
        v = self.V_weights(x)
        q = einops.rearrange(q, 'B S (n_heads head_dim) -> B n_heads S head_dim',
                             B = x.shape[0], S = x.shape[1], n_heads = self.n_heads, head_dim = self.head_dim)
        k = einops.rearrange(k, 'B S (n_heads head_dim) -> B n_heads S head_dim',
                             B = x.shape[0], S = x.shape[1], n_heads = self.n_heads, head_dim = self.head_dim)
        v = einops.rearrange(v, 'B S (n_heads head_dim) -> B n_heads S head_dim',
                             B = x.shape[0], S = x.shape[1], n_heads = self.n_heads, head_dim = self.head_dim)
        scores = F.softmax((q @ k.transpose(-1,-2))/math.sqrt(self.head_dim), dim = -1)
        scores = self.drop(scores)
        scores = scores @ v
        scores = einops.rearrange(scores, 'B n_heads S head_dim -> B S (n_heads head_dim)',
                                  B = x.shape[0], S = x.shape[1], n_heads = self.n_heads, head_dim = self.head_dim)
        scores = self.W_out_weights(scores)
        return scores
    

class SwiGLU(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.W1 = nn.Linear(embed_dim, 2*embed_dim)
        self.W2 = nn.Linear(embed_dim, 2*embed_dim)
        self.W3 = nn.Linear(2*embed_dim, embed_dim)

    def forward(self, x):
        out1 = F.silu(self.W1(x))
        out2 = self.W2(x)
        out1 = out1 * out2
        out1 = self.W3(out1)
        return out1
    

class SimplePatchEmbeddings(nn.Module):
    def __init__(self, input_img_size, embed_dim, patch_size, in_channels=3):
        super().__init__()
        assert input_img_size % patch_size == 0, f'Input image dimension {input_img_size} should be divisible by patch size ({patch_size})!'
        self.patchify = nn.Conv2d(in_channels=in_channels, out_channels=embed_dim,
                                  kernel_size=patch_size, stride=patch_size, padding=0)
        self.n_total_patches = (input_img_size // patch_size)**2
        self.n_patches_per_side = input_img_size // patch_size
        self.pos_embeddings = nn.Parameter(torch.randn(1, self.n_total_patches, embed_dim))
        self.embed_dim =embed_dim

    def forward(self, x): # x has shape B C H W
        patches = self.patchify(x) # shape B embed_dim N_patches_per_side N_patches_per_side
        patches = einops.rearrange(patches, 'B emb_dim n_patches_side_h n_patches_side_w -> B (n_patches_side_h n_patches_side_w) emb_dim',
                                   n_patches_side_h = self.n_patches_per_side, n_patches_side_w = self.n_patches_per_side, emb_dim = self.embed_dim)
        pos_emb = einops.repeat(self.pos_embeddings, '1 n_patches emb_dim -> B n_patches emb_dim',
                                B = x.shape[0], n_patches = self.n_total_patches, emb_dim = self.embed_dim)
        return patches + pos_emb


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout=0.2):
        super().__init__()
        self.l_norm1 = nn.LayerNorm(embed_dim)
        self.l_norm2 = nn.LayerNorm(embed_dim)
        self.mha = SelfMHA(embed_dim=embed_dim, n_heads=n_heads, dropout=dropout)
        self.ff = SwiGLU(embed_dim=embed_dim)

    def forward(self, x):
        x = x + self.mha(self.l_norm1(x))
        x = x + self.ff(self.l_norm2(x))
        return x



class Encoder(nn.Module):
    def __init__(self, n_blocks, embed_dim, n_heads, dropout=0.2):
        super().__init__()
        self.enc = [EncoderBlock(embed_dim=embed_dim, n_heads=n_heads, dropout=dropout) for i in range(n_blocks)]

    def forward(self, x):
        for block in self.enc:
            x = block(x)
        return x
    

class ViTPatchifier(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.model = timm.create_model(
            'vit_base_patch16_224',
            pretrained=True,
            num_classes=0 # we don't need the classifier, because we only
            )             # use the model for the embeddings
        for param in self.model.parameters(): # no finetuning
            param.requires_grad = False
        self.embed_dim = embed_dim
        self.proj = nn.Linear(in_features=768, out_features=embed_dim, bias=False) # ViT outputs embeddings with embed_dim = 768
                                                                                   # This is needed to get the produced embedding to the target dim
        
    def forward(self, x):
        return self.proj(self.model(x))

    

class ViTRM(nn.Module):
    def __init__(self, n_blocks, embed_dim, n_heads, patch_size, 
                 M, T, n_classes, input_img_size, k=3, dropout=0.2, in_channels=3,
                 use_ViT_embeddings = False,):
        super().__init__() 
        # M is the number of steps for refining memory (z) (1st cycle)
        # T is the number of reasoning steps (for refining y) (2nd cycle)
        # k is the last elements on dimension 1 of the z vector ([:,-k,:])
        self.use_ViT_embeddings = use_ViT_embeddings
        if self.use_ViT_embeddings:
            self.patch_embeddings = ViTPatchifier(embed_dim=embed_dim)
        else:
            self.patch_embeddings = SimplePatchEmbeddings(input_img_size=input_img_size, 
                                                        embed_dim=embed_dim, patch_size=patch_size, in_channels=in_channels)
        self.encoder = Encoder(n_blocks, embed_dim, n_heads, dropout)
        self.y0 = nn.Parameter(torch.randn(1,1,embed_dim))
        self.z0 = nn.Parameter(torch.randn(1,k,embed_dim))
        self.classification_head = nn.Linear(embed_dim, n_classes)
        self.halting_head = nn.Linear(embed_dim, 1)
        self.M = M
        self.T = T
        self.k = k


    def refine_memory(self,x,y,z):
        if self.use_ViT_embeddings:
                x = x.unsqueeze(1) # ViT gives back embeddings shaped like B, embd_dim. But y and z have an extra dimension
        for m_step in range(self.M):
            conc = torch.cat([x,y,z], dim=1)
            conc = self.encoder(conc)
            z = conc[:,-self.k:,:]
        return z
    
    def update_y(self, y,z):
        conc = torch.cat([y,z], dim=1)
        conc = self.encoder(conc)
        y = conc[:,:-self.k,:]
        return y

    def forward(self, x, y, z):
        # we assume that x is shaped B, seq_len, embed_dim
        # i.e. x is a matrix of embeddings
        for t_step in range(self.T):
            z = self.refine_memory(x,y,z)
            y = self.update_y(y, z)
        pred = self.classification_head(y.squeeze(1))
        halting_prob = torch.sigmoid(self.halting_head(y.squeeze(1)))
        return pred, halting_prob, y, z
    


if __name__ == '__main__':
    model = ViTRM(n_blocks=1,embed_dim=100,n_heads=10, patch_size=10,
                  M = 5, T = 1, n_classes=4, input_img_size=300, use_ViT_embeddings=True)
    
    dummy_input = torch.rand([2,3,224,224])
    y = einops.repeat(model.y0, '1 1 emb_dim -> B 1 emb_dim', B=dummy_input.shape[0])
    z = einops.repeat(model.z0, '1 k emb_dim -> B k emb_dim', B=dummy_input.shape[0], k=model.k)
    patches = model.patch_embeddings(dummy_input)
    model(patches, y, z)
    print('done')

