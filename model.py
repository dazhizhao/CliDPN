import torch.nn as nn
import torch.nn.functional as F
import torch


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True) + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)

class VQVAE(nn.Module):
    def __init__(self, dim=64, embedding_dim=16, num_embeddings=256, beta=0.25, target_channels=3):
        super().__init__()
        self.data_variance = 1.0
        self.target_channels = target_channels
        self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim, beta)
        self.filmfusion = FiLMFusion()
        self.vector_to_image = AuxiliaryMLP()
        self.feature_fusion = MultiHeadAttentionFusion(input_dim=32, num_heads=4)
        
        self.Encoder = nn.Sequential(
            nn.Conv2d(in_channels=self.target_channels, out_channels=dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
            nn.Conv2d(dim, embedding_dim, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

        self.Decoder = nn.Sequential(
            nn.Conv2d(embedding_dim, dim, kernel_size=3, stride=1, padding=1),
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
            nn.ReLU(),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(dim, self.target_channels, 4, 2, 1),
            nn.Sigmoid()
        )

    def encode(self, x, feat, one_hot):
        vector = self.filmfusion(feat, one_hot)
        vector_feat = self.vector_to_image(vector)
        z_e_x = self.Encoder(x)

        fusion_data = self.feature_fusion(z_e_x, vector_feat)

        loss, quantized, perplexity, encodings = self._vq_vae(fusion_data)
        
        return loss, quantized, perplexity, encodings

    def decode(self, latents):
        x_recon = self.Decoder(latents)
        return x_recon

class AuxiliaryMLP(nn.Module):
    def __init__(self, input_dim=16, outout_dim=32):
        super(AuxiliaryMLP, self).__init__()
        self.input_dim = input_dim
        self.outout_dim = outout_dim
        self.mlp = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 16*32),
            nn.ReLU(),
            nn.Linear( 16*32 , 16*32*32 ),
            nn.Sigmoid()
        )

    def forward(self, vector):
        batch_size = vector.size(0)
        out = self.mlp(vector)
        out = out.view(batch_size, self.input_dim, self.outout_dim, self.outout_dim)
        return out
    # resize [B,16,32,32]

class FiLMFusion(nn.Module):
    def __init__(self, cont_dim=4, one_hot_dim=4):
        super().__init__()
        self.film_generator = nn.Sequential(
            nn.Linear(one_hot_dim, 32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU(),
            nn.Linear(32, cont_dim * 2)
        )
        
    def forward(self, cont_features, one_hot):
        film_params = self.film_generator(one_hot)
        gamma, beta = torch.chunk(film_params, 2, dim=1)
        
        modulated_features = gamma * cont_features + beta
        return modulated_features  # [batch_size, cont_dim]
    
    
class MultiHeadAttentionFusion(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(MultiHeadAttentionFusion, self).__init__()
        self.num_heads = num_heads
        self.input_dim = input_dim

        # MultiheadAttention layer
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)
        self.output_transform = nn.Conv2d(input_dim, input_dim // 2, kernel_size=1)
        # Output normalization layer
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, feature_r, feature_n):
        # Concatenate the two input features along the channel dimension
        feature_c = torch.cat([feature_r, feature_n], dim=1)  # [B, 32, 32, 32]
        batch_size, _, height, width = feature_c.shape

        # Reshape to match MultiheadAttention's expected input shape
        # MultiheadAttention expects input of shape [batch_size, seq_length, embed_dim]
        reshaped_feature_c = feature_c.view(batch_size, -1, height * width).permute(0, 2, 1)  # [B, H*W, C]

        # MultiheadAttention forward
        # Here, we use the same tensor as query, key, and value
        attention_output, _ = self.attention(reshaped_feature_c, reshaped_feature_c, reshaped_feature_c)  # [B, H*W, C]

        # Reshape back to the original spatial dimensions
        output = attention_output.permute(0, 2, 1).view(batch_size, -1, height, width)  # [B, C, H, W]

        output = output + feature_c
        # Normalize
        output = self.norm(output)
        output = self.output_transform(output)

        return output

class Predictor(nn.Module):
    def __init__(self, embedding_dim=16, size=32):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.size = size
        self.filmfusion = FiLMFusion()
        
        self.predictor = nn.Sequential(
            nn.Linear(4, embedding_dim),
            nn.LeakyReLU(),
            
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(),
            
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(),
            
            nn.Linear(embedding_dim, embedding_dim * size * size),
            nn.LeakyReLU()
        )
        
        self.conv = nn.Sequential(
            nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embedding_dim),
            nn.LeakyReLU(),
            
            nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embedding_dim),
            nn.LeakyReLU(),
            
            nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embedding_dim),
            nn.LeakyReLU(),
            
            ResBlock(embedding_dim),
            ResBlock(embedding_dim),
            ResBlock(embedding_dim),
            
            nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embedding_dim),
            nn.LeakyReLU(),
        )
        
    def forward(self, x, one_hot):
        b, v = x.size()
        vector = self.filmfusion(x, one_hot)
        vector = self.predictor(vector)
        latents = torch.reshape(vector, (b, self.embedding_dim, self.size, self.size))
        latents = self.conv(latents)
        return latents

