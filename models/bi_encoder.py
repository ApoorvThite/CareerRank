"""
Bi-Encoder (Two-Tower) Model for Profile Retrieval
CareerRank Project - Day 3

Architecture:
- Shared encoder for both profile_a and profile_b
- Mean pooling over last hidden states
- L2 normalized embeddings
- Cosine similarity for matching
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path


class BiEncoder(nn.Module):
    """
    Bi-Encoder model with shared encoder for both towers.
    Uses mean pooling and L2 normalization.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 device: Optional[str] = None):
        """
        Initialize bi-encoder model.
        
        Args:
            model_name: Hugging Face model name
            device: Device to use (cuda/cpu)
        """
        super(BiEncoder, self).__init__()
        
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load transformer model
        try:
            self.encoder = AutoModel.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except:
            # Fallback to distilbert
            print(f"Failed to load {model_name}, falling back to distilbert-base-uncased")
            model_name = "distilbert-base-uncased"
            self.model_name = model_name
            self.encoder = AutoModel.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.encoder.to(self.device)
        
        # Get embedding dimension
        self.embedding_dim = self.encoder.config.hidden_size
        
        print(f"Initialized BiEncoder with {model_name}")
        print(f"Embedding dimension: {self.embedding_dim}")
        print(f"Device: {self.device}")
    
    def mean_pooling(self, token_embeddings: torch.Tensor, 
                     attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Mean pooling over token embeddings using attention mask.
        
        Args:
            token_embeddings: [batch_size, seq_len, hidden_dim]
            attention_mask: [batch_size, seq_len]
        
        Returns:
            pooled: [batch_size, hidden_dim]
        """
        # Expand attention mask to match token embeddings shape
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # Sum embeddings
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        
        # Sum mask (avoid division by zero)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        
        # Mean pooling
        pooled = sum_embeddings / sum_mask
        
        return pooled
    
    def encode(self, input_ids: torch.Tensor, 
               attention_mask: torch.Tensor,
               normalize: bool = True) -> torch.Tensor:
        """
        Encode input texts to embeddings.
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            normalize: Whether to L2 normalize embeddings
        
        Returns:
            embeddings: [batch_size, embedding_dim]
        """
        # Get model outputs
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Mean pooling
        embeddings = self.mean_pooling(outputs.last_hidden_state, attention_mask)
        
        # L2 normalization
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def forward(self, profile_a_inputs: Dict[str, torch.Tensor],
                profile_b_inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for both towers.
        
        Args:
            profile_a_inputs: Dict with 'input_ids' and 'attention_mask' for profile A
            profile_b_inputs: Dict with 'input_ids' and 'attention_mask' for profile B
        
        Returns:
            Dict with 'embeddings_a' and 'embeddings_b'
        """
        # Encode profile A
        embeddings_a = self.encode(
            profile_a_inputs['input_ids'],
            profile_a_inputs['attention_mask'],
            normalize=True
        )
        
        # Encode profile B
        embeddings_b = self.encode(
            profile_b_inputs['input_ids'],
            profile_b_inputs['attention_mask'],
            normalize=True
        )
        
        return {
            'embeddings_a': embeddings_a,
            'embeddings_b': embeddings_b
        }
    
    def encode_texts(self, texts: List[str], batch_size: int = 32,
                     max_length: int = 256, show_progress: bool = False) -> np.ndarray:
        """
        Encode a list of texts to embeddings (for inference).
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            max_length: Maximum sequence length
            show_progress: Whether to show progress bar
        
        Returns:
            embeddings: [num_texts, embedding_dim] numpy array
        """
        self.encoder.eval()
        
        all_embeddings = []
        
        # Process in batches
        num_batches = (len(texts) + batch_size - 1) // batch_size
        
        if show_progress:
            from tqdm import tqdm
            batch_iterator = tqdm(range(num_batches), desc="Encoding texts")
        else:
            batch_iterator = range(num_batches)
        
        with torch.no_grad():
            for i in batch_iterator:
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(texts))
                batch_texts = texts[start_idx:end_idx]
                
                # Tokenize
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors='pt'
                )
                
                # Move to device
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                
                # Encode
                embeddings = self.encode(input_ids, attention_mask, normalize=True)
                
                # Move to CPU and convert to numpy
                all_embeddings.append(embeddings.cpu().numpy())
        
        # Concatenate all batches
        all_embeddings = np.vstack(all_embeddings)
        
        return all_embeddings
    
    def save_pretrained(self, save_directory: str):
        """
        Save model and tokenizer to directory.
        
        Args:
            save_directory: Directory to save model
        """
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save encoder and tokenizer
        self.encoder.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        
        # Save config
        import json
        config = {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim
        }
        with open(save_path / 'bi_encoder_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Model saved to {save_directory}")
    
    @classmethod
    def load_pretrained(cls, load_directory: str, device: Optional[str] = None):
        """
        Load model from directory.
        
        Args:
            load_directory: Directory containing saved model
            device: Device to load model on
        
        Returns:
            BiEncoder instance
        """
        load_path = Path(load_directory)
        
        # Load config
        import json
        with open(load_path / 'bi_encoder_config.json', 'r') as f:
            config = json.load(f)
        
        # Create model instance
        model = cls(model_name=load_directory, device=device)
        
        print(f"Model loaded from {load_directory}")
        
        return model


def compute_infonce_loss(embeddings_a: torch.Tensor,
                         embeddings_b: torch.Tensor,
                         temperature: float = 0.05) -> torch.Tensor:
    """
    Compute InfoNCE (contrastive) loss with in-batch negatives.
    
    Args:
        embeddings_a: [batch_size, embedding_dim] - query embeddings
        embeddings_b: [batch_size, embedding_dim] - positive embeddings
        temperature: Temperature for scaling similarities
    
    Returns:
        loss: Scalar loss value
    """
    # Compute similarity matrix: [batch_size, batch_size]
    # Each row i contains similarities between query i and all candidates
    similarity_matrix = torch.matmul(embeddings_a, embeddings_b.T) / temperature
    
    # Labels: diagonal elements are positives (i matches with i)
    batch_size = embeddings_a.size(0)
    labels = torch.arange(batch_size, device=embeddings_a.device)
    
    # Cross-entropy loss (InfoNCE)
    loss = F.cross_entropy(similarity_matrix, labels)
    
    return loss


if __name__ == "__main__":
    # Test the model
    print("Testing BiEncoder...")
    
    model = BiEncoder()
    
    # Test encoding
    test_texts = [
        "Software engineer with 5 years of Python experience",
        "Data scientist specializing in machine learning",
        "Product manager with startup experience"
    ]
    
    embeddings = model.encode_texts(test_texts)
    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"Embeddings are normalized: {np.allclose(np.linalg.norm(embeddings, axis=1), 1.0)}")
    
    # Test similarity
    sim_matrix = np.dot(embeddings, embeddings.T)
    print(f"\nSimilarity matrix:\n{sim_matrix}")
    
    print("\n✓ BiEncoder test passed!")
