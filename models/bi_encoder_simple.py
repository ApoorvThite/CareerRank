"""
Simplified Bi-Encoder using sentence-transformers library
This avoids the transformers packaging import issue
"""

import torch
import numpy as np
from typing import List
from pathlib import Path
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader


class SimpleBiEncoder:
    """
    Simplified bi-encoder using sentence-transformers library.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: str = None):
        """
        Initialize bi-encoder.
        
        Args:
            model_name: Model name (without sentence-transformers/ prefix)
            device: Device to use
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
        except:
            # Try with sentence-transformers prefix
            self.model = SentenceTransformer(f'sentence-transformers/{model_name}', device=self.device)
        
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        print(f"Initialized SimpleBiEncoder with {model_name}")
        print(f"Embedding dimension: {self.embedding_dim}")
        print(f"Device: {self.device}")
    
    def encode_texts(self, texts: List[str], batch_size: int = 32,
                     show_progress: bool = False, normalize: bool = True) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Args:
            texts: List of text strings
            batch_size: Batch size
            show_progress: Show progress bar
            normalize: Normalize embeddings
        
        Returns:
            embeddings: numpy array of shape [num_texts, embedding_dim]
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )
        
        return embeddings
    
    def save_pretrained(self, save_directory: str):
        """Save model to directory."""
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.model.save(str(save_path))
        
        # Save config
        import json
        config = {
            'embedding_dim': self.embedding_dim,
            'model_type': 'sentence-transformers'
        }
        with open(save_path / 'bi_encoder_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Model saved to {save_directory}")
    
    @classmethod
    def load_pretrained(cls, load_directory: str, device: str = None):
        """Load model from directory."""
        model = cls.__new__(cls)
        model.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        model.model = SentenceTransformer(load_directory, device=model.device)
        model.embedding_dim = model.model.get_sentence_embedding_dimension()
        
        print(f"Model loaded from {load_directory}")
        return model


if __name__ == "__main__":
    # Test
    print("Testing SimpleBiEncoder...")
    
    model = SimpleBiEncoder()
    
    test_texts = [
        "Software engineer with Python experience",
        "Data scientist with ML skills",
        "Product manager"
    ]
    
    embeddings = model.encode_texts(test_texts)
    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"Normalized: {np.allclose(np.linalg.norm(embeddings, axis=1), 1.0)}")
    
    # Test similarity
    sim_matrix = np.dot(embeddings, embeddings.T)
    print(f"\nSimilarity matrix:\n{sim_matrix}")
    
    print("\n✓ SimpleBiEncoder test passed!")
