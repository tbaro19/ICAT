"""
Vietnamese text processing utilities
"""
from typing import List, Optional
import re


class VietnameseTextProcessor:
    """Utilities for processing Vietnamese text"""
    
    def __init__(self):
        """Initialize Vietnamese text processor"""
        self.tokenizer = None
        self.segmenter = None
        
        # Try to load Vietnamese NLP tools
        try:
            from underthesea import word_tokenize as vn_tokenize
            self.vn_tokenize = vn_tokenize
            print("Loaded underthesea for Vietnamese tokenization")
        except ImportError:
            print("Warning: underthesea not available. Install with: pip install underthesea")
            self.vn_tokenize = None
        
        try:
            from pyvi import ViTokenizer
            self.vi_tokenizer = ViTokenizer
            print("Loaded pyvi for Vietnamese tokenization")
        except ImportError:
            print("Warning: pyvi not available. Install with: pip install pyvi")
            self.vi_tokenizer = None
    
    def tokenize(self, text: str, method: str = 'pyvi') -> str:
        """
        Tokenize Vietnamese text with word segmentation
        
        Args:
            text: Input Vietnamese text
            method: Tokenization method ('pyvi' or 'underthesea')
            
        Returns:
            Tokenized text with spaces between words
        """
        if method == 'pyvi' and self.vi_tokenizer:
            return self.vi_tokenizer.tokenize(text)
        elif method == 'underthesea' and self.vn_tokenize:
            tokens = self.vn_tokenize(text)
            return ' '.join(tokens)
        else:
            # Fallback: simple space tokenization
            return text
    
    def normalize(self, text: str) -> str:
        """
        Normalize Vietnamese text
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Lowercase
        text = text.lower()
        
        return text
    
    def preprocess_for_model(self, texts: List[str], method: str = 'pyvi') -> List[str]:
        """
        Preprocess Vietnamese texts for VLM models
        
        Args:
            texts: List of Vietnamese texts
            method: Tokenization method
            
        Returns:
            List of preprocessed texts
        """
        processed = []
        for text in texts:
            # Normalize
            text = self.normalize(text)
            
            # Tokenize
            text = self.tokenize(text, method=method)
            
            processed.append(text)
        
        return processed
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between Vietnamese texts
        Requires PhoBERT or similar Vietnamese language model
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score [0, 1]
        """
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch
            
            # Load PhoBERT
            if not hasattr(self, 'phobert_model'):
                print("Loading PhoBERT for semantic similarity...")
                self.phobert_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
                self.phobert_model = AutoModel.from_pretrained("vinai/phobert-base")
                self.phobert_model.eval()
            
            # Tokenize
            text1_seg = self.tokenize(text1, method='pyvi')
            text2_seg = self.tokenize(text2, method='pyvi')
            
            # Encode
            with torch.no_grad():
                inputs1 = self.phobert_tokenizer(text1_seg, return_tensors='pt', padding=True, truncation=True)
                inputs2 = self.phobert_tokenizer(text2_seg, return_tensors='pt', padding=True, truncation=True)
                
                outputs1 = self.phobert_model(**inputs1)
                outputs2 = self.phobert_model(**inputs2)
                
                # Use [CLS] token embeddings
                emb1 = outputs1.last_hidden_state[:, 0, :]
                emb2 = outputs2.last_hidden_state[:, 0, :]
                
                # Cosine similarity
                similarity = torch.nn.functional.cosine_similarity(emb1, emb2)
                
                return similarity.item()
        
        except Exception as e:
            print(f"Error computing Vietnamese similarity: {e}")
            # Fallback: simple word overlap
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            if len(words1) == 0 or len(words2) == 0:
                return 0.0
            
            overlap = len(words1 & words2)
            union = len(words1 | words2)
            
            return overlap / union if union > 0 else 0.0
    
    def detect_language(self, text: str) -> str:
        """
        Detect if text is Vietnamese or English
        
        Args:
            text: Input text
            
        Returns:
            'vi' for Vietnamese, 'en' for English
        """
        # Vietnamese characters
        vietnamese_chars = set('áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ')
        
        text_lower = text.lower()
        vn_count = sum(1 for char in text_lower if char in vietnamese_chars)
        
        # If more than 5% of characters are Vietnamese-specific, classify as Vietnamese
        if len(text) > 0 and vn_count / len(text) > 0.05:
            return 'vi'
        else:
            return 'en'
