import re
import json
from collections import Counter
import torch
from typing import List, Dict, Tuple

class XENOTRONTokenizer:
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.reverse_vocab = {}
        self.special_tokens = {
            '<PAD>': 0,
            '<START>': 1,
            '<END>': 2,
            '<UNK>': 3,
            '<USER>': 4,
            '<BOT>': 5,
            '<SARCASM>': 6,
            '<MEME>': 7,
            '<EMOJI>': 8,
            '<MENTION>': 9
        }
        self._build_vocab()
        
    def _build_vocab(self):
        # Start with special tokens
        for token, idx in self.special_tokens.items():
            self.vocab[token] = idx
            self.reverse_vocab[idx] = token
            
        # Build character-level vocabulary
        chars = [chr(i) for i in range(32, 127)]  # Printable ASCII
        chars.extend(['<SPACE>', '<NEWLINE>', '<TAB>'])
        
        # Add common Gen Z/Sarcastic terms
        gen_z_terms = [
            'lol', 'rofl', 'wtf', 'omg', 'idk', 'tbh', 'imo', 'fyi', 
            'brb', 'gtg', 'ttyl', 'omw', 'asap', 'fml', 'yolo', 'smh',
            'no cap', 'sus', 'yeet', 'fam', 'stan', 'sksksk', 'lmao',
            'salty', 'flex', 'goated', 'stan', 'sksk', 'fr', 'yeet',
            'simp', 'mood', 'vibe', 'sus', 'bussin', 'period', 'no cap',
            'fam', 'skrrt', 'sksk', 'wassup', 'sup', 'lil', 'big', 'yesss',
            'nooo', 'uhhh', 'yikes', 'haha', 'bruh', 'dang', 'dude', 'yo',
            'wtf', 'idk', 'tbh', 'imo', 'fyi', 'brb', 'gtg', 'ttyl'
        ]
        
        # Add sarcasm markers
        sarcasm_markers = [
            'oh great', 'oh wow', 'sure thing', 'as if', 'obviously',
            'of course', 'totally', 'absolutely', 'definitely', 'no doubt',
            'yeah right', 'of course not', 'obviously not', 'duh', 'obviously'
        ]
        
        # Combine all terms
        all_terms = chars + gen_z_terms + sarcasm_markers
        
        # Create vocabulary with frequency-based ranking
        term_freq = Counter(all_terms)
        sorted_terms = sorted(term_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Add remaining terms to vocabulary
        for i, (term, freq) in enumerate(sorted_terms[:self.vocab_size - len(self.special_tokens)]):
            if term not in self.vocab:
                idx = len(self.vocab)
                self.vocab[term] = idx
                self.reverse_vocab[idx] = term
    
    def encode(self, text: str, max_length: int = 2048, add_special_tokens: bool = True) -> List[int]:
        tokens = []
        
        if add_special_tokens:
            tokens.append(self.vocab.get('<START>', 0))
            
        # Preprocess text for Gen Z/Sarcastic patterns
        text = self._preprocess_text(text)
        
        # Simple tokenization - split by spaces and preserve special characters
        words = re.findall(r'\w+|[^\w\s]', text)
        
        for word in words:
            # Try to find exact match first
            if word.lower() in self.vocab:
                tokens.append(self.vocab[word.lower()])
            elif word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                # Try to match character by character
                char_tokens = []
                for char in word:
                    if char in self.vocab:
                        char_tokens.append(self.vocab[char])
                    else:
                        char_tokens.append(self.vocab.get('<UNK>', 3))
                
                # Add individual character tokens if the word wasn't found
                if not char_tokens or len(char_tokens) == 0:
                    tokens.append(self.vocab.get('<UNK>', 3))
                else:
                    tokens.extend(char_tokens)
            
            # Add space token if needed
            if word != words[-1]:
                tokens.append(self.vocab.get('<SPACE>', 0))
            
        if add_special_tokens:
            tokens.append(self.vocab.get('<END>', 0))
            
        # Pad or truncate to max_length
        if len(tokens) < max_length:
            tokens.extend([self.vocab.get('<PAD>', 0)] * (max_length - len(tokens)))
        else:
            tokens = tokens[:max_length]
            
        return tokens
    
    def decode(self, tokens: List[int]) -> str:
        chars = []
        for token in tokens:
            if token in self.reverse_vocab:
                char = self.reverse_vocab[token]
                if char not in ['<PAD>', '<START>', '<END>', '<UNK>']:
                    chars.append(char)
        return ''.join(chars)
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for Gen Z/Sarcastic patterns"""
        # Convert to lowercase
        text = text.lower()
        
        # Replace common Gen Z slang with standardized versions
        replacements = {
            'u': 'you',
            'r': 'are',
            'ur': 'your',
            'gr8': 'great',
            'thx': 'thanks',
            'plz': 'please',
            'cuz': 'because',
            'b/c': 'because',
            'u': 'you',
            'u': 'you',
            'u': 'you'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
            
        return text
    
    def get_vocab_size(self) -> int:
        return len(self.vocab)
    
    def save_vocab(self, path: str):
        """Save vocabulary to file"""
        with open(path, 'w') as f:
            json.dump(self.vocab, f)
    
    def load_vocab(self, path: str):
        """Load vocabulary from file"""
        with open(path, 'r') as f:
            self.vocab = json.load(f)
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}