from datasets import load_dataset
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class XENOTRONDataProcessor:
    def __init__(self):
        self.data_sources = []
        self.processed_data = []
        
    def add_data_source(self, source: str, source_type: str = "file"):
        """Add a data source for training"""
        self.data_sources.append({
            "source": source,
            "type": source_type
        })
        logger.info(f"Added data source: {source} ({source_type})")
    
    def load_from_file(self, file_path: str) -> List[str]:
        """Load data from a text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                return [line.strip() for line in lines if line.strip()]
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            return []
    
    def load_from_api(self, api_url: str, headers: Dict = None) -> List[str]:
        """Load data from an API endpoint"""
        try:
            response = requests.get(api_url, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            # Extract text content (adjust based on API structure)
            texts = []
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and 'text' in item:
                        texts.append(item['text'])
                    elif isinstance(item, str):
                        texts.append(item)
            elif isinstance(data, dict) and 'text' in data:
                texts.append(data['text'])
            elif isinstance(data, str):
                texts.append(data)
                
            return texts
        except Exception as e:
            logger.error(f"Error loading data from API {api_url}: {e}")
            return []
    
    def load_from_huggingface(self, dataset_name: str, split: str = "train") -> List[str]:
        """Load data from Hugging Face datasets"""
        try:
            dataset = load_dataset(dataset_name, split=split)
            texts = [item['text'] for item in dataset if 'text' in item]
            return texts
        except Exception as e:
            logger.error(f"Error loading data from Hugging Face {dataset_name}: {e}")
            return []
    
    def process_data(self) -> List[str]:
        """Process all data sources and return clean text data"""
        all_texts = []
        
        for source in self.data_sources:
            if source["type"] == "file":
                texts = self.load_from_file(source["source"])
                all_texts.extend(texts)
            elif source["type"] == "api":
                texts = self.load_from_api(source["source"])
                all_texts.extend(texts)
            elif source["type"] == "huggingface":
                texts = self.load_from_huggingface(source["source"])
                all_texts.extend(texts)
        
        # Clean and filter data
        cleaned_texts = []
        for text in all_texts:
            if text and len(text.strip()) > 0:
                cleaned_texts.append(text.strip())
        
        self.processed_data = cleaned_texts
        logger.info(f"Processed {len(cleaned_texts)} texts")
        return cleaned_texts
    
    def sample_data(self, n_samples: int = 1000) -> List[str]:
        """Sample data for training"""
        if len(self.processed_data) == 0:
            self.process_data()
            
        if len(self.processed_data) <= n_samples:
            return self.processed_data
            
        return random.sample(self.processed_data, n_samples)
    
    def get_data_stats(self) -> Dict[str, Any]:
        """Get statistics about the processed data"""
        if len(self.processed_data) == 0:
            self.process_data()
            
        total_chars = sum(len(text) for text in self.processed_data)
        total_words = sum(len(text.split()) for text in self.processed_data)
        
        return {
            "total_texts": len(self.processed_data),
            "total_characters": total_chars,
            "total_words": total_words,
            "avg_text_length": total_chars / len(self.processed_data) if self.processed_data else 0,
            "data_sources": len(self.data_sources)
        }

class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int = 2048):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.text