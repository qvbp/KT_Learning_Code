from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer
import torch
import numpy as np
import json

class BertEmbedding:
    def __init__(self, device='cuda'):
        """
        Initialize BERT model for English text
        """
        self.device = device
        # 由于是中文文本，使用中文预训练模型
        self.tokenizer = AutoTokenizer.from_pretrained('./bge-large-zh-v1.5')
        self.model = BertModel.from_pretrained('./bge-large-zh-v1.5').to(device)
        self.model.eval()

    def get_embeddings(self, texts):
        """
        Get BERT embeddings for a list of texts
        
        Args:
            texts: list of strings
            
        Returns:
            numpy array of shape (n_texts, 768)
        """
        # 将文本分批处理，避免内存溢出
        batch_size = 128
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=1024,  # 设置最大长度
                return_tensors='pt'
            ).to(self.device)
            
            # Get BERT embeddings
            with torch.no_grad():
                outputs = self.model(**encoded)
                # Use [CLS] token embeddings
                embeddings = outputs.last_hidden_state[:, 0, :]
                all_embeddings.append(embeddings.cpu())
        
        # Concatenate all batches
        all_embeddings = torch.cat(all_embeddings, dim=0)
        return all_embeddings.numpy()

    def save_embeddings(self, embeddings, output_file):
        """
        保存embeddings到文件
        """
        np.save(output_file, embeddings)

def load_json_file(file_path):
    """
    加载JSON文件并提取文本
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取所有的value值（文本）
    texts = list(data.values())
    # 保存对应的keys，以便后续参考
    keys = list(data.keys())
    return texts, keys

if __name__ == "__main__":
    # 初始化embedder
    embedder = BertEmbedding()
    
    # 加载JSON文件
    json_file_path = './data/kcs_context_peiyou.json'  # 替换为你的JSON文件路径
    texts, keys = load_json_file(json_file_path)
    
    print(f"Total number of texts to process: {len(texts)}")
    
    print(texts)
    # 获取embeddings
    vectors = embedder.get_embeddings(texts)
    
    print(f"Number of vectors: {len(vectors)}")
    print(f"Vector shape: {vectors.shape}")
    
    # 保存embeddings到文件
    output_file = './data/kc_embeddings_peiyou_bge.npy'
    embedder.save_embeddings(vectors, output_file)
    
    # 计算相似度矩阵（可选）
    similarity_matrix = np.dot(vectors, vectors.T)
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    similarity_matrix = similarity_matrix / (norm * norm.T)
    
    # 打印一些示例相似度（可选）
    print("\nExample similarities:")
    for i in range(min(5, len(texts))):
        print(f"\nText: {texts[i]}")
        print(f"Key: {keys[i]}")
        # 找出最相似的3个其他文本
        most_similar = np.argsort(similarity_matrix[i])[-4:-1][::-1]  # 不包括自己
        for idx in most_similar:
            print(f"Similar text ({similarity_matrix[i][idx]:.3f}): {texts[idx]} (Key: {keys[idx]})")

