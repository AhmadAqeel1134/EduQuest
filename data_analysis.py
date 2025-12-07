"""
Data loading, preprocessing, and visualization
Fulfills rubric requirement for dataset handling
"""

import os
from typing import List, Dict, Any
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Fix for langchain-huggingface compatibility
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except (ImportError, AttributeError):
    from sentence_transformers import SentenceTransformer
    from langchain_core.embeddings import Embeddings
    
    class HuggingFaceEmbeddings(Embeddings):
        def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
        
        def embed_documents(self, texts):
            return self.model.encode(texts).tolist()
        
        def embed_query(self, text):
            return self.model.encode([text])[0].tolist()


class DataAnalyzer:
    """Handles data loading, preprocessing, and visualization"""
    
    def __init__(self, data_dir: str = "pdfs"):
        self.data_dir = data_dir
        self.documents = []
        self.stats = {}
    
    def load_all_pdfs(self) -> List[Dict[str, Any]]:
        """Load all PDFs from data directory"""
        pdf_files = list(Path(self.data_dir).glob("*.pdf"))
        documents = []
        
        print(f"Found {len(pdf_files)} PDF files")
        
        for pdf_path in pdf_files:
            try:
                loader = PyMuPDFLoader(str(pdf_path))
                docs = loader.load()
                
                for doc in docs:
                    documents.append({
                        'source': pdf_path.name,
                        'content': doc.page_content,
                        'page': doc.metadata.get('page', 0),
                        'length': len(doc.page_content)
                    })
                
                print(f"✓ Loaded {pdf_path.name}: {len(docs)} pages")
            except Exception as e:
                print(f"✗ Error loading {pdf_path.name}: {e}")
        
        self.documents = documents
        return documents
    
    def preprocess_data(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
        """Preprocess documents by splitting into chunks"""
        if not self.documents:
            self.load_all_pdfs()
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        chunks = []
        for doc in self.documents:
            splits = splitter.split_text(doc['content'])
            for i, split in enumerate(splits):
                chunks.append({
                    'source': doc['source'],
                    'chunk_id': i,
                    'content': split,
                    'length': len(split),
                    'word_count': len(split.split())
                })
        
        return chunks
    
    def calculate_statistics(self) -> Dict[str, Any]:
        """Calculate dataset statistics"""
        if not self.documents:
            self.load_all_pdfs()
        
        total_docs = len(set(d['source'] for d in self.documents))
        total_pages = len(self.documents)
        total_chars = sum(d['length'] for d in self.documents)
        total_words = sum(len(d['content'].split()) for d in self.documents)
        
        avg_doc_length = total_chars / total_pages if total_pages > 0 else 0
        avg_word_count = total_words / total_pages if total_pages > 0 else 0
        
        self.stats = {
            'total_pdfs': total_docs,
            'total_pages': total_pages,
            'total_characters': total_chars,
            'total_words': total_words,
            'average_doc_length': avg_doc_length,
            'average_word_count': avg_word_count,
            'documents': [d['source'] for d in self.documents]
        }
        
        return self.stats
    
    def visualize_data(self, output_dir: str = "visualizations"):
        """Create visualizations of the dataset"""
        if not self.stats:
            self.calculate_statistics()
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Document length distribution
        doc_lengths = [d['length'] for d in self.documents]
        plt.figure(figsize=(10, 6))
        plt.hist(doc_lengths, bins=30, edgecolor='black')
        plt.xlabel('Document Length (characters)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Document Lengths')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{output_dir}/doc_length_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Documents per PDF
        pdf_counts = {}
        for doc in self.documents:
            pdf_counts[doc['source']] = pdf_counts.get(doc['source'], 0) + 1
        
        plt.figure(figsize=(12, 6))
        pdfs = list(pdf_counts.keys())
        counts = list(pdf_counts.values())
        plt.bar(range(len(pdfs)), counts, edgecolor='black')
        plt.xlabel('PDF Files')
        plt.ylabel('Number of Pages')
        plt.title('Pages per PDF Document')
        plt.xticks(range(len(pdfs)), [p[:20] + '...' if len(p) > 20 else p for p in pdfs], 
                   rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/pages_per_pdf.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Word count distribution
        word_counts = [len(d['content'].split()) for d in self.documents]
        plt.figure(figsize=(10, 6))
        plt.hist(word_counts, bins=30, edgecolor='black', color='green', alpha=0.7)
        plt.xlabel('Word Count per Document')
        plt.ylabel('Frequency')
        plt.title('Distribution of Word Counts')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{output_dir}/word_count_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Summary statistics table
        stats_df = pd.DataFrame([self.stats])
        stats_df.to_csv(f"{output_dir}/dataset_statistics.csv", index=False)
        
        print(f"✓ Visualizations saved to {output_dir}/")
        return output_dir
    
    def generate_report(self) -> str:
        """Generate text report of dataset analysis"""
        if not self.stats:
            self.calculate_statistics()
        
        report = "=" * 60 + "\n"
        report += "DATASET ANALYSIS REPORT\n"
        report += "=" * 60 + "\n\n"
        report += f"Total PDF Files: {self.stats['total_pdfs']}\n"
        report += f"Total Pages: {self.stats['total_pages']}\n"
        report += f"Total Characters: {self.stats['total_characters']:,}\n"
        report += f"Total Words: {self.stats['total_words']:,}\n"
        report += f"Average Document Length: {self.stats['average_doc_length']:.0f} characters\n"
        report += f"Average Word Count: {self.stats['average_word_count']:.0f} words\n\n"
        report += "PDF Files in Dataset:\n"
        for pdf in set(self.stats['documents']):
            report += f"  - {pdf}\n"
        
        return report
