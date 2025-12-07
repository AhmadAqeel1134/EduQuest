"""
Model implementations for EduQuest Question Generator
Supports multiple generative models: Phi-3-mini, Gemini, and Baseline
"""

import os
import torch
import time
from typing import Optional, Dict, Any
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Optional import for cache checking
try:
    from huggingface_hub import try_to_load_from_cache
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

# Patch LangChain compatibility issues FIRST
try:
    import langchain_patch  # Auto-patches langchain
except ImportError:
    # Manual patch if langchain_patch not available
    import langchain
    if not hasattr(langchain, 'debug'):
        langchain.debug = False
    if not hasattr(langchain, 'verbose'):
        langchain.verbose = False

try:
    import google.generativeai as genai
    # Don't import ChatGoogleGenerativeAI - it has compatibility issues
    # from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-generativeai not installed. Gemini model will not be available.")


def _check_model_cached(model_id: str) -> bool:
    """Check if model is already cached locally"""
    try:
        if HF_HUB_AVAILABLE:
            # Check if model files exist in cache using huggingface_hub
            cached_path = try_to_load_from_cache(model_id, filename="config.json")
            return cached_path is not None
        else:
            # Fallback: check default cache directory
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            model_path = os.path.join(cache_dir, f"models--{model_id.replace('/', '--')}")
            return os.path.exists(model_path)
    except:
        # Final fallback: check Windows cache location
        try:
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            if not os.path.exists(cache_dir):
                # Try Windows AppData location
                cache_dir = os.path.join(os.getenv("LOCALAPPDATA", ""), "huggingface", "hub")
            model_path = os.path.join(cache_dir, f"models--{model_id.replace('/', '--')}")
            return os.path.exists(model_path) if model_path else False
        except:
            return False


class BaseQuestionGenerator:
    """Base class for all question generators"""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.tokenizer = None
        
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate question from prompt"""
        raise NotImplementedError
        
    def get_model_info(self) -> Dict[str, Any]:
        """Return model information"""
        return {"name": self.name}


class Phi3Generator(BaseQuestionGenerator):
    """Phi-3-mini model implementation for CPU"""
    
    def __init__(self):
        super().__init__("Phi-3-mini")
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load Phi-3 model with retry logic"""
        MODEL = "microsoft/Phi-3-mini-4k-instruct"
        max_retries = 3
        
        # Check if model is cached
        is_cached = _check_model_cached(MODEL)
        if is_cached:
            print(f"📦 Phi-3-mini found in cache - loading from disk (no download needed)")
        else:
            print(f"⬇️  Phi-3-mini not in cache - will download (~7.6GB, first time only)")
        
        for attempt in range(max_retries):
            try:
                print(f"Loading Phi-3-mini (attempt {attempt + 1}/{max_retries})...")
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    MODEL, trust_remote_code=True
                )
                
                if is_cached:
                    print("💾 Loading model weights from cache into memory (this may take 1-3 minutes)...")
                    print("   (This is normal - models must be loaded into RAM each time you run the app)")
                else:
                    print("⬇️  Downloading and loading model weights (~7.6GB, this may take 10-30 minutes)...")
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    MODEL,
                    torch_dtype=torch.float16,
                    device_map="cpu",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                
                # Apply compatibility patch
                self._patch_model()
                print("✓ Phi-3-mini loaded successfully!")
                return
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(5)
                else:
                    raise RuntimeError(f"Failed to load Phi-3 after {max_retries} attempts: {e}")
    
    def _patch_model(self):
        """Patch model to fix seen_tokens compatibility issue"""
        try:
            from transformers.cache_utils import DynamicCache
            if not hasattr(DynamicCache, '_original_getattr'):
                DynamicCache._original_getattr = DynamicCache.__getattribute__
            
            def patched_getattribute(self, name):
                if name == 'seen_tokens':
                    if not hasattr(self, '_seen_tokens'):
                        if hasattr(self, 'key_cache') and len(self.key_cache) > 0:
                            try:
                                self._seen_tokens = self.key_cache[0].shape[2] if len(self.key_cache[0].shape) > 2 else 0
                            except:
                                self._seen_tokens = 0
                        else:
                            self._seen_tokens = 0
                    return self._seen_tokens
                return DynamicCache._original_getattr(self, name)
            
            DynamicCache.__getattribute__ = patched_getattribute
        except:
            pass
    
    def generate(self, prompt: str, max_tokens: int = 200, **kwargs) -> str:
        """Generate question using Phi-3"""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded")
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=False,
                top_p=0.9,
                repetition_penalty=1.1,
            )
        
        generated_ids = outputs[0][input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return generated_text.strip()
    
    def get_model_info(self) -> Dict[str, Any]:
        info = super().get_model_info()
        info.update({
            "type": "Local LLM",
            "parameters": "3.8B",
            "precision": "float16",
            "device": "CPU"
        })
        return info


class GeminiGenerator(BaseQuestionGenerator):
    """Google Gemini 2.0 Flash API implementation (Free Tier Compatible)"""
    
    def __init__(self, api_key: str):
        super().__init__("Gemini 2.0 Flash")
        self.api_key = api_key
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Gemini client"""
        if not GEMINI_AVAILABLE:
            raise RuntimeError("google-generativeai package not installed. Install with: pip install google-generativeai langchain-google-genai")
        try:
            genai.configure(api_key=self.api_key)
            # Use Gemini 2.0 Flash (free tier compatible)
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
            
            # Use direct API instead of LangChain wrapper to avoid compatibility issues
            self.use_langchain = False
            print("✓ Gemini 2.0 Flash initialized successfully! (Using direct API)")
        except Exception as e:
            # Try alternative model name if the first one fails
            try:
                self.model = genai.GenerativeModel('gemini-2.0-flash')
                self.use_langchain = False
                print("✓ Gemini 2.0 Flash initialized successfully! (Using direct API)")
            except Exception as e2:
                raise RuntimeError(f"Failed to initialize Gemini: {e2}. Tried both gemini-2.0-flash-exp and gemini-2.0-flash")
    
    def generate(self, prompt: str, max_tokens: int = 200, **kwargs) -> str:
        """Generate question using Gemini API with rate limiting"""
        import time
        
        max_retries = 3
        retry_delay = 60  # Wait 60 seconds if quota exceeded
        
        for attempt in range(max_retries):
            try:
                # Use direct API call to avoid LangChain compatibility issues
                # For long questions, ensure we have enough tokens for comprehensive answers
                effective_max_tokens = max_tokens
                if max_tokens >= 400:  # Long question type
                    effective_max_tokens = max(max_tokens, 500)  # Ensure at least 500 tokens
                
                generation_config = {
                    'max_output_tokens': effective_max_tokens,
                    'temperature': kwargs.get('temperature', 0.7),
                }
                
                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                
                # Extract text from response
                if hasattr(response, 'text'):
                    return response.text.strip()
                elif hasattr(response, 'candidates') and len(response.candidates) > 0:
                    return response.candidates[0].content.parts[0].text.strip()
                else:
                    return str(response).strip()
                    
            except Exception as e:
                error_str = str(e)
                # Check if it's a quota/rate limit error
                if '429' in error_str or 'quota' in error_str.lower() or 'rate' in error_str.lower():
                    if attempt < max_retries - 1:
                        wait_time = retry_delay
                        print(f"⚠️  Gemini API quota exceeded. Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}...")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise RuntimeError(f"Gemini API quota exceeded after {max_retries} attempts. Please wait and try again later, or use Phi-3 or Baseline model instead.")
                else:
                    raise RuntimeError(f"Gemini generation failed: {e}")
        
        raise RuntimeError("Gemini generation failed after all retries")
    
    def get_model_info(self) -> Dict[str, Any]:
        info = super().get_model_info()
        info.update({
            "type": "API-based LLM",
            "provider": "Google",
            "model": "gemini-2.0-flash-exp"
        })
        return info


class BaselineGenerator(BaseQuestionGenerator):
    """Improved rule-based baseline for comparison"""
    
    def __init__(self):
        super().__init__("Baseline (Rule-based)")
        self.question_templates = [
            "What is the primary purpose of {concept}?",
            "Which of the following best describes {concept}?",
            "What is a key characteristic of {concept}?",
            "How is {concept} typically used?",
            "What is the main advantage of {concept}?"
        ]
        self.used_concepts = []  # Track used concepts to avoid repetition
    
    def _extract_meaningful_concepts(self, context: str) -> list:
        """Extract meaningful concepts from context using better heuristics"""
        import re
        
        # Reset used concepts if we're starting fresh (optional - can be called externally)
        # self.used_concepts = []
        
        # Remove common stop words and extract meaningful phrases
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which', 'who', 'where', 'when', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'}
        
        # Common words to exclude (not meaningful concepts)
        exclude_words = {'using', 'output', 'conference', 'figure', 'review', 'paper', 'iclr', 'bottou', 'leon', 'loosli', 'gaelle', 'canu', 'stephene', 'goal', 'create', 'please', 'alert', 'feed', 'forgetting', 'draw', 'certain', 'objects', 'addition', 'representations', 'learnt', 'under', 'as', 'show', 'that', 'space', 'learned', 'performed', 'similar', 'arithmetic', 'vectors', 'rows', 'interpolation', 'between', 'series', 'random', 'points'}
        
        # Extract sentences
        sentences = re.split(r'[.!?]+', context)
        
        # Find important terms (nouns, technical terms)
        concepts = []
        
        # Method 1: Extract multi-word technical terms (Title Case)
        title_case_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b'
        title_matches = re.findall(title_case_pattern, context)
        concepts.extend([m for m in title_matches if len(m.split()) <= 4 and m.lower() not in stop_words])
        
        # Method 2: Extract ALL CAPS terms (likely important)
        all_caps_pattern = r'\b([A-Z]{3,}(?:\s+[A-Z]{3,}){0,2})\b'
        all_caps_matches = re.findall(all_caps_pattern, context)
        concepts.extend([m for m in all_caps_matches if len(m) > 3 and len(m) < 40])
        
        # Method 3: Extract terms after colons (definitions)
        colon_pattern = r':\s*([A-Z][a-zA-Z\s]{5,40})'
        colon_matches = re.findall(colon_pattern, context)
        concepts.extend([m.strip() for m in colon_matches if len(m.strip().split()) <= 5])
        
        # Method 4: Extract quoted terms
        quoted = re.findall(r'"([^"]{5,40})"', context)
        concepts.extend([q for q in quoted if len(q.split()) <= 5])
        
        # Method 5: Extract technical terms (capitalized words, 5+ chars)
        words = context.split()
        for i, word in enumerate(words):
            word_clean = word.strip('.,!?;:()[]{}"\'').lower()
            if word_clean in stop_words or len(word_clean) < 5:
                continue
            
            # Single capitalized word (likely a concept)
            if word[0].isupper() and word[1:].islower() and len(word) >= 5:
                # Check if it's part of a multi-word concept
                if i < len(words) - 1:
                    next_word = words[i+1].strip('.,!?;:()[]{}"\'')
                    if next_word and next_word[0].isupper() and next_word[1:].islower():
                        concept = f"{word} {next_word}"
                        if concept not in concepts and len(concept) < 50:
                            concepts.append(concept)
                    elif word not in concepts:
                        concepts.append(word)
        
        # Filter and clean concepts
        filtered_concepts = []
        for c in concepts:
            c_clean = c.strip('.,!?;:()[]{}"\'')
            c_lower = c_clean.lower()
            
            # Skip if too short, too long, or contains unwanted patterns
            if (len(c_clean) < 5 or len(c_clean) > 50):
                continue
            
            # Skip common/excluded words
            if any(excluded in c_lower for excluded in exclude_words):
                continue
            
            # Skip if contains unwanted patterns
            if any(bad in c_lower for bad in ['figure', 'table', 'page', 'section', 'chapter', 'http', 'www', 'doi:', 'arxiv', 'review as', 'conference paper']):
                continue
            
            # Skip if it's just numbers or single letters
            if c_clean.replace(' ', '').isdigit() or len(c_clean.replace(' ', '')) < 5:
                continue
            
            # Skip if already used
            if c_lower in [uc.lower() for uc in self.used_concepts]:
                continue
            
            # Skip if it's a common word (like "Unsupervised" alone, "Conference", etc.)
            if len(c_clean.split()) == 1 and c_lower in exclude_words:
                continue
            
            filtered_concepts.append(c_clean)
        
        # Remove duplicates (case-insensitive)
        seen = set()
        unique_concepts = []
        for c in filtered_concepts:
            c_lower = c.lower()
            if c_lower not in seen:
                seen.add(c_lower)
                unique_concepts.append(c)
        
        return unique_concepts[:15]  # Return top 15 unique concepts
    
    def _generate_options(self, concept: str, context: str) -> tuple:
        """Generate meaningful multiple choice options"""
        import random
        import re
        
        # Extract sentences mentioning the concept
        sentences = [s.strip() for s in context.split('.') if concept.lower() in s.lower() and len(s.strip()) > 20][:4]
        
        # If we have good sentences, use them
        if len(sentences) >= 2:
            # Option A: First relevant sentence (truncated if needed)
            option_a_text = sentences[0][:100].strip()
            if len(sentences[0]) > 100:
                option_a_text += "..."
            option_a = f"A) {option_a_text}"
            
            # Option B: Second relevant sentence or related concept
            if len(sentences) > 1:
                option_b_text = sentences[1][:100].strip()
                if len(sentences[1]) > 100:
                    option_b_text += "..."
                option_b = f"B) {option_b_text}"
            else:
                option_b = "B) A related method or approach"
            
            # Option C: Different concept from context
            other_concepts = [c for c in self._extract_meaningful_concepts(context) if c.lower() != concept.lower()][:1]
            if other_concepts:
                option_c = f"C) Related to {other_concepts[0]}"
            else:
                option_c = "C) A different approach or technique"
            
            option_d = "D) None of the above"
            
            correct = random.choice(["A", "B"])
            
        else:
            # Fallback: Generate generic but meaningful options
            option_a = f"A) {concept} is a fundamental concept in this field"
            option_b = f"B) {concept} refers to a specific method or technique"
            option_c = f"C) {concept} is a type of system or framework"
            option_d = "D) All of the above"
            
            correct = "A"
        
        options = (option_a, option_b, option_c, option_d)
        return options, correct
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate question using improved rule-based approach"""
        # Detect question type from prompt
        is_mcq = "multiple-choice" in prompt.lower() or "mcq" in prompt.lower() or "option" in prompt.lower() or "A)" in prompt.lower()
        is_short = "short-answer" in prompt.lower() or "short answer" in prompt.lower()
        is_long = "long-answer" in prompt.lower() or "long answer" in prompt.lower() or "descriptive" in prompt.lower() or "detailed" in prompt.lower()
        
        # Extract context from prompt
        if "Context:" in prompt:
            context = prompt.split("Context:")[1].strip()
        else:
            context = prompt
        
        # Extract meaningful concepts (avoiding already used ones)
        concepts = self._extract_meaningful_concepts(context)
        
        if not concepts or len(concepts) == 0:
            # Fallback if no concepts found
            concepts = ["the main topic", "the subject", "the concept"]
        
        import random
        
        # Select a concept that hasn't been used yet
        available_concepts = [c for c in concepts if c.lower() not in [uc.lower() for uc in self.used_concepts]]
        if not available_concepts:
            # Reset if all concepts used (allow reuse after all are used once)
            self.used_concepts = []
            available_concepts = concepts if concepts else ["the main topic"]
        
        if not available_concepts:
            available_concepts = ["the main topic", "the subject", "the concept"]
        
        concept = random.choice(available_concepts)
        concept = concept.strip('.,!?;:()[]{}"\'')
        
        # Mark as used (case-insensitive)
        self.used_concepts.append(concept)
        
        # Generate question based on type
        if is_long:
            # Long answer question
            long_templates = [
                "Explain in detail: {concept}",
                "Describe comprehensively: {concept}",
                "Discuss the significance and applications of {concept}",
                "Provide a detailed explanation of {concept} and its importance",
                "Analyze {concept} and explain its role in the context"
            ]
            template = random.choice(long_templates)
            question = template.format(concept=concept)
            
            # Generate detailed answer from context
            import re
            sentences = [s.strip() for s in context.split('.') if concept.lower() in s.lower() and len(s.strip()) > 30][:3]
            if sentences and len(sentences) >= 2:
                # Combine 2-3 sentences, clean them up
                answer_parts = []
                for s in sentences[:2]:
                    s_clean = s.strip()
                    # Remove figure references, citations, etc.
                    s_clean = re.sub(r'Figure \d+:', '', s_clean)
                    s_clean = re.sub(r'\([^)]*\)', '', s_clean)  # Remove citations
                    if len(s_clean) > 20:
                        answer_parts.append(s_clean)
                
                if answer_parts:
                    answer = '. '.join(answer_parts) + '.'
                else:
                    answer = f"{concept} is an important concept discussed in the context. It involves key principles and has significant applications in the field."
            else:
                answer = f"{concept} is a key topic in the provided context. It involves important principles and applications that are relevant to understanding the subject matter. The concept plays a significant role in the field and has various implications for practical applications."
            
            result = f"""Question: {question}
Answer: {answer}"""
            
        elif is_short:
            # Short answer question
            short_templates = [
                "What is {concept}?",
                "Define {concept}.",
                "Explain {concept} briefly.",
                "What does {concept} refer to?",
                "Describe {concept} in one sentence."
            ]
            template = random.choice(short_templates)
            question = template.format(concept=concept)
            
            # Generate concise answer - find best sentence mentioning concept
            import re
            sentences = [s.strip() for s in context.split('.') if concept.lower() in s.lower() and len(s.strip()) > 20]
            if sentences:
                # Take first good sentence, clean it
                answer = sentences[0]
                # Remove figure references, citations
                answer = re.sub(r'Figure \d+:', '', answer)
                answer = re.sub(r'\([^)]*\)', '', answer)
                answer = answer.strip()[:200]  # Max 200 chars
                if len(answer) < 20:
                    answer = f"{concept} is a concept discussed in the context."
            else:
                answer = f"{concept} is an important concept in the provided context."
            
            result = f"""Question: {question}
Answer: {answer}"""
            
        else:
            # MCQ (default)
            template = random.choice(self.question_templates)
            question = template.format(concept=concept)
            
            # Generate meaningful options
            options, correct_answer = self._generate_options(concept, context)
            
            # Generate explanation
            explanation = f"This question tests understanding of {concept.lower()} as discussed in the context."
            
            result = f"""Question: {question}
{options[0]}
{options[1]}
{options[2]}
{options[3]}
Correct Answer: {correct_answer}
Explanation: {explanation}"""
        
        return result
    
    def get_model_info(self) -> Dict[str, Any]:
        info = super().get_model_info()
        info.update({
            "type": "Rule-based",
            "method": "Template-based generation"
        })
        return info


class TinyLlamaGenerator(BaseQuestionGenerator):
    """TinyLlama model - CPU-friendly, fast generation"""
    
    def __init__(self):
        super().__init__("TinyLlama")
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load TinyLlama model"""
        MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        max_retries = 2
        
        # Check if model is cached
        is_cached = _check_model_cached(MODEL)
        if is_cached:
            print(f"📦 TinyLlama found in cache - loading from disk (no download needed)")
        else:
            print(f"⬇️  TinyLlama not in cache - will download (~2GB, first time only)")
        
        for attempt in range(max_retries):
            try:
                print(f"Loading TinyLlama (attempt {attempt + 1}/{max_retries})...")
                
                self.tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
                
                if is_cached:
                    print("💾 Loading TinyLlama weights from cache into memory (this may take 1-2 minutes)...")
                    print("   (This is normal - models must be loaded into RAM each time you run the app)")
                else:
                    print("⬇️  Downloading and loading TinyLlama weights (~2GB, this may take 5-15 minutes)...")
                    print("⚠️  WARNING: Requires ~2GB free disk space")
                self.model = AutoModelForCausalLM.from_pretrained(
                    MODEL,
                    torch_dtype=torch.float16,
                    device_map="cpu",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                
                print("✓ TinyLlama loaded successfully!")
                return
                
            except (OSError, RuntimeError) as e:
                error_str = str(e).lower()
                if "no space" in error_str or "disk" in error_str or "space" in error_str:
                    raise RuntimeError(f"Insufficient disk space to load TinyLlama (need ~2GB): {e}")
                elif attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(3)
                else:
                    raise RuntimeError(f"Failed to load TinyLlama after {max_retries} attempts: {e}")
            except KeyboardInterrupt:
                raise RuntimeError("TinyLlama loading cancelled by user")
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(3)
                else:
                    raise RuntimeError(f"Failed to load TinyLlama after {max_retries} attempts: {e}")
    
    def generate(self, prompt: str, max_tokens: int = 200, **kwargs) -> str:
        """Generate question using TinyLlama"""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded")
        
        # Format prompt for chat model
        chat_prompt = f"<|user|>\n{prompt}<|assistant|>\n"
        
        inputs = self.tokenizer(chat_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True,  # TinyLlama supports cache
                top_p=0.9,
                repetition_penalty=1.1,
            )
        
        generated_ids = outputs[0][input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return generated_text.strip()
    
    def get_model_info(self) -> Dict[str, Any]:
        info = super().get_model_info()
        info.update({
            "type": "Local LLM",
            "parameters": "1.1B",
            "precision": "float16",
            "device": "CPU"
        })
        return info


class Phi2Generator(BaseQuestionGenerator):
    """Phi-2 model - Better quality than Phi-3-mini, smaller size"""
    
    def __init__(self):
        super().__init__("Phi-2")
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load Phi-2 model"""
        MODEL = "microsoft/phi-2"
        max_retries = 2
        
        # Check if model is cached
        is_cached = _check_model_cached(MODEL)
        if is_cached:
            print(f"📦 Phi-2 found in cache - loading from disk (no download needed)")
        else:
            print(f"⬇️  Phi-2 not in cache - will download (~5GB, first time only)")
        
        for attempt in range(max_retries):
            try:
                print(f"Loading Phi-2 (attempt {attempt + 1}/{max_retries})...")
                
                self.tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
                
                if is_cached:
                    print("💾 Loading Phi-2 weights from cache into memory (this may take 2-3 minutes)...")
                    print("   (This is normal - models must be loaded into RAM each time you run the app)")
                else:
                    print("⬇️  Downloading and loading Phi-2 weights (~5GB, this may take 15-30 minutes)...")
                    print("⚠️  WARNING: Requires ~5GB free disk space")
                self.model = AutoModelForCausalLM.from_pretrained(
                    MODEL,
                    torch_dtype=torch.float16,
                    device_map="cpu",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                
                print("✓ Phi-2 loaded successfully!")
                return
                
            except (OSError, RuntimeError) as e:
                error_str = str(e).lower()
                if "no space" in error_str or "disk" in error_str or "space" in error_str:
                    raise RuntimeError(f"Insufficient disk space to load Phi-2 (need ~5GB): {e}")
                elif attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(3)
                else:
                    raise RuntimeError(f"Failed to load Phi-2 after {max_retries} attempts: {e}")
            except KeyboardInterrupt:
                raise RuntimeError("Phi-2 loading cancelled by user")
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(3)
                else:
                    raise RuntimeError(f"Failed to load Phi-2 after {max_retries} attempts: {e}")
    
    def generate(self, prompt: str, max_tokens: int = 200, **kwargs) -> str:
        """Generate question using Phi-2"""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded")
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=False,  # Disable cache to avoid compatibility issues
                top_p=0.9,
                repetition_penalty=1.1,
            )
        
        generated_ids = outputs[0][input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return generated_text.strip()
    
    def get_model_info(self) -> Dict[str, Any]:
        info = super().get_model_info()
        info.update({
            "type": "Local LLM",
            "parameters": "2.7B",
            "precision": "float16",
            "device": "CPU"
        })
        return info


def get_model(model_name: str, api_key: Optional[str] = None) -> BaseQuestionGenerator:
    """Factory function to get model instance"""
    model_lower = model_name.lower()
    
    if model_lower in ["phi3", "phi-3"]:
        return Phi3Generator()
    elif model_lower == "gemini":
        if not api_key:
            raise ValueError("API key required for Gemini")
        return GeminiGenerator(api_key)
    elif model_lower == "baseline":
        return BaselineGenerator()
    elif model_lower in ["tinylama", "tinyllama", "tiny-llama"]:
        return TinyLlamaGenerator()
    elif model_lower in ["phi2", "phi-2"]:
        return Phi2Generator()
    elif model_lower in ["phi1", "phi-1"]:
        # Phi-1 is deprecated, use Phi-2 instead
        print("⚠️  Phi-1 is deprecated. Using Phi-2 instead.")
        return Phi2Generator()
    else:
        raise ValueError(f"Unknown model: {model_name}. Available: phi3, gemini, baseline, tinylama, phi2")
