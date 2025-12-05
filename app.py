# app.py - FINAL 100% WORKING VERSION (Windows CPU, Torch 2.3.0, No Errors)
import os
import torch
import time
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFacePipeline
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import LLMResult, Generation
from typing import List, Optional, Any
import gradio as gr

# Disable Xet storage warnings and set longer timeouts
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"  # 5 minutes timeout

# Phi-3-mini (Best & Fastest on CPU)
MODEL = "microsoft/Phi-3-mini-4k-instruct"

def load_model_with_retry(max_retries=3, retry_delay=5):
    """Load model with retry logic for network issues"""
    for attempt in range(max_retries):
        try:
            print(f"Loading Phi-3-mini (attempt {attempt + 1}/{max_retries})...")
            if attempt == 0:
                print("Note: First download is ~7.6GB and may take 10-30 minutes depending on internet speed.")
                print("If download fails, it will resume automatically on next run.")
            
            print("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL, 
                trust_remote_code=True
            )
            print("✓ Tokenizer loaded!")
            
            print("\n" + "-"*60)
            print("LOADING MODEL WEIGHTS INTO MEMORY")
            print("-"*60)
            print("⚠️  WARNING: This requires ~8-10GB free RAM")
            print("⚠️  This step can take 2-5 minutes - DO NOT CLOSE THE WINDOW")
            print("⚠️  The process may appear frozen, but it's working...")
            print("-"*60 + "\n")
            
            # Try float16 first (uses half the memory), fallback to float32 if needed
            try:
                print("Attempting to load with float16 (3.8GB RAM)...")
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL, 
                    torch_dtype=torch.float16,  # Half precision - uses half the memory!
                    device_map="cpu", 
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                print("✓ Loaded with float16 (lower memory usage)")
            except (RuntimeError, OSError) as e:
                error_str = str(e).lower()
                if "out of memory" in error_str or "memory" in error_str or "cuda" in error_str:
                    print("\n✗ OUT OF MEMORY ERROR!")
                    print("The model is too large for your available RAM.")
                    print("\nTrying float32 instead (slower but more compatible)...")
                    try:
                        model = AutoModelForCausalLM.from_pretrained(
                            MODEL, 
                            torch_dtype=torch.float32,
                            device_map="cpu", 
                            trust_remote_code=True,
                            low_cpu_mem_usage=True
                        )
                        print("✓ Loaded with float32")
                    except Exception as e2:
                        print(f"\n✗ Failed even with float32: {e2}")
                        print("\nSOLUTIONS:")
                        print("1. Close other applications (browser, etc.) to free RAM")
                        print("2. Restart your computer")
                        print("3. You need at least 8-10GB free RAM for this model")
                        print("4. Consider using a smaller model")
                        raise RuntimeError(f"Model loading failed: {e2}") from e2
                else:
                    # Some other error, try float32
                    print(f"Float16 failed ({e}), trying float32...")
                    model = AutoModelForCausalLM.from_pretrained(
                        MODEL, 
                        torch_dtype=torch.float32,
                        device_map="cpu", 
                        trust_remote_code=True,
                        low_cpu_mem_usage=True
                    )
            
            print("✓ Model loaded successfully!")
            return tokenizer, model
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"✗ Attempt {attempt + 1} failed: {str(e)}")
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"\n✗ Failed to load model after {max_retries} attempts.")
                print(f"Last error: {str(e)}")
                print("\nTroubleshooting tips:")
                print("1. Check your internet connection")
                print("2. Try again later (Hugging Face servers may be busy)")
                print("3. Download manually from: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct")
                print("4. Or use a different model by changing the MODEL variable")
                raise

try:
    print("\n" + "="*60)
    print("STARTING MODEL LOADING PROCESS")
    print("="*60 + "\n")
    
    tokenizer, model = load_model_with_retry()
    
    # Fix for Phi-3 compatibility: Patch DynamicCache to add seen_tokens attribute
    # This fixes the 'DynamicCache' object has no attribute 'seen_tokens' error
    print("Applying compatibility patch for Phi-3...")
    try:
        from transformers.cache_utils import DynamicCache
        
        # Store original __getattribute__ if it exists
        if not hasattr(DynamicCache, '_original_getattr'):
            DynamicCache._original_getattr = DynamicCache.__getattribute__
        
        def patched_getattribute(self, name):
            if name == 'seen_tokens':
                # Calculate seen_tokens from cache if it doesn't exist
                if not hasattr(self, '_seen_tokens'):
                    if hasattr(self, 'key_cache') and len(self.key_cache) > 0:
                        try:
                            # Get the sequence length from the cache
                            self._seen_tokens = self.key_cache[0].shape[2] if len(self.key_cache[0].shape) > 2 else 0
                        except:
                            self._seen_tokens = 0
                    else:
                        self._seen_tokens = 0
                return self._seen_tokens
            return DynamicCache._original_getattr(self, name)
        
        DynamicCache.__getattribute__ = patched_getattribute
        print("✓ Compatibility patch applied!")
    except Exception as e:
        print(f"Warning: Could not apply cache patch ({e}), trying alternative method...")
        # Alternative: Patch the model's method directly
        original_prepare = model.prepare_inputs_for_generation
        
        def patched_prepare(input_ids, past_key_values=None, **kwargs):
            if past_key_values is not None:
                if not hasattr(past_key_values, 'seen_tokens'):
                    try:
                        if hasattr(past_key_values, 'key_cache') and len(past_key_values.key_cache) > 0:
                            past_key_values.seen_tokens = past_key_values.key_cache[0].shape[2] if len(past_key_values.key_cache[0].shape) > 2 else 0
                        else:
                            past_key_values.seen_tokens = 0
                    except:
                        past_key_values.seen_tokens = 0
            return original_prepare(input_ids, past_key_values=past_key_values, **kwargs)
        
        model.prepare_inputs_for_generation = patched_prepare
        print("✓ Alternative patch applied!")
    
    print("\nCreating text generation pipeline...")
    
    # Create a custom LangChain LLM wrapper to avoid the seen_tokens issue
    # Using a simple wrapper class instead of BaseLLM to avoid Pydantic issues
    class FixedPhi3LLM:
        def __init__(self, model, tokenizer):
            self._model = model
            self._tokenizer = tokenizer
            self._model.eval()
            self.temperature = 0.7
        
        def invoke(self, prompt: str, max_new_tokens=200, **kwargs) -> str:
            """Invoke the LLM with a prompt - optimized for speed"""
            # Tokenize input
            inputs = self._tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"]
            
            # Optimize generation parameters for CPU speed
            # Use smaller max_new_tokens for faster generation
            # Use top_p instead of temperature for more focused generation
            with torch.no_grad():
                outputs = self._model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,  # Reduced from 512 for speed
                    temperature=kwargs.get("temperature", self.temperature),
                    do_sample=True,
                    pad_token_id=self._tokenizer.eos_token_id,
                    use_cache=False,  # Disable cache to avoid seen_tokens error
                    top_p=0.9,  # Nucleus sampling for faster convergence
                    repetition_penalty=1.1,  # Reduce repetition
                )
            
            # Decode only the new tokens (remove input)
            generated_ids = outputs[0][input_ids.shape[1]:]
            generated_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
            return generated_text.strip()
        
        def __call__(self, prompt: str, **kwargs) -> str:
            """Make it callable"""
            return self.invoke(prompt, **kwargs)
    
    # Create our fixed LLM
    llm = FixedPhi3LLM(model, tokenizer)
    print("✓ Pipeline created successfully!")
    
except KeyboardInterrupt:
    print("\n\n✗ Process interrupted by user (Ctrl+C)")
    print("Model loading was cancelled.")
    exit(1)
except MemoryError:
    print("\n\n✗ OUT OF MEMORY!")
    print("Your system ran out of RAM while loading the model.")
    print("\nSOLUTIONS:")
    print("1. Close all other applications")
    print("2. Restart your computer")
    print("3. You need 8-10GB free RAM minimum")
    print("4. Consider using a smaller model")
    exit(1)
except Exception as e:
    print(f"\n\n✗ FATAL ERROR: Could not load model")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    print("\nFull error details:")
    import traceback
    traceback.print_exc()
    print("\n" + "="*60)
    print("TROUBLESHOOTING:")
    print("1. Check if you have at least 8GB free RAM")
    print("2. Close other applications")
    print("3. Try restarting your computer")
    print("4. Check Windows Task Manager for memory usage")
    print("="*60)
    exit(1)

print("Loading embeddings model...")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
print("✓ Embeddings model loaded!")
retriever = None

def process_pdf(pdf_file):
    global retriever
    # Handle Gradio file input - it can be a string path or a file object
    if pdf_file is None:
        return "Error: No file uploaded. Please upload a PDF file."
    
    # Extract file path from Gradio file input
    if isinstance(pdf_file, str):
        file_path = pdf_file
    elif isinstance(pdf_file, (list, tuple)) and len(pdf_file) > 0:
        file_path = pdf_file[0] if isinstance(pdf_file[0], str) else pdf_file[0].name
    elif hasattr(pdf_file, 'name'):
        file_path = pdf_file.name
    else:
        return f"Error: Invalid file format. Received: {type(pdf_file)}"
    
    if not os.path.exists(file_path):
        return f"Error: File not found: {file_path}"
    
    try:
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(splits, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        return f"Loaded: {os.path.basename(file_path)} → {len(splits)} chunks ready!"
    except Exception as e:
        return f"Error processing PDF: {str(e)}"

def generate(qtype, num=5):
    if retriever is None:
        return "Please upload a PDF first!"
    
    try:
        template = open(f"prompts/{qtype}.txt", "r", encoding="utf-8").read()
    except FileNotFoundError:
        return f"Error: Prompt template '{qtype}.txt' not found in prompts/ directory."
    except Exception as e:
        return f"Error reading prompt template: {str(e)}"
    
    # Set max tokens based on question type for faster generation
    max_tokens_map = {
        "mcq": 150,      # MCQs are shorter - 150 tokens is enough
        "short": 200,    # Short answers need a bit more
        "long": 300      # Long answers need more tokens
    }
    max_tokens = max_tokens_map.get(qtype, 200)
    
    results = []
    total = int(num)
    
    # Retrieve context once (more efficient)
    try:
        docs = retriever.invoke("Extract important concepts")
        context = "\n\n".join([doc.page_content for doc in docs])[:3000]
    except Exception as e:
        return f"Error retrieving context: {str(e)}"
    
    for i in range(total):
        try:
            print(f"Generating question {i+1}/{total}...")  # Progress to console
            
            prompt = template.format(context=context)
            
            # Generate with optimized token limit
            output = llm.invoke(prompt, max_new_tokens=max_tokens)
            
            # Handle different response formats
            if isinstance(output, dict):
                output = output.get("text", str(output))
            elif hasattr(output, "content"):
                output = output.content
            elif isinstance(output, str):
                pass  # Already a string
            else:
                output = str(output)
            
            # Extract just the generated text (remove prompt if present)
            output_text = str(output).strip()
            results.append(output_text)
            
            # Yield progress (for Gradio streaming if needed)
            # For now, just append to results
            
        except Exception as e:
            error_msg = f"Error generating question {i+1}: {str(e)}"
            print(error_msg)  # Print to console for debugging
            results.append(f"[Error: {error_msg}]")
    
    return "\n\n".join([f"Question {i+1}:\n{r}\n" + "─"*50 for i, r in enumerate(results)])

# Gradio UI
with gr.Blocks(title="EduQuest-Lite - Final Working") as demo:
    gr.Markdown("# EduQuest-Lite — Automatic Question Generator")
    gr.Markdown("**Talha Khuram (22i-0790) & Ahmad Aqeel (22i-1134)** | GenAI Fall 2025")
    
    pdf = gr.File(label="Upload Your FAST Lecture PDF")
    process = gr.Button("Process PDF")
    status = gr.Textbox(label="Status")
    process.click(process_pdf, pdf, status)
    
    with gr.Row():
        qtype = gr.Radio(["mcq", "short", "long"], label="Type", value="mcq")
        num = gr.Slider(1, 10, value=5, step=1, label="Number of Questions")
    
    btn = gr.Button("Generate Questions", variant="primary")
    out = gr.Textbox(label="Generated Questions", lines=30)
    btn.click(generate, [qtype, num], out)

print("Launching EduQuest-Lite... Opening http://127.0.0.1:7860")
demo.launch(server_port=7860)