# 🚀 Quick Start Guide

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** If you don't want to use Phi-3 (saves time and RAM), you can skip it and only use Gemini.

## Step 2: Run the Application

### Option A: Use Refactored Version (Recommended)
```bash
python app_refactored.py
```

### Option B: Use Original Version
```bash
python app.py
```

## Step 3: Access the Interface

Open your browser and go to: `http://localhost:7860`

## Step 4: Generate Questions

1. Upload a PDF from the `pdfs/` folder
2. Click "Process PDF"
3. Select a model (Gemini is fastest, Phi-3 is local)
4. Select question type (MCQ, Short, Long)
5. Click "Generate Questions"

## 🐳 Docker Quick Start

```bash
# Build image
docker build -t eduquest .

# Run container
docker run -p 7860:7860 -v $(pwd)/pdfs:/app/pdfs eduquest
```

## 📊 Run Ablation Study

```bash
python ablation_study.py
```

This will test different hyperparameters and save results to `ablation_results.json`.

## ✅ What's Been Implemented

### Code Evaluation (95 marks) - COMPLETE ✅

1. ✅ **Dataset (5 marks)**: `data_analysis.py` - loading, preprocessing, visualizations
2. ✅ **Model Implementation (15 marks)**: `models.py` - Phi-3, Gemini, Baseline
3. ✅ **Model Evaluation (15 marks)**: `evaluation.py` - BLEU, ROUGE, quality metrics
4. ✅ **Prompt Engineering (10 marks)**: `prompts/` folder with structured prompts
5. ✅ **Code Quality (10 marks)**: Modular classes, documentation, README
6. ✅ **Deployment (10 marks)**: `Dockerfile`, `docker-compose.yml`
7. ✅ **Modern Tools (10 marks)**: LangChain, Chroma, Docker, GitHub ready
8. ✅ **Bonus (20 marks)**: Model comparison, ablation study framework

## 📁 Files Created

- `app_refactored.py` - Main application with all features
- `models.py` - Model implementations (Phi-3, Gemini, Baseline)
- `evaluation.py` - Evaluation metrics
- `data_analysis.py` - Data processing and visualization
- `ablation_study.py` - Ablation study framework
- `Dockerfile` - Docker configuration
- `docker-compose.yml` - Docker Compose config
- `README.md` - Comprehensive documentation
- `QUICK_START.md` - This file

## 🎯 Next Steps for Report

1. **Run the system** and generate questions
2. **Run ablation study** to get hyperparameter analysis
3. **Take screenshots** of the interface and results
4. **Document findings** in your research paper
5. **Compare models** using the comparison feature
6. **Generate visualizations** using data analysis tab

## 💡 Tips

- **For faster testing**: Use Gemini only (skip Phi-3 initialization)
- **For local testing**: Use Phi-3 (no API needed)
- **For comparison**: Use both models
- **For ablation study**: Run `ablation_study.py` separately

## 🐛 Troubleshooting

**Import errors?**
```bash
pip install google-generativeai langchain-google-genai
```

**Phi-3 too slow?**
- Edit `app_refactored.py` and set `use_phi3=False`
- Use Gemini only for faster results

**Docker issues?**
- Make sure Docker is running
- Check port 7860 is available
- Try: `docker-compose up --build`
