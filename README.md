# Distributed GPT-2 Training with Parameter-Efficient Fine-Tuning

A distributed training framework that splits GPT-2 into client-server architecture with LoRA/DoRA parameter-efficient fine-tuning on the E2E NLG dataset, supporting true incremental training across multiple H100 GPUs.

## Architecture Overview
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CLIENT GPU 0  │    │   SERVER GPU 0  │    │   CLIENT GPU 1  │
│                 │    │                 │    │                 │
│  HEAD LAYERS    │◄──►│   BODY LAYERS   │◄──►│  HEAD LAYERS    │
│  • Embedding    │    │   • Middle 8    │    │  • Embedding    │
│  • First 2      │    │     Transformer │    │  • First 2      │
│    Transformer  │    │     Layers      │    │    Transformer  │
│                 │    │                 │    │                 │
│  TAIL LAYERS    │    │   SERVER GPU 1  │    │  TAIL LAYERS    │
│  • Last 2       │    │                 │    │  • Last 2       │
│    Transformer  │    │   BODY LAYERS   │    │    Transformer  │
│  • LM Head      │    │   • Middle 8    │    │  • LM Head      │
│                 │    │     Transformer │    │                 │
│  LoRA/DoRA      │    │     Layers      │    │  LoRA/DoRA      │
│  Adapters       │    │                 │    │  Adapters       │
└─────────────────┘    │   LoRA/DoRA     │    └─────────────────┘
                       │   Adapters      │
                       └─────────────────┘
```

## Key Features

-   **Model Splitting**: GPT-2 intelligently split across client-server architecture
    
-   **Parameter-Efficient Training**: LoRA with DoRA for minimal memory usage
    
-   **True Distributed Training**: Full utilization of multiple H100 GPUs with DDP
    
-  **E2E NLG Dataset**: Natural language generation task with automatic preprocessing
    
-   **Incremental Training**: Perfect state preservation (1+1=2 epochs equivalence)
    
-  **Automatic Evaluation**: BLEU and METEOR metrics using Hugging Face
    
-  **Load Balancing**: Intelligent distribution of client requests across servers
    
-   **Robust Communication**: HTTP/JSON-based client-server protocol with error handling

## Prerequisites

### Hardware Requirements

-   **Minimum**: 2x NVIDIA H100 GPUs (or equivalent 80GB+ VRAM GPUs)
    
-   **Recommended**: NVLINK connection between GPUs for optimal performance
    
-   **System**: CUDA 11.8+ compatible environment


## Quick Start Guide

### Step 1: Launch Distributed Server

` chmod +x server_launch.sh `
`./server_launch.sh` 


### Step 2: Initial Training (1 Epoch)

`chmod +x client_launch.sh`   
`./client_launch.sh`

### Step 3: Incremental Training (Next Session)


` chmod +x load_launch.sh `   
`./load_launch.sh`


