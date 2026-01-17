"""
Unified evaluation runner for LLaDA and QWEN on MMTU.

Usage:
    # Run LLaDA evaluation
    python run_evaluation.py --model llada --input verification_sample.jsonl --output results/llada_verify.jsonl

    # Run QWEN evaluation
    python run_evaluation.py --model qwen --input verification_sample.jsonl --output results/qwen_verify.jsonl
"""

import argparse
import json
import os
import time
from tqdm import tqdm
import pandas as pd

# Model implementations will be imported based on selection
LLADA_MODEL = None
QWEN_MODEL = None


def load_llada_model(model_path=None):
    """Load LLaDA model."""
    global LLADA_MODEL
    if LLADA_MODEL is not None:
        return LLADA_MODEL

    # Try environment variable first
    if model_path is None:
        model_path = os.environ.get("LLADA_MODEL_PATH", "/models/llada-8b-instruct")

    print(f"Loading LLaDA model from {model_path}...")

    try:
        # Import LLaDA-specific modules
        import torch
        from transformers import AutoTokenizer

        # Adjust this import based on actual LLaDA repository structure
        import sys
        llada_path = os.environ.get("LLADA_CODE_PATH", "/workspace/LLaDA")
        if llada_path not in sys.path:
            sys.path.insert(0, llada_path)

        # Try different import patterns based on LLaDA version
        try:
            from models.llada import LLaDAModel
            model_class = LLaDAModel
        except ImportError:
            try:
                from llada import LLaDAForCausalLM
                model_class = LLaDAForCausalLM
            except ImportError:
                # Fallback: use AutoModel with trust_remote_code
                from transformers import AutoModelForCausalLM
                model_class = AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = model_class.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()

        LLADA_MODEL = (model, tokenizer)
        print("LLaDA model loaded successfully!")
        return LLADA_MODEL

    except Exception as e:
        print(f"Error loading LLaDA model: {e}")
        print("Make sure LLaDA code is at LLADA_CODE_PATH and model weights at LLADA_MODEL_PATH")
        raise


def load_qwen_model(model_path=None):
    """Load QWEN model."""
    global QWEN_MODEL
    if QWEN_MODEL is not None:
        return QWEN_MODEL

    if model_path is None:
        model_path = os.environ.get("QWEN_MODEL_PATH", "Qwen/Qwen2-7B-Instruct")

    print(f"Loading QWEN model from {model_path}...")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    QWEN_MODEL = (model, tokenizer)
    print("QWEN model loaded successfully!")
    return QWEN_MODEL


def generate_llada(prompt, max_new_tokens=512, temperature=0.0, steps=64):
    """Generate response using LLaDA."""
    import torch

    model, tokenizer = load_llada_model()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        # LLaDA diffusion generation
        # NOTE: Adjust parameters based on actual LLaDA API
        try:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_diffusion_steps=steps,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=temperature > 0,
            )
        except TypeError:
            # Fallback if diffusion steps not supported in this version
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=temperature > 0,
            )

    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )
    return response


def generate_qwen(prompt, max_new_tokens=512, temperature=0.0):
    """Generate response using QWEN."""
    import torch

    model, tokenizer = load_qwen_model()

    # Apply chat template
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else 1.0,
            do_sample=temperature > 0,
        )

    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )
    return response


def run_evaluation(input_file, output_file, model_name, temperature=0.0, max_new_tokens=512):
    """Run evaluation on input file."""
    print(f"\n{'='*60}")
    print(f"Running {model_name} evaluation")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"{'='*60}\n")

    # Load input data
    data = pd.read_json(input_file, lines=True)
    print(f"Loaded {len(data)} examples")

    # Select generation function
    if model_name.lower() == "llada":
        generate_fn = generate_llada
    elif model_name.lower() == "qwen":
        generate_fn = generate_qwen
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Check for existing progress
    completed = set()
    if os.path.exists(output_file):
        existing = pd.read_json(output_file, lines=True)
        completed = set(existing['metadata'].tolist())
        print(f"Found {len(completed)} completed examples, resuming...")

    # Create output directory
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)

    # Process examples
    results = []
    errors = 0

    for idx, row in tqdm(data.iterrows(), total=len(data), desc=f"Evaluating with {model_name}"):
        metadata = row['metadata']

        # Skip if already completed
        if metadata in completed:
            continue

        prompt = row['prompt']

        try:
            t0 = time.time()
            response = generate_fn(prompt, max_new_tokens=max_new_tokens, temperature=temperature)
            t1 = time.time()

            result = {
                "prompt": prompt,
                "metadata": metadata,
                "response": response,
                "prompt_tokens": None,
                "completion_tokens": None,
                "time_taken": t1 - t0,
                "model_name": model_name
            }

        except Exception as e:
            print(f"\nError on example {idx}: {e}")
            errors += 1
            result = {
                "prompt": prompt,
                "metadata": metadata,
                "response": f"Error: {str(e)}",
                "prompt_tokens": None,
                "completion_tokens": None,
                "time_taken": None,
                "model_name": model_name
            }

        results.append(result)

        # Write incrementally
        with open(output_file, 'a') as f:
            f.write(json.dumps(result) + "\n")

    print(f"\n{'='*60}")
    print(f"Evaluation complete!")
    print(f"Total: {len(data)} examples")
    print(f"Errors: {errors}")
    print(f"Output saved to: {output_file}")
    print(f"{'='*60}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run MMTU evaluation with LLaDA or QWEN")

    parser.add_argument("--model", "-m", required=True, choices=["llada", "qwen"],
                        help="Model to use for evaluation")
    parser.add_argument("--input", "-i", required=True,
                        help="Input JSONL file with prompts")
    parser.add_argument("--output", "-o", required=True,
                        help="Output JSONL file for results")
    parser.add_argument("--temperature", "-t", type=float, default=0.0,
                        help="Sampling temperature (default: 0.0 for greedy)")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Maximum new tokens to generate (default: 512)")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to model weights (overrides environment variable)")

    args = parser.parse_args()

    # Set model path if provided
    if args.model_path:
        if args.model == "llada":
            os.environ["LLADA_MODEL_PATH"] = args.model_path
        else:
            os.environ["QWEN_MODEL_PATH"] = args.model_path

    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1

    # Run evaluation
    run_evaluation(
        input_file=args.input,
        output_file=args.output,
        model_name=args.model,
        temperature=args.temperature,
        max_new_tokens=args.max_tokens
    )

    return 0


if __name__ == "__main__":
    exit(main())
