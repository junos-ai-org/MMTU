"""
Unified evaluation runner for LLaDA and QWEN on MMTU.

Usage:
    # Run LLaDA evaluation
    python run_evaluation.py --model llada --input verification_sample.jsonl --output results/llada_verify.jsonl

    # Run QWEN evaluation
    python run_evaluation.py --model qwen --input verification_sample.jsonl --output results/qwen_verify.jsonl

LLaDA Reference: https://github.com/ML-GSAI/LLaDA
"""

import argparse
import json
import os
import time
from tqdm import tqdm
import pandas as pd

# Model implementations will be imported based on selection
LLADA_MODEL = None
LLADA_GENERATE_FN = None
QWEN_MODEL = None


def load_llada_model(model_path=None):
    """
    Load LLaDA model and generate function.

    LLaDA uses a standalone generate() function, not a model.generate() method.
    Reference: https://github.com/ML-GSAI/LLaDA/blob/main/generate.py
    """
    global LLADA_MODEL, LLADA_GENERATE_FN
    if LLADA_MODEL is not None:
        return LLADA_MODEL, LLADA_GENERATE_FN

    if model_path is None:
        model_path = os.environ.get("LLADA_MODEL_PATH", "GSAI-ML/LLaDA-8B-Instruct")

    print(f"Loading LLaDA model from {model_path}...")

    import torch
    from transformers import AutoModel, AutoTokenizer

    # Add LLaDA code path for the generate function
    import sys
    llada_code_path = os.environ.get("LLADA_CODE_PATH")
    if llada_code_path is None:
        # Check common locations
        candidates = [
            os.path.join(os.path.dirname(__file__), "llada"),  # submodule in MMTU
            "/workspace/LLaDA",  # Docker default
            "/workspace/MMTU/llada",  # Docker submodule path
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                llada_code_path = candidate
                break

    if llada_code_path and os.path.exists(llada_code_path) and llada_code_path not in sys.path:
        sys.path.insert(0, llada_code_path)
        print(f"Added LLaDA code path: {llada_code_path}")

    # Import the generate function from LLaDA repo
    try:
        from generate import generate as llada_generate
        print("Loaded generate() from local LLaDA repository")
    except ImportError:
        # Fallback: define generate inline (copied from ML-GSAI/LLaDA)
        print("Warning: Could not import generate from LLaDA repo, using inline implementation")
        llada_generate = _llada_generate_fallback

    # Load model - LLaDA uses AutoModel, not AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).cuda().eval()

    LLADA_MODEL = (model, tokenizer)
    LLADA_GENERATE_FN = llada_generate
    print("LLaDA model loaded successfully!")
    return LLADA_MODEL, LLADA_GENERATE_FN


def _llada_generate_fallback(model, prompt, steps=128, gen_length=128, block_length=128,
                              temperature=0., cfg_scale=0., remasking='low_confidence',
                              mask_id=126336):
    """
    Fallback implementation of LLaDA generate function.
    Based on: https://github.com/ML-GSAI/LLaDA/blob/main/generate.py

    This is a simplified version - prefer using the actual generate.py from the repo.
    """
    import torch

    device = prompt.device
    batch_size = prompt.shape[0]
    prompt_length = prompt.shape[1]
    total_length = prompt_length + gen_length

    # Initialize with prompt + mask tokens
    x = torch.full((batch_size, total_length), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_length] = prompt

    # Track which positions are prompt (not to be modified)
    prompt_mask = torch.zeros(total_length, dtype=torch.bool, device=device)
    prompt_mask[:prompt_length] = True

    # Number of tokens to unmask per step
    num_to_unmask = gen_length // steps

    for step in range(steps):
        # Get model predictions
        with torch.no_grad():
            logits = model(x).logits

        # Only consider masked positions
        mask_positions = (x == mask_id)

        if not mask_positions.any():
            break

        # Get probabilities for masked positions
        probs = torch.softmax(logits / max(temperature, 1e-8), dim=-1)

        # Sample or argmax
        if temperature > 0:
            # Gumbel softmax sampling
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(probs) + 1e-8) + 1e-8)
            sampled = (torch.log(probs + 1e-8) + gumbel_noise * temperature).argmax(dim=-1)
        else:
            sampled = logits.argmax(dim=-1)

        # Get confidence scores
        confidence = probs.max(dim=-1).values
        confidence[~mask_positions] = float('inf')  # Don't remask unmasked positions

        # Determine how many to unmask this step
        n_masked = mask_positions.sum().item()
        n_unmask = min(num_to_unmask, n_masked)

        if n_unmask > 0:
            # Find top-k confident positions
            flat_conf = confidence.view(-1)
            _, top_indices = flat_conf.topk(n_unmask, largest=False)  # lowest confidence = inf (already unmasked)

            # Actually we want highest confidence among masked
            masked_conf = confidence.clone()
            masked_conf[~mask_positions] = -float('inf')
            _, top_indices = masked_conf.view(-1).topk(n_unmask, largest=True)

            # Unmask these positions
            x.view(-1)[top_indices] = sampled.view(-1)[top_indices]

    return x


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


def generate_llada(prompt_text, max_new_tokens=128, temperature=0.0, steps=128):
    """
    Generate response using LLaDA.

    Note: LLaDA performs best when steps == gen_length.
    The default of 128 steps for 128 tokens is optimal.
    """
    import torch

    (model, tokenizer), generate_fn = load_llada_model()

    # Apply chat template for instruct model
    messages = [{"role": "user", "content": prompt_text}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )

    # Tokenize
    input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids.cuda()
    prompt_length = input_ids.shape[1]

    # Generate using LLaDA's diffusion process
    # Key parameters:
    # - steps: number of diffusion iterations (optimal: steps == gen_length)
    # - gen_length: number of tokens to generate
    # - block_length: for semi-autoregressive generation (32 for chat, 128 for batch)
    # - temperature: 0 for greedy, >0 for sampling
    # - remasking: 'low_confidence' (default) or 'random'
    output = generate_fn(
        model,
        input_ids,
        steps=steps,
        gen_length=max_new_tokens,
        block_length=32,  # 32 for conversational, as in chat.py
        temperature=temperature,
        cfg_scale=0.,
        remasking='low_confidence',
    )

    # Decode only the generated part
    response = tokenizer.decode(
        output[0, prompt_length:],
        skip_special_tokens=True
    )

    return response.strip()


def generate_qwen(prompt_text, max_new_tokens=512, temperature=0.0):
    """Generate response using QWEN."""
    import torch

    model, tokenizer = load_qwen_model()

    # Apply chat template
    messages = [{"role": "user", "content": prompt_text}]
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
    return response.strip()


def run_evaluation(input_file, output_file, model_name, temperature=0.0, max_new_tokens=512, steps=128):
    """Run evaluation on input file."""
    print(f"\n{'='*60}")
    print(f"Running {model_name.upper()} evaluation")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Temperature: {temperature}")
    print(f"Max tokens: {max_new_tokens}")
    if model_name.lower() == "llada":
        print(f"Diffusion steps: {steps}")
    print(f"{'='*60}\n")

    # Load input data
    data = pd.read_json(input_file, lines=True)
    print(f"Loaded {len(data)} examples")

    # Select generation function
    if model_name.lower() == "llada":
        def generate_fn(prompt, max_new_tokens, temperature):
            return generate_llada(prompt, max_new_tokens, temperature, steps=steps)
    elif model_name.lower() == "qwen":
        generate_fn = generate_qwen
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Check for existing progress (for resumable runs)
    completed_indices = set()
    if os.path.exists(output_file):
        try:
            existing = pd.read_json(output_file, lines=True)
            # Use index to track completion
            completed_indices = set(range(len(existing)))
            print(f"Found {len(completed_indices)} completed examples, resuming...")
        except:
            pass

    # Create output directory
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)

    # Process examples
    errors = 0

    pbar = tqdm(data.iterrows(), total=len(data), desc=f"Evaluating with {model_name}")
    for idx, row in pbar:
        # Skip if already completed
        if idx in completed_indices:
            continue

        metadata = row['metadata']
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

            pbar.set_postfix({"time": f"{t1-t0:.1f}s", "errors": errors})

        except Exception as e:
            print(f"\nError on example {idx}: {e}")
            import traceback
            traceback.print_exc()
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

        # Write incrementally (append mode)
        with open(output_file, 'a') as f:
            f.write(json.dumps(result) + "\n")

    print(f"\n{'='*60}")
    print(f"Evaluation complete!")
    print(f"Total: {len(data)} examples")
    print(f"Errors: {errors}")
    print(f"Output saved to: {output_file}")
    print(f"{'='*60}")


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
    parser.add_argument("--max-tokens", type=int, default=128,
                        help="Maximum new tokens to generate (default: 128)")
    parser.add_argument("--steps", type=int, default=128,
                        help="LLaDA diffusion steps (default: 128, optimal when steps=max_tokens)")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to model weights (overrides environment variable)")
    parser.add_argument("--llada-code-path", type=str, default=None,
                        help="Path to LLaDA repository code (for generate.py)")

    args = parser.parse_args()

    # Set paths if provided
    if args.model_path:
        if args.model == "llada":
            os.environ["LLADA_MODEL_PATH"] = args.model_path
        else:
            os.environ["QWEN_MODEL_PATH"] = args.model_path

    if args.llada_code_path:
        os.environ["LLADA_CODE_PATH"] = args.llada_code_path

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
        max_new_tokens=args.max_tokens,
        steps=args.steps
    )

    return 0


if __name__ == "__main__":
    exit(main())
