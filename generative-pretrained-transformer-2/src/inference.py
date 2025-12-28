import torch
import argparse
import sys
from pathlib import Path
from transformers import GPT2Tokenizer
from .models import GPT2Model
from .config import InferenceConfig, GPT2Config


class TextGenerator:
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        if 'config' in checkpoint:
            self.config = checkpoint['config']
        else:
            self.config = GPT2Config()
        
        self.model = GPT2Model(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Model loaded from {model_path}")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M")
    
    def generate_text(self, prompt: str, config: InferenceConfig):
        encoding = self.tokenizer(prompt, return_tensors='pt')
        input_ids = encoding['input_ids'].to(self.device)
        
        print(f"\nPrompt: {prompt}")
        print(f"Generated: ", end='', flush=True)
        
        generated_tokens = []
        
        if config.stream:
            for token in self.model.generate(
                input_ids,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_k=config.top_k,
                top_p=config.top_p,
                repetition_penalty=config.repetition_penalty,
                do_sample=config.do_sample
            ):
                token_id = token.item()
                token_text = self.tokenizer.decode([token_id])
                print(token_text, end='', flush=True)
                generated_tokens.append(token_id)
                
                if token_id == self.tokenizer.eos_token_id:
                    break
        else:
            output_ids = None
            for token in self.model.generate(
                input_ids,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_k=config.top_k,
                top_p=config.top_p,
                repetition_penalty=config.repetition_penalty,
                do_sample=config.do_sample
            ):
                output_ids = torch.cat([input_ids, token], dim=1)
                input_ids = output_ids
            
            if output_ids is not None:
                generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                print(generated_text[len(prompt):])
        
        print("\n")
    
    def interactive_mode(self, config: InferenceConfig):
        print("\n" + "=" * 50)
        print("GPT-2 Interactive Text Generation")
        print("=" * 50)
        print("Type 'quit' or 'exit' to stop")
        print(f"Settings: temp={config.temperature}, top_k={config.top_k}, top_p={config.top_p}")
        print("=" * 50 + "\n")
        
        while True:
            try:
                prompt = input("Enter prompt: ").strip()
                
                if prompt.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not prompt:
                    print("Please enter a valid prompt.")
                    continue
                
                self.generate_text(prompt, config)
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description='GPT-2 Text Generation')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--prompt', type=str, default=None, help='Text prompt for generation')
    parser.add_argument('--max_new_tokens', type=int, default=100, help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling')
    parser.add_argument('--top_p', type=float, default=0.95, help='Top-p (nucleus) sampling')
    parser.add_argument('--repetition_penalty', type=float, default=1.2, help='Repetition penalty')
    parser.add_argument('--no_sample', action='store_true', help='Use greedy decoding')
    parser.add_argument('--no_stream', action='store_true', help='Disable streaming output')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    
    args = parser.parse_args()
    
    if not Path(args.model_path).exists():
        print(f"Error: Model file not found at {args.model_path}")
        sys.exit(1)
    
    generator = TextGenerator(args.model_path, args.device)
    
    inference_config = InferenceConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        do_sample=not args.no_sample,
        stream=not args.no_stream
    )
    
    if args.interactive:
        generator.interactive_mode(inference_config)
    elif args.prompt:
        generator.generate_text(args.prompt, inference_config)
    else:
        print("Error: Please provide either --prompt or use --interactive mode")
        sys.exit(1)


if __name__ == '__main__':
    main()
