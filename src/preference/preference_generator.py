"""
Preference Data Generator for DPO Training
Uses OpenAssistant Reward Model to score responses and create chosen/rejected pairs
"""

import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import random


@dataclass
class PreferencePair:
    """A preference pair with chosen and rejected responses."""
    instruction: str
    chosen: str
    rejected: str
    chosen_score: float
    rejected_score: float
    margin: float


class PreferenceGenerator:
    """Generate preference pairs using reward model scoring."""

    def __init__(self, model, tokenizer, config: Optional[Dict] = None):
        """
        Initialize preference generator.

        Args:
            model: Reward model (OpenAssistant)
            tokenizer: Tokenizer for the reward model
            config: Configuration dictionary
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or {}

        # Generation settings
        self.num_responses_per_instruction = self.config.get('num_responses_per_instruction', 3)
        self.min_score_margin = self.config.get('min_score_margin', 0.5)
        self.temperature = self.config.get('temperature', 0.8)
        self.max_new_tokens = self.config.get('max_new_tokens', 512)

        # Llama 3.1 chat template
        self.instruction_template = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        self.response_template = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    def score_response(self, instruction: str, response: str) -> float:
        """
        Score a response using the reward model.

        Args:
            instruction: The instruction/prompt
            response: The response to score

        Returns:
            Reward score (higher is better)
        """
        # Format as conversation
        text = f"{self.instruction_template}{instruction}{self.response_template}{response}"

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)

        # Get reward score
        with torch.no_grad():
            outputs = self.model(**inputs)
            # OpenAssistant reward model outputs logits, take the score
            score = outputs.logits[0].item() if hasattr(outputs.logits, 'item') else outputs.logits[0][0].item()

        return score

    def generate_response_variant(self, generator_model, generator_tokenizer,
                                 instruction: str, temperature: float) -> Optional[str]:
        """
        Generate a response variant with specified temperature.

        Args:
            generator_model: Model for generating responses (e.g., Llama-3.1-8B)
            generator_tokenizer: Tokenizer for generator
            instruction: The instruction
            temperature: Sampling temperature

        Returns:
            Generated response or None if failed
        """
        prompt = f"{self.instruction_template}{instruction}{self.response_template}"

        inputs = generator_tokenizer(
            prompt,
            return_tensors="pt"
        ).to(generator_model.device)

        try:
            with torch.no_grad():
                outputs = generator_model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=generator_tokenizer.eos_token_id,
                    eos_token_id=[
                        generator_tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                        generator_tokenizer.eos_token_id
                    ]
                )

            generated = generator_tokenizer.decode(outputs[0], skip_special_tokens=False)
            response = self._parse_response(generated)

            return response if response and len(response) > 10 else None

        except Exception as e:
            print(f"Generation error: {e}")
            return None

    def _parse_response(self, text: str) -> Optional[str]:
        """Extract response from generated text."""
        try:
            if "<|start_header_id|>assistant<|end_header_id|>" in text:
                parts = text.split("<|start_header_id|>assistant<|end_header_id|>")
                if len(parts) > 1:
                    response = parts[-1]
                    for end_token in ["<|eot_id|>", "<|end_of_text|>"]:
                        if end_token in response:
                            response = response.split(end_token)[0]
                    return response.strip()
        except Exception as e:
            print(f"Parse error: {e}")
        return None

    def create_preference_pair(self, generator_model, generator_tokenizer,
                              instruction: str, original_response: Optional[str] = None) -> Optional[PreferencePair]:
        """
        Create a preference pair by generating multiple responses and scoring them.

        Args:
            generator_model: Model for generating response variants
            generator_tokenizer: Tokenizer for generator
            instruction: The instruction
            original_response: Optional original response to include

        Returns:
            PreferencePair or None if failed
        """
        responses = []

        # Include original response if provided
        if original_response:
            responses.append(original_response)

        # Generate additional response variants with different temperatures
        temperatures = [0.6, 0.8, 1.0, 1.2][:self.num_responses_per_instruction]

        for temp in temperatures:
            if len(responses) >= self.num_responses_per_instruction:
                break

            response = self.generate_response_variant(
                generator_model,
                generator_tokenizer,
                instruction,
                temp
            )

            if response and response not in responses:
                responses.append(response)

        # Need at least 2 responses
        if len(responses) < 2:
            return None

        # Score all responses
        scored_responses = []
        for response in responses:
            try:
                score = self.score_response(instruction, response)
                scored_responses.append((response, score))
            except Exception as e:
                print(f"Scoring error: {e}")
                continue

        # Need at least 2 scored responses
        if len(scored_responses) < 2:
            return None

        # Sort by score (descending)
        scored_responses.sort(key=lambda x: x[1], reverse=True)

        # Take best as chosen, worst as rejected
        chosen, chosen_score = scored_responses[0]
        rejected, rejected_score = scored_responses[-1]

        margin = chosen_score - rejected_score

        # Check minimum margin
        if margin < self.min_score_margin:
            return None

        return PreferencePair(
            instruction=instruction,
            chosen=chosen,
            rejected=rejected,
            chosen_score=chosen_score,
            rejected_score=rejected_score,
            margin=margin
        )

    def generate_preference_dataset(self, generator_model, generator_tokenizer,
                                   instructions: List[Dict], verbose: bool = False) -> List[Dict]:
        """
        Generate preference dataset from instruction data.

        Args:
            generator_model: Model for generating responses
            generator_tokenizer: Tokenizer for generator
            instructions: List of instruction-response dictionaries
            verbose: Whether to print progress

        Returns:
            List of preference pair dictionaries
        """
        preference_data = []
        failed_count = 0

        for i, sample in enumerate(instructions):
            instruction = sample['instruction']
            original_response = sample.get('response')

            try:
                pair = self.create_preference_pair(
                    generator_model,
                    generator_tokenizer,
                    instruction,
                    original_response
                )

                if pair:
                    preference_data.append({
                        'instruction': pair.instruction,
                        'chosen': pair.chosen,
                        'rejected': pair.rejected,
                        'chosen_score': pair.chosen_score,
                        'rejected_score': pair.rejected_score,
                        'margin': pair.margin
                    })
                    failed_count = 0
                else:
                    failed_count += 1

                if verbose and (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{len(instructions)}, Generated {len(preference_data)} pairs")

            except Exception as e:
                print(f"Error at sample {i}: {e}")
                failed_count += 1
                continue

            # Stop if too many consecutive failures
            if failed_count >= 50:
                print(f"Too many consecutive failures. Stopping.")
                break

        return preference_data

    def get_statistics(self, preference_data: List[Dict]) -> Dict:
        """Calculate statistics for preference dataset."""
        if not preference_data:
            return {}

        margins = [p['margin'] for p in preference_data]
        chosen_scores = [p['chosen_score'] for p in preference_data]
        rejected_scores = [p['rejected_score'] for p in preference_data]

        return {
            'total_pairs': len(preference_data),
            'margin': {
                'mean': sum(margins) / len(margins),
                'min': min(margins),
                'max': max(margins),
            },
            'chosen_score': {
                'mean': sum(chosen_scores) / len(chosen_scores),
                'min': min(chosen_scores),
                'max': max(chosen_scores),
            },
            'rejected_score': {
                'mean': sum(rejected_scores) / len(rejected_scores),
                'min': min(rejected_scores),
                'max': max(rejected_scores),
            }
        }
