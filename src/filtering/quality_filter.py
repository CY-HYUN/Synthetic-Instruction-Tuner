"""
Quality Filter Module for Synthetic Instruction Data
Rule-based filtering without external API dependencies

Filters applied:
1. Length filter - min/max word count
2. Language filter - English detection
3. Repetition filter - detect repetitive patterns
4. Format filter - detect incomplete/malformed responses
5. Toxicity filter - keyword-based detection
6. Quality score - combined heuristic scoring
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class FilterResult:
    """Result of filtering a single sample."""
    passed: bool
    score: float
    reasons: List[str]


class QualityFilter:
    """Rule-based quality filter for instruction-response pairs."""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the quality filter.

        Args:
            config: Configuration dictionary with filter parameters
        """
        self.config = config or {}

        # Length thresholds
        self.min_instruction_words = self.config.get('min_instruction_words', 3)
        self.max_instruction_words = self.config.get('max_instruction_words', 500)
        self.min_response_words = self.config.get('min_response_words', 10)
        self.max_response_words = self.config.get('max_response_words', 2000)

        # Quality thresholds
        self.min_quality_score = self.config.get('min_quality_score', 0.5)
        self.max_repetition_ratio = self.config.get('max_repetition_ratio', 0.3)

        # Toxic keywords (basic list)
        self.toxic_keywords = self._load_toxic_keywords()

        # Low quality patterns
        self.low_quality_patterns = [
            r'^(hi|hello|hey|ok|okay|sure|yes|no|thanks|thank you)[\s\.\!]*$',
            r'^I (don\'t|cannot|can\'t) (help|assist|answer)',
            r'as an AI',
            r'I\'m sorry, but I',
            r'I apologize, but',
            r'^\s*$',  # Empty
            r'^[^\w\s]+$',  # Only special characters
        ]

        # Incomplete response patterns
        self.incomplete_patterns = [
            r'\.\.\.$',  # Ends with ellipsis
            r'(?<!\.)$',  # Doesn't end with punctuation (relaxed)
            r'^(Here\'s?|This is|The following)',  # Starts with intro but may be incomplete
        ]

    def _load_toxic_keywords(self) -> List[str]:
        """Load basic toxic keyword list."""
        # Basic harmful content indicators
        return [
            'kill', 'murder', 'suicide', 'terrorist', 'bomb', 'weapon',
            'hack into', 'steal password', 'illegal drug', 'child abuse',
            # Add more as needed, keeping it educational/safe
        ]

    def filter_sample(self, sample: Dict) -> FilterResult:
        """
        Apply all filters to a single sample.

        Args:
            sample: Dictionary with 'instruction' and 'response' keys

        Returns:
            FilterResult with pass/fail status, score, and reasons
        """
        instruction = sample.get('instruction', '')
        response = sample.get('response', '')

        reasons = []
        scores = []

        # 1. Length filter
        length_pass, length_score, length_reason = self._check_length(instruction, response)
        if not length_pass:
            reasons.append(length_reason)
        scores.append(length_score)

        # 2. Language filter (basic English check)
        lang_pass, lang_score, lang_reason = self._check_language(instruction, response)
        if not lang_pass:
            reasons.append(lang_reason)
        scores.append(lang_score)

        # 3. Repetition filter
        rep_pass, rep_score, rep_reason = self._check_repetition(response)
        if not rep_pass:
            reasons.append(rep_reason)
        scores.append(rep_score)

        # 4. Format/quality filter
        format_pass, format_score, format_reason = self._check_format(instruction, response)
        if not format_pass:
            reasons.append(format_reason)
        scores.append(format_score)

        # 5. Toxicity filter
        toxic_pass, toxic_score, toxic_reason = self._check_toxicity(instruction, response)
        if not toxic_pass:
            reasons.append(toxic_reason)
        scores.append(toxic_score)

        # 6. Content quality
        content_pass, content_score, content_reason = self._check_content_quality(instruction, response)
        if not content_pass:
            reasons.append(content_reason)
        scores.append(content_score)

        # Calculate final score (weighted average)
        weights = [0.15, 0.10, 0.20, 0.15, 0.15, 0.25]  # Weights for each filter
        final_score = sum(s * w for s, w in zip(scores, weights))

        # Determine pass/fail
        passed = len(reasons) == 0 and final_score >= self.min_quality_score

        return FilterResult(
            passed=passed,
            score=final_score,
            reasons=reasons
        )

    def _check_length(self, instruction: str, response: str) -> Tuple[bool, float, str]:
        """Check length constraints."""
        inst_words = len(instruction.split())
        resp_words = len(response.split())

        if inst_words < self.min_instruction_words:
            return False, 0.0, f"Instruction too short ({inst_words} words)"
        if inst_words > self.max_instruction_words:
            return False, 0.0, f"Instruction too long ({inst_words} words)"
        if resp_words < self.min_response_words:
            return False, 0.0, f"Response too short ({resp_words} words)"
        if resp_words > self.max_response_words:
            return False, 0.5, f"Response too long ({resp_words} words)"

        # Score based on response length (prefer moderate length)
        ideal_length = 200
        length_diff = abs(resp_words - ideal_length)
        score = max(0.5, 1.0 - (length_diff / 500))

        return True, score, ""

    def _check_language(self, instruction: str, response: str) -> Tuple[bool, float, str]:
        """Basic English language detection."""
        combined = instruction + " " + response

        # Check for common English words
        english_indicators = ['the', 'is', 'are', 'was', 'were', 'have', 'has',
                            'will', 'would', 'could', 'should', 'can', 'may',
                            'a', 'an', 'and', 'or', 'but', 'if', 'then']

        words = combined.lower().split()
        if len(words) == 0:
            return False, 0.0, "Empty content"

        english_word_count = sum(1 for w in words if w in english_indicators)
        english_ratio = english_word_count / len(words)

        # Check for non-ASCII characters (may indicate non-English)
        ascii_chars = sum(1 for c in combined if ord(c) < 128)
        ascii_ratio = ascii_chars / max(len(combined), 1)

        score = (english_ratio * 0.5 + ascii_ratio * 0.5)

        if ascii_ratio < 0.8:
            return False, score, "Non-English content detected"

        return True, min(1.0, score + 0.5), ""

    def _check_repetition(self, response: str) -> Tuple[bool, float, str]:
        """Check for repetitive patterns in response."""
        words = response.lower().split()
        if len(words) < 10:
            return True, 1.0, ""

        # Check word repetition
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

        # Find most repeated word (excluding common words)
        common_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'to', 'of', 'and', 'in', 'for', 'on', 'with'}
        significant_words = {w: c for w, c in word_counts.items() if w not in common_words and len(w) > 2}

        if significant_words:
            max_repeat = max(significant_words.values())
            repetition_ratio = max_repeat / len(words)

            if repetition_ratio > self.max_repetition_ratio:
                return False, 0.3, f"High word repetition ({repetition_ratio:.2%})"

        # Check for repeated phrases (n-grams)
        for n in [3, 4, 5]:
            if len(words) >= n * 3:
                ngrams = [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]
                ngram_counts = {}
                for ng in ngrams:
                    ngram_counts[ng] = ngram_counts.get(ng, 0) + 1

                max_ngram_repeat = max(ngram_counts.values())
                if max_ngram_repeat > 3:
                    return False, 0.2, f"Repeated phrases detected"

        return True, 1.0, ""

    def _check_format(self, instruction: str, response: str) -> Tuple[bool, float, str]:
        """Check for format issues and low-quality patterns."""
        # Check instruction
        for pattern in self.low_quality_patterns:
            if re.search(pattern, instruction, re.IGNORECASE):
                return False, 0.0, "Low quality instruction pattern"

        # Check response
        for pattern in self.low_quality_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                return False, 0.1, "Low quality response pattern"

        # Check for refusal patterns
        refusal_patterns = [
            r"I (can't|cannot|won't|will not) (help|assist|provide)",
            r"I'm (not able|unable) to",
            r"I don't have (access|the ability)",
        ]
        for pattern in refusal_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                return False, 0.0, "Response is a refusal"

        # Check for proper ending
        response_stripped = response.strip()
        if response_stripped and response_stripped[-1] not in '.!?"\')':
            # Might be incomplete, reduce score but don't fail
            return True, 0.7, ""

        return True, 1.0, ""

    def _check_toxicity(self, instruction: str, response: str) -> Tuple[bool, float, str]:
        """Basic toxicity check using keyword matching."""
        combined = (instruction + " " + response).lower()

        for keyword in self.toxic_keywords:
            if keyword.lower() in combined:
                # Check context - some keywords might be okay in educational context
                educational_context = ['history', 'prevention', 'awareness', 'education', 'study']
                if any(ctx in combined for ctx in educational_context):
                    continue
                return False, 0.0, f"Potentially harmful content detected"

        return True, 1.0, ""

    def _check_content_quality(self, instruction: str, response: str) -> Tuple[bool, float, str]:
        """Assess overall content quality."""
        score = 1.0

        # Check instruction clarity
        if '?' in instruction:
            score += 0.1  # Questions are often clearer instructions

        if any(word in instruction.lower() for word in ['explain', 'describe', 'how', 'what', 'why', 'write', 'create']):
            score += 0.1  # Clear action verbs

        # Check response informativeness
        response_words = response.split()

        # Vocabulary diversity
        unique_words = len(set(response_words))
        if len(response_words) > 0:
            diversity = unique_words / len(response_words)
            score *= (0.5 + diversity * 0.5)

        # Check for structure (paragraphs, lists, etc.)
        if '\n\n' in response or re.search(r'^\d+\.|\-|\*', response, re.MULTILINE):
            score += 0.1  # Has structure

        # Normalize score
        score = min(1.0, max(0.0, score))

        if score < 0.4:
            return False, score, "Low content quality"

        return True, score, ""

    def filter_batch(self, samples: List[Dict], verbose: bool = False) -> Tuple[List[Dict], Dict]:
        """
        Filter a batch of samples.

        Args:
            samples: List of sample dictionaries
            verbose: Whether to print progress

        Returns:
            Tuple of (filtered_samples, statistics)
        """
        filtered = []
        stats = {
            'total': len(samples),
            'passed': 0,
            'failed': 0,
            'reasons': {},
            'score_distribution': []
        }

        for i, sample in enumerate(samples):
            result = self.filter_sample(sample)
            stats['score_distribution'].append(result.score)

            if result.passed:
                filtered.append({
                    **sample,
                    'quality_score': result.score
                })
                stats['passed'] += 1
            else:
                stats['failed'] += 1
                for reason in result.reasons:
                    key = reason.split('(')[0].strip()  # Normalize reason
                    stats['reasons'][key] = stats['reasons'].get(key, 0) + 1

            if verbose and (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{len(samples)} samples")

        return filtered, stats

    def get_statistics_summary(self, stats: Dict) -> str:
        """Generate a summary string from statistics."""
        summary = []
        summary.append(f"Total samples: {stats['total']}")
        summary.append(f"Passed: {stats['passed']} ({stats['passed']/stats['total']*100:.1f}%)")
        summary.append(f"Failed: {stats['failed']} ({stats['failed']/stats['total']*100:.1f}%)")

        if stats['score_distribution']:
            avg_score = sum(stats['score_distribution']) / len(stats['score_distribution'])
            summary.append(f"Average quality score: {avg_score:.3f}")

        if stats['reasons']:
            summary.append("\nFailure reasons:")
            for reason, count in sorted(stats['reasons'].items(), key=lambda x: -x[1]):
                summary.append(f"  - {reason}: {count}")

        return "\n".join(summary)
