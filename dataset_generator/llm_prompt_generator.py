import json
import logging
import time
from typing import Optional, Dict, Any
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError


class LLMPromptGenerator:
    """Generates natural user prompts from code using an external LLM."""

    def __init__(self, server_url: str = "http://localhost:11434", model: str="qwen2.5-coder:14b"):
        self.server_url = server_url.rstrip('/')
        self.model = model
        self.api_endpoint = f"{self.server_url}/api/generate"
        self.logger = logging.getLogger(__name__)

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests

        # Prompt template for generating user prompts
        self.system_prompt = """
            You are a helpful coding assistant. You are 
            given lines of code and need to understand their purpose.
            Your task is to write a prompt that would generate the code above. 
            The prompt should be:
            - Not too long (1-2 sentences)
            - Look like it was written by a human developer
            - Be simple and direct
            - Focus on what the code does, not implementation details
            - Sound natural and conversational

            Answer only with the prompt, nothing else."""

    def _rate_limit(self):
        """Implement simple rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()

    def _clean_code_for_prompt(self, code: str) -> str:
        """Clean code to make it suitable for prompt generation."""
        # Remove excessive whitespace
        lines = [line.rstrip() for line in code.split('\n')]
        # Remove empty lines at start and end
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()

        # Limit code length for better prompt generation
        if len(lines) > 30:
            lines = lines[:30] + ["    # ... (truncated)"]

        return '\n'.join(lines)

    def _create_generation_request(self, code: str) -> Dict[str, Any]:
        """Create the request payload for LLM prompt generation."""
        cleaned_code = self._clean_code_for_prompt(code)

        full_prompt = f"""
            {self.system_prompt}
            Code:
            ```
                {cleaned_code}
            ```
            Generate a user prompt for this code:"""

        return {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "format": "json",
            "options": {
                # "temperature": 0.3,
                # "top_p": 0.8,
                # "max_tokens": 150,
                "stop": ["\n\n", "```"]  # Stop at double newline or code blocks
            }
        }

    def _extract_prompt_from_response(self, response_text: str) -> Optional[str]:
        """Extract the generated prompt from LLM response."""
        try:
            # Try to parse as JSON first
            if response_text.strip().startswith('{'):
                data = json.loads(response_text)
                response_content = data.get('response', '').strip()
            else:
                response_content = response_text.strip()
            
            # Clean the response
            response_content = response_content.strip('"').strip("'").strip()
            
            # Remove common prefixes that might be added by the LLM
            prefixes_to_remove = [
                "Here's a prompt:",
                "Prompt:",
                "User prompt:",
                "The prompt would be:",
                "A suitable prompt would be:",
                "Here is a prompt:",
            ]

            for prefix in prefixes_to_remove:
                if response_content.lower().startswith(prefix.lower()):
                    response_content = response_content[len(prefix):].strip()

            # Validate the prompt
            if len(response_content) < 10 or len(response_content) > 500:
                return None

            # Ensure it ends with proper punctuation
            if not response_content.endswith(('.', '!', '?')):
                response_content += '.'

            return response_content

        except (json.JSONDecodeError, KeyError, AttributeError) as e:
            self.logger.warning(f"Failed to parse LLM response: {e}")
            return None

    def generate_prompt(self, code: str, fallback_instruction: str = None) -> str:
        """
        Generate a natural user prompt for the given code.
        
        Args:
            code: The code to generate a prompt for
            fallback_instruction: Fallback instruction if LLM generation fails
            
        Returns:
            Generated prompt or fallback instruction
        """
        if not code or not code.strip():
            return fallback_instruction or "Write some code."
        
        try:
            # Rate limiting
            # self._rate_limit()
            
            # Create request
            request_data = self._create_generation_request(code)
            
            # Make request with timeout
            self.logger.debug(f"Making LLM request to {self.api_endpoint}")
            response = requests.post(
                self.api_endpoint,
                json=request_data,
                timeout=180,
                headers={'Content-Type': 'application/json'}
            )
            
            response.raise_for_status()
            
            # Extract prompt from response
            generated_prompt = self._extract_prompt_from_response(response.text)
            
            if generated_prompt:
                self.logger.debug(f"Generated prompt: {generated_prompt[:100]}...")
                return generated_prompt
            else:
                self.logger.warning("Failed to extract valid prompt from LLM response")
                return fallback_instruction or self._generate_fallback_prompt(code)
                
        except (ConnectionError, Timeout) as e:
            self.logger.warning(f"Connection error with LLM server: {e}")
            return fallback_instruction or self._generate_fallback_prompt(code)
            
        except RequestException as e:
            self.logger.warning(f"Request failed: {e}")
            return fallback_instruction or self._generate_fallback_prompt(code)
            
        except Exception as e:
            self.logger.error(f"Unexpected error generating prompt: {e}")
            return fallback_instruction or self._generate_fallback_prompt(code)
    
    def _generate_fallback_prompt(self, code: str) -> str:
        """Generate a simple fallback prompt based on code analysis."""
        code_lower = code.lower()
        
        # Simple heuristics for fallback prompts
        if 'class ' in code_lower:
            if 'def __init__' in code_lower:
                return "Create a class with proper initialization and methods."
            else:
                return "Write a class with the necessary methods."
        elif 'def ' in code_lower:
            if 'return ' in code_lower:
                return "Write a function that processes input and returns a result."
            else:
                return "Create a function to handle the required functionality."
        elif 'import ' in code_lower or 'from ' in code_lower:
            return "Write code with the necessary imports and implementation."
        elif 'if __name__' in code_lower:
            return "Create a script with a main execution block."
        elif 'try:' in code_lower or 'except' in code_lower:
            return "Write code with proper error handling."
        else:
            return "Write the necessary code implementation."
    
    def test_connection(self) -> bool:
        """Test if the LLM server is accessible."""
        try:
            # Try a simple health check or minimal request
            test_request = {
                "model": self.model,
                "prompt": "Hello",
                "stream": False,
                "options": {"max_tokens": 1}
            }
            
            response = requests.post(
                self.api_endpoint,
                json=test_request,
                timeout=180,
                headers={'Content-Type': 'application/json'}
            )
            
            return response.status_code == 200
            
        except Exception as e:
            self.logger.warning(f"LLM server connection test failed: {e}")
            return False
    
    def generate_batch_prompts(self, code_examples: list, max_concurrent: int = 3) -> list:
        """
        Generate prompts for multiple code examples with limited concurrency.
        
        Args:
            code_examples: List of (code, fallback_instruction) tuples
            max_concurrent: Maximum concurrent requests
            
        Returns:
            List of generated prompts
        """
        import concurrent.futures
        import threading
        
        # Thread-safe counter for rate limiting
        self._request_lock = threading.Lock()
        
        def generate_single(code_and_fallback):
            code, fallback = code_and_fallback
            with self._request_lock:
                return self.generate_prompt(code, fallback)
        
        # Use ThreadPoolExecutor for concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = [executor.submit(generate_single, example) for example in code_examples]
            results = []
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Error in batch prompt generation: {e}")
                    results.append("Write the required code implementation.")
        
        return results