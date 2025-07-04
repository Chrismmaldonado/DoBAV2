import os
import sys
import re
import json
import requests

# Try to import LM Studio API endpoint from DobAEI.py
try:
    from DobAEI import LMSTUDIO_API
except ImportError:
    # Default endpoints if import fails
    LMSTUDIO_API = "http://localhost:1234/v1/chat/completions"

def analyze_own_code(num_lines=100, file_path=None):
    """
    Analyze the first N lines of the AI's own code.

    Args:
        num_lines: Number of lines to analyze
        file_path: Optional specific file to analyze, otherwise analyzes the main file

    Returns:
        str: Analysis results
    """
    try:
        # If no specific file is provided, use the main file (DobAEI.py)
        if file_path is None:
            # Get the directory of the current script
            current_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(current_dir, "DobAEI.py")

        # Check if the file exists
        if not os.path.exists(file_path):
            return f"Error: File {file_path} does not exist."

        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Get the first N lines
        first_n_lines = lines[:num_lines]
        code_content = ''.join(first_n_lines)

        # Determine the language based on file extension
        _, ext = os.path.splitext(file_path)
        language = "Python" if ext.lower() == '.py' else "Unknown"

        # Perform a detailed manual analysis
        detailed_analysis = perform_detailed_analysis(first_n_lines)

        # First, create a basic analysis output
        output = f"📊 Analysis of the first {num_lines} lines of {os.path.basename(file_path)}:\n\n"
        output += f"File: {file_path}\n"
        output += f"Language: {language}\n"
        output += f"Lines analyzed: {num_lines} of {len(lines)} total lines\n\n"

        # Add code structure information from detailed analysis
        output += "📋 Code Structure:\n"

        # Add imports
        if detailed_analysis["imports"]:
            output += "  Imports:\n"
            for imp in detailed_analysis["imports"]:
                output += f"    - {imp}\n"

        # Add classes
        if detailed_analysis["classes"]:
            output += "  Classes:\n"
            for cls in detailed_analysis["classes"]:
                output += f"    - {cls}\n"

        # Add functions
        if detailed_analysis["functions"]:
            output += "  Functions:\n"
            for func in detailed_analysis["functions"]:
                output += f"    - {func}\n"

        # Add variables
        if detailed_analysis["variables"]:
            output += "  Key Variables:\n"
            for var in detailed_analysis["variables"][:10]:  # Limit to 10 variables
                output += f"    - {var}\n"

        # Add quality issues
        if detailed_analysis["quality_issues"]:
            output += "\n📈 Code Quality:\n"
            output += "  Manual quality analysis:\n"
            for issue in detailed_analysis["quality_issues"]:
                output += f"    - {issue}\n"

        # Add the first few lines of code for reference
        output += "\n📄 First 10 lines of code:\n"
        for i, line in enumerate(first_n_lines[:10], 1):
            output += f"{i}: {line}"

        # Now, send the code to LM Studio for deeper analysis
        try:
            print(f"📊 Preparing to send {num_lines} lines of code to LM Studio for analysis...")

            # Create a prompt for the LLM to analyze the code
            prompt = f"""
Please analyze the following Python code from {os.path.basename(file_path)}:

```python
{code_content}
```

Provide a detailed analysis including:
1. A summary of what this code does
2. Key components and their purpose
3. Any patterns or design principles used
4. Potential improvements or issues
5. Overall code quality assessment

Your analysis should be thorough and insightful. This is the AI's own code, so your analysis will help the AI understand itself better.
"""

            print(f"📊 Sending code to LM Studio for analysis...")

            # Send the code to LM Studio for analysis
            llm_analysis = send_to_lm_studio(prompt)

            if llm_analysis:
                # Add the LLM analysis to the output
                print(f"✅ Received analysis from LM Studio")
                output += "\n\n🧠 LLM Analysis:\n"
                output += llm_analysis
            else:
                print(f"⚠️ No analysis received from LM Studio")
                output += "\n\nNote: LM Studio did not provide an analysis. This could be because LM Studio is not running or is not properly configured."

            return output
        except Exception as e:
            # If LM Studio analysis fails, return the basic analysis
            print(f"❌ Error sending code to LM Studio: {str(e)}")
            output += "\n\nNote: LLM analysis via LM Studio failed. Showing basic analysis only."
            output += f"\nError: {str(e)}"
            return output

    except Exception as e:
        return f"Error analyzing code: {str(e)}"

def send_to_lm_studio(prompt):
    """
    Send a prompt to LM Studio for processing.

    Args:
        prompt: The prompt to send to LM Studio

    Returns:
        str: The response from LM Studio
    """
    try:
        # Use the imported LMSTUDIO_API endpoint
        # Also try the alternative port 5000 as a fallback
        endpoints = [
            LMSTUDIO_API,
            LMSTUDIO_API.replace("1234", "5000"),
            "http://localhost:11434/api/chat",  # Ollama API endpoint
            "http://127.0.0.1:1234/v1/chat/completions",  # Try 127.0.0.1 instead of localhost
            "http://127.0.0.1:5000/v1/chat/completions"   # Try 127.0.0.1 with port 5000
        ]

        # Try multiple models that might be available in LM Studio
        models = [
            "qwen2-vl-7b-instruct",
            "nous-hermes-2-mistral-7b-dpo",
            "qari-ocr-0.2.2.1-vl-2b-instruct",
            "llama3",
            "mistral",
            "gemma",
            "mixtral",
            "phi3",
            "default"
        ]

        # We'll try each model with each endpoint
        payload_template = {
            "messages": [
                {"role": "system", "content": "You are a code analysis expert. Analyze the provided code thoroughly and provide detailed insights."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 2000,
            "stream": False
        }

        # For Ollama-specific payload
        ollama_payload = {
            "model": "llama2",  # Default model for Ollama
            "messages": [
                {"role": "system", "content": "You are a code analysis expert. Analyze the provided code thoroughly and provide detailed insights."},
                {"role": "user", "content": prompt}
            ],
            "stream": False
        }

        response = None

        print(f"🔄 Attempting to send code to LLM for analysis...")

        # Try each endpoint with each model
        success = False

        for endpoint in endpoints:
            # Skip trying multiple models for Ollama endpoint
            if "11434" in endpoint:
                try:
                    print(f"🔄 Trying Ollama endpoint: {endpoint}")

                    # Use the json parameter instead of manually serializing
                    response = requests.post(
                        endpoint,
                        json=ollama_payload,
                        headers={"Content-Type": "application/json"},
                        timeout=60  # Increased timeout for larger code analysis
                    )

                    print(f"🔄 Response status code: {response.status_code}")

                    if response.status_code == 200:
                        print(f"✅ Successfully connected to Ollama at {endpoint}")
                        success = True
                        break
                    else:
                        print(f"⚠️ Received status code {response.status_code} from {endpoint}")
                        print(f"Response content: {response.text[:200]}...")
                except requests.exceptions.RequestException as e:
                    print(f"⚠️ Failed to connect to {endpoint}: {str(e)}")
                except Exception as e:
                    print(f"⚠️ Unexpected error with {endpoint}: {str(e)}")
            else:
                # Try each model with this endpoint
                for model in models:
                    try:
                        # Create a payload with the current model
                        current_payload = payload_template.copy()
                        current_payload["model"] = model

                        print(f"🔄 Trying endpoint: {endpoint} with model: {model}")

                        # Use the json parameter instead of manually serializing
                        response = requests.post(
                            endpoint,
                            json=current_payload,
                            headers={"Content-Type": "application/json"},
                            timeout=60  # Increased timeout for larger code analysis
                        )

                        print(f"🔄 Response status code: {response.status_code}")

                        if response.status_code == 200:
                            print(f"✅ Successfully connected to LLM at {endpoint} with model {model}")
                            success = True
                            break
                        else:
                            print(f"⚠️ Received status code {response.status_code} from {endpoint} with model {model}")
                            # Only print response content for non-404 errors to reduce noise
                            if response.status_code != 404:
                                print(f"Response content: {response.text[:200]}...")
                    except requests.exceptions.RequestException as e:
                        print(f"⚠️ Failed to connect to {endpoint} with model {model}: {str(e)}")
                    except Exception as e:
                        print(f"⚠️ Unexpected error with {endpoint} and model {model}: {str(e)}")

                # If we found a working model, break out of the endpoint loop
                if success:
                    break

        if response and response.status_code == 200:
            try:
                result = response.json()
                print(f"✅ Received JSON response: {str(result)[:200]}...")

                # Handle different API response formats
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"]
                elif "message" in result:  # Ollama format
                    return result["message"]["content"]
                else:
                    print(f"⚠️ Unexpected response format: {str(result)[:200]}...")
            except json.JSONDecodeError as e:
                print(f"⚠️ Failed to decode JSON response: {str(e)}")
                print(f"Response content: {response.text[:200]}...")

        return "Error: Unable to get a response from any LLM service. Please ensure LM Studio or another compatible LLM service is running."

    except Exception as e:
        return f"Error communicating with LM Studio: {str(e)}"

def perform_detailed_analysis(lines):
    """
    Perform a more detailed manual analysis of the code.

    Args:
        lines: List of code lines to analyze

    Returns:
        dict: Detailed analysis results
    """
    result = {
        "imports": [],
        "classes": [],
        "functions": [],
        "variables": [],
        "quality_issues": []
    }

    # Track indentation levels for quality analysis
    indentation_levels = set()
    long_lines = 0

    # Regular expressions for pattern matching
    import_pattern = r'^(?:from\s+(\S+)\s+import\s+(.+)|import\s+(.+))'
    class_pattern = r'^\s*class\s+([A-Za-z0-9_]+)'
    function_pattern = r'^\s*def\s+([A-Za-z0-9_]+)'
    variable_pattern = r'^\s*([A-Za-z0-9_]+)\s*='

    # Track try-except blocks
    try_blocks = 0
    except_blocks = 0

    # Analyze each line
    for line in lines:
        # Skip empty lines
        if not line.strip():
            continue

        # Check indentation
        if line.strip() and line.startswith(' '):
            indent_size = len(line) - len(line.lstrip(' '))
            indentation_levels.add(indent_size)

        # Check line length
        if len(line) > 100:
            long_lines += 1

        # Check for imports
        import_match = re.search(import_pattern, line)
        if import_match:
            if import_match.group(1) and import_match.group(2):
                # from X import Y
                module = import_match.group(1)
                imports = import_match.group(2).split(',')
                for imp in imports:
                    imp = imp.strip()
                    if imp:
                        result["imports"].append(f"{module}.{imp}")
            elif import_match.group(3):
                # import X
                imports = import_match.group(3).split(',')
                for imp in imports:
                    imp = imp.strip()
                    if imp:
                        result["imports"].append(imp)

        # Check for classes
        class_match = re.search(class_pattern, line)
        if class_match:
            result["classes"].append(class_match.group(1))

        # Check for functions
        function_match = re.search(function_pattern, line)
        if function_match:
            result["functions"].append(function_match.group(1))

        # Check for variables
        variable_match = re.search(variable_pattern, line)
        if variable_match and not line.strip().startswith('#'):
            var_name = variable_match.group(1)
            if var_name not in ['self', 'cls'] and not var_name.startswith('_'):
                result["variables"].append(var_name)

        # Track try-except blocks
        if re.search(r'^\s*try\s*:', line):
            try_blocks += 1
        if re.search(r'^\s*except\s+', line):
            except_blocks += 1

    # Add quality issues
    if long_lines > 0:
        result["quality_issues"].append(f"Readability: Found {long_lines} lines longer than 100 characters")

    if len(indentation_levels) > 1:
        result["quality_issues"].append(f"Readability: Inconsistent indentation: found {indentation_levels} different indent sizes")

    if try_blocks != except_blocks:
        result["quality_issues"].append(f"Error handling: Mismatched try-except blocks")

    # Remove duplicates
    result["imports"] = list(set(result["imports"]))
    result["classes"] = list(set(result["classes"]))
    result["functions"] = list(set(result["functions"]))
    result["variables"] = list(set(result["variables"]))

    return result

def main():
    """Run the self-code analysis with command line arguments."""
    # Default to 100 lines
    num_lines = 100
    file_path = None

    # Parse command line arguments
    if len(sys.argv) > 1:
        try:
            num_lines = int(sys.argv[1])
        except ValueError:
            print(f"Invalid number of lines: {sys.argv[1]}. Using default: 100")

    if len(sys.argv) > 2:
        file_path = sys.argv[2]

    # Run the analysis
    result = analyze_own_code(num_lines, file_path)
    print(result)

if __name__ == "__main__":
    main()
