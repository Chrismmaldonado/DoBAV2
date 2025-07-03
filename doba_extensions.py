"""
DoBA Extensions Module
Provides additional capabilities for DoBA:
1. Web search functionality
2. Root access to the system
3. OCR (Optical Character Recognition) for screen reading
4. Code analysis for reading, analyzing, and critiquing code

This module is designed to work with AMD GPUs using ROCm.
"""

import os
import sys
import subprocess
import requests
import json
import time
import random
import uuid
from datetime import datetime
import threading
import re
import base64
from typing import List, Dict, Any, Optional, Union, Tuple

# For OCR functionality
try:
    import pytesseract
    import PIL.Image
    import PIL.ImageGrab
    import numpy as np
    OCR_AVAILABLE = True
    print("✅ OCR support available")
except ImportError:
    OCR_AVAILABLE = False
    print("⚠️ OCR support not available - install pytesseract and Pillow")

# For keyboard and mouse control
try:
    import pyautogui
    CONTROL_AVAILABLE = True
    print("✅ Keyboard and mouse control available")
except ImportError:
    CONTROL_AVAILABLE = False
    print("⚠️ Keyboard and mouse control not available - install pyautogui")

# For web search functionality
try:
    import requests
    from bs4 import BeautifulSoup
    WEB_SEARCH_AVAILABLE = True
    print("✅ Web search support available (Startpage)")
except ImportError:
    WEB_SEARCH_AVAILABLE = False
    print("⚠️ Web search support not available - install requests and beautifulsoup4")

# For browser automation
try:
    import selenium
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, NoSuchElementException
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from webdriver_manager.chrome import ChromeDriverManager
    BROWSER_AUTOMATION_AVAILABLE = True
    print("✅ Browser automation support available")
except ImportError:
    BROWSER_AUTOMATION_AVAILABLE = False
    print("⚠️ Browser automation not available - install selenium and webdriver_manager")

# For multi-monitor support
try:
    import screeninfo
    MULTI_MONITOR_AVAILABLE = True
    print("✅ Multi-monitor support available")
except ImportError:
    MULTI_MONITOR_AVAILABLE = False
    print("⚠️ Multi-monitor support not available - install screeninfo")

# For ROCm support
try:
    import torch
    if torch.cuda.is_available():
        print(f"✅ GPU acceleration available: {torch.cuda.get_device_name(0)}")
        ROCm_AVAILABLE = True
    else:
        print("⚠️ GPU acceleration not available")
        ROCm_AVAILABLE = False
except ImportError:
    ROCm_AVAILABLE = False
    print("⚠️ PyTorch not available - install PyTorch with ROCm support")


class WebSearch:
    """Provides web search capabilities using browser or Startpage"""

    def __init__(self):
        self.use_browser = BROWSER_AUTOMATION_AVAILABLE
        self.browser = None

    def search(self, query: str, max_results: int = 5, use_browser: bool = True, priority: int = 5) -> List[Dict[str, str]]:
        """
        Search the web for the given query

        Args:
            query: The search query
            max_results: Maximum number of results to return
            use_browser: Whether to use browser-based search (if available)
            priority: Priority of the search request (1-10, higher is more important)

        Returns:
            List of search results with title, body/snippet, and href/url
        """
        # Validate priority
        priority = max(1, min(10, priority))

        # Use browser-based search if available and requested
        if self.use_browser and use_browser:
            # Lazy initialization of browser automation
            if self.browser is None:
                from doba_extensions import BrowserAutomation
                self.browser = BrowserAutomation()

            # Perform browser-based search with priority
            results = self.browser.search(query, max_results, priority)

            # If browser search failed, fall back to Startpage
            if results and "error" in results[0]:
                print(f"⚠️ Browser search failed: {results[0]['error']}. Falling back to Startpage.")
                return self._search_with_startpage(query, max_results)

            return results
        else:
            # Try Startpage search first
            results = self._search_with_startpage(query, max_results)

            # Check if we got a rate limit error
            if results and "error" in results[0] and "rate limit" in results[0]["error"].lower():
                # Try fallback search methods
                return self._fallback_search(query, max_results)

            return results

    def _fallback_search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """
        Fallback search method when Startpage is rate limited

        Args:
            query: The search query
            max_results: Maximum number of results to return

        Returns:
            List of search results with title, body, and href
        """
        print(f"🔍 Using fallback search method for query: {query}")

        # Try to use a simple web scraping approach as fallback
        try:
            # Create a simple user agent rotation
            user_agents = [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"
            ]

            # Use a random user agent
            headers = {"User-Agent": random.choice(user_agents)}

            # Try to get results from a search engine that might be more tolerant
            # Format query for URL
            encoded_query = query.replace(' ', '+')

            # Try to get results from Bing
            response = requests.get(f"https://www.bing.com/search?q={encoded_query}", headers=headers, timeout=10)

            if response.status_code == 200:
                # Extract basic information from the response
                # This is a very simple extraction and might break if the page structure changes
                results = []

                # Extract titles and links using regex
                title_pattern = r'<h2><a href="([^"]+)"[^>]*>(.*?)</a></h2>'
                matches = re.findall(title_pattern, response.text)

                for i, (href, title) in enumerate(matches):
                    if i >= max_results:
                        break

                    # Clean up the title (remove HTML tags)
                    clean_title = re.sub(r'<[^>]+>', '', title)

                    # Extract a snippet of text around the match
                    snippet_pattern = rf'<p>(.*?{re.escape(clean_title[:20])}.*?)</p>'
                    snippet_match = re.search(snippet_pattern, response.text, re.DOTALL)
                    body = snippet_match.group(1) if snippet_match else "No description available"

                    # Clean up the body (remove HTML tags)
                    clean_body = re.sub(r'<[^>]+>', '', body)

                    results.append({
                        "title": clean_title,
                        "body": clean_body[:200] + "..." if len(clean_body) > 200 else clean_body,
                        "href": href
                    })

                if results:
                    print(f"✅ Fallback search successful: Found {len(results)} results")
                    return results

            # If we couldn't get results from Bing, return a helpful message
            return [{"error": "Search temporarily unavailable. Startpage is rate limited and fallback search failed. Please try again later."}]

        except Exception as e:
            print(f"❌ Fallback search error: {e}")
            return [{"error": f"All search methods failed. Please try again later. Error: {str(e)}"}]

    def _search_with_startpage(self, query: str, max_results: int = 5, max_retries: int = 3) -> List[Dict[str, str]]:
        """
        Search the web using Startpage with improved reliability and error handling

        Args:
            query: The search query
            max_results: Maximum number of results to return
            max_retries: Maximum number of retry attempts for transient failures

        Returns:
            List of search results with title, body, and href
        """
        if not WEB_SEARCH_AVAILABLE:
            return [{"error": "Web search not available. Install requests and beautifulsoup4 packages."}]

        # Simple in-memory cache to avoid repetitive identical searches
        if not hasattr(self, '_search_cache'):
            self._search_cache = {}

        # Rate limiting tracker to implement circuit breaker pattern
        if not hasattr(self, '_rate_limit_tracker'):
            self._rate_limit_tracker = {
                'last_rate_limit': 0,  # Timestamp of last rate limit
                'consecutive_failures': 0,  # Count of consecutive rate limit failures
                'cooldown_until': 0  # Timestamp until which we should avoid searches
            }

        # Check if we're in a cooldown period due to rate limiting
        current_time = time.time()
        if current_time < self._rate_limit_tracker['cooldown_until']:
            cooldown_remaining = int(self._rate_limit_tracker['cooldown_until'] - current_time)
            print(f"⚠️ Startpage search in cooldown period for {cooldown_remaining} more seconds due to rate limiting")
            return [{"error": f"Search temporarily unavailable due to rate limiting. Try again in {cooldown_remaining} seconds."}]

        # Check cache first (with a TTL of 5 minutes)
        cache_key = f"{query}_{max_results}"
        if cache_key in self._search_cache:
            cache_time, cache_results = self._search_cache[cache_key]
            # Cache valid for 5 minutes
            if time.time() - cache_time < 300:
                print(f"🔍 Using cached results for query: {query}")
                return cache_results

        # Track retry attempts
        retry_count = 0
        last_error = None

        # Different language settings to try if one fails
        languages_to_try = ["en", "all", "english", "auto"]

        # Randomize the order of languages to distribute load
        random.shuffle(languages_to_try)

        # User agent rotation to avoid detection
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:90.0) Gecko/20100101 Firefox/90.0"
        ]

        while retry_count < max_retries:
            try:
                # Use a different language for each retry to avoid rate limiting
                language = languages_to_try[min(retry_count, len(languages_to_try) - 1)]

                # Use a random user agent
                user_agent = random.choice(user_agents)

                # Add a delay between retries with exponential backoff
                if retry_count > 0:
                    # Exponential backoff with jitter: base_delay * (2^retry) + random_jitter
                    base_delay = 2.0
                    max_delay = 30.0  # Cap at 30 seconds
                    exponential_factor = min(2 ** retry_count, 8)  # Cap the exponential growth
                    jitter = random.uniform(0, 1)  # Add randomness to prevent synchronized retries
                    delay = min(base_delay * exponential_factor + jitter, max_delay)

                    print(f"🔍 Retrying Startpage search (attempt {retry_count+1}/{max_retries}) with language {language} after {delay:.1f}s delay...")
                    time.sleep(delay)

                # Format the query for URL
                encoded_query = query.replace(' ', '+')

                # Set up headers with user agent
                headers = {
                    "User-Agent": user_agent,
                    "Accept-Language": "en-US,en;q=0.9",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                    "Referer": "https://www.startpage.com/",
                    "DNT": "1"
                }

                # Make the request to Startpage
                url = f"https://www.startpage.com/sp/search?q={encoded_query}&language={language}"
                response = requests.get(url, headers=headers, timeout=15)

                # Check if we got a successful response
                if response.status_code != 200:
                    raise Exception(f"HTTP error: {response.status_code}")

                # Parse the HTML with BeautifulSoup
                soup = BeautifulSoup(response.text, 'html.parser')

                # Extract search results
                results = []

                # Find all search result containers
                # Note: This selector might need adjustment based on Startpage's actual HTML structure
                result_containers = soup.select('.search-result')

                # If we can't find results with the primary selector, try alternative selectors
                if not result_containers:
                    result_containers = soup.select('.w-gl__result')  # Alternative selector

                if not result_containers:
                    result_containers = soup.select('article')  # Another alternative

                if not result_containers:
                    result_containers = soup.select('.result')  # Common class for search results

                if not result_containers:
                    result_containers = soup.select('.web-result')  # Another common class

                if not result_containers:
                    # Try to find any div that might contain search results
                    result_containers = soup.select('div.card')  # Startpage sometimes uses card layout

                # Debug: If still no results, try to understand the page structure
                if not result_containers:
                    print("⚠️ No search results found with standard selectors. Analyzing page structure...")

                    # Look for any elements that might contain search results
                    potential_containers = soup.select('div[class*="result"], div[class*="search"], div[class*="listing"]')
                    if potential_containers:
                        print(f"🔍 Found {len(potential_containers)} potential result containers with generic selectors")
                        result_containers = potential_containers
                    else:
                        # Last resort: try to identify the structure of the page
                        print("🔍 Analyzing HTML structure to identify search results...")

                        # Check if we're getting a captcha or block page
                        if soup.select_one('form[action*="captcha"]') or soup.select_one('div[class*="captcha"]'):
                            print("⚠️ Detected CAPTCHA challenge on the page")
                            # Save the HTML for debugging (in a safe location)
                            try:
                                debug_dir = os.path.join(os.path.expanduser("~"), ".cache", "doba_debug")
                                os.makedirs(debug_dir, exist_ok=True)
                                debug_file = os.path.join(debug_dir, f"startpage_captcha_{int(time.time())}.html")
                                with open(debug_file, 'w', encoding='utf-8') as f:
                                    f.write(response.text[:10000])  # Save first 10KB to avoid huge files
                                print(f"📝 Saved first 10KB of HTML to {debug_file} for debugging")
                            except Exception as e:
                                print(f"⚠️ Failed to save debug HTML: {e}")

                        # Try to find any links on the page as a last resort
                        all_links = soup.select('a[href]')
                        if all_links:
                            print(f"🔍 Found {len(all_links)} links on the page, attempting to extract search results")

                            # Filter links that might be search results (exclude navigation, etc.)
                            potential_result_links = [
                                link for link in all_links 
                                if not any(nav_term in link.get('href', '').lower() 
                                          for nav_term in ['javascript:', 'startpage.com', 'about', 'settings', 'login'])
                            ]

                            if potential_result_links:
                                print(f"🔍 Identified {len(potential_result_links)} potential search result links")
                                # Create simple container objects from these links
                                from types import SimpleNamespace
                                result_containers = []

                                for link in potential_result_links[:max_results]:
                                    # Create a simple container with the link
                                    container = SimpleNamespace()
                                    container.link = link
                                    container.select_one = lambda selector: link if selector == 'a[href]' else None
                                    container.get_text = lambda: link.get_text()
                                    container.find_parent = lambda tag: None
                                    result_containers.append(container)

                # Process each result
                for i, container in enumerate(result_containers):
                    if i >= max_results:
                        break

                    try:
                        # Handle SimpleNamespace objects from last-resort fallback
                        if hasattr(container, 'link') and not callable(getattr(container, 'select_one', None)):
                            # This is a SimpleNamespace object from our fallback
                            link = container.link
                            title = link.get_text().strip() or "No title"
                            href = link.get('href', "")
                            body = "No description available (extracted from link)"
                        else:
                            # Normal processing for BeautifulSoup elements
                            # Extract title - try multiple selectors
                            title_elem = (
                                container.select_one('h3') or 
                                container.select_one('.w-gl__result-title') or
                                container.select_one('h2') or
                                container.select_one('[class*="title"]') or
                                container.select_one('a[href]')  # Last resort: use link text as title
                            )
                            title = title_elem.get_text().strip() if title_elem else "No title"

                            # Extract URL - try multiple approaches
                            url_elem = None
                            # First try to find a direct link
                            url_elem = container.select_one('a[href]')

                            # If no direct link, try to find a link in the title element
                            if not url_elem and title_elem:
                                if hasattr(title_elem, 'name') and title_elem.name == 'a' and title_elem.has_attr('href'):
                                    url_elem = title_elem
                                elif hasattr(title_elem, 'find_parent'):
                                    url_elem = title_elem.find_parent('a')

                            # If still no URL, look for any element with a data-url attribute
                            if not url_elem:
                                data_url_elem = container.select_one('[data-url]')
                                if data_url_elem:
                                    href = data_url_elem.get('data-url', "")
                                else:
                                    href = ""
                            else:
                                href = url_elem.get('href', "")

                            # Extract description/snippet - try multiple selectors
                            desc_elem = (
                                container.select_one('p') or 
                                container.select_one('.w-gl__description') or
                                container.select_one('[class*="desc"]') or
                                container.select_one('[class*="snippet"]') or
                                container.select_one('[class*="summary"]')
                            )

                            # If no specific description element found, try to get any text not in the title
                            if not desc_elem:
                                # Get all text from the container
                                all_text = container.get_text().strip()
                                # Remove the title text
                                if title in all_text:
                                    body_text = all_text.replace(title, '', 1).strip()
                                    body = body_text if body_text else "No description"
                                else:
                                    body = "No description"
                            else:
                                body = desc_elem.get_text().strip()

                        # If the URL is relative, make it absolute
                        if href and href.startswith('/'):
                            href = f"https://www.startpage.com{href}"

                        # Handle special case where URL might be encoded or in a different format
                        if href and "url=" in href:
                            try:
                                # Extract the actual URL from a redirect URL
                                import urllib.parse
                                parsed_url = urllib.parse.urlparse(href)
                                query_params = urllib.parse.parse_qs(parsed_url.query)
                                if 'url' in query_params and query_params['url']:
                                    href = query_params['url'][0]
                            except Exception as e:
                                print(f"⚠️ Error extracting URL from redirect: {e}")

                    except Exception as extraction_error:
                        # Handle any errors during extraction
                        print(f"⚠️ Error extracting data from search result: {extraction_error}")
                        title = "Error extracting result"
                        body = f"An error occurred while processing this search result: {str(extraction_error)}"
                        href = ""

                    # Add to results
                    results.append({
                        "title": title,
                        "body": body,
                        "href": href
                    })

                # If we got results, cache them and return
                if results:
                    # Reset rate limit tracker on success
                    self._rate_limit_tracker['consecutive_failures'] = 0

                    # Store in cache
                    self._search_cache[cache_key] = (time.time(), results)

                    # Limit cache size to prevent memory issues
                    if len(self._search_cache) > 100:
                        # Remove oldest entries
                        oldest_keys = sorted(self._search_cache.keys(), 
                                            key=lambda k: self._search_cache[k][0])[:50]
                        for key in oldest_keys:
                            del self._search_cache[key]

                    return results
                else:
                    # Empty results, try again with a different language
                    last_error = "No results found"
                    retry_count += 1

            except Exception as e:
                last_error = str(e)
                retry_count += 1
                print(f"⚠️ Startpage search error (attempt {retry_count}/{max_retries}): {last_error}")

                # Check for rate limiting errors
                is_rate_limit = any(term in str(last_error).lower() for term in 
                                   ["429", "too many requests", "rate limit", "ratelimit", "blocked", "captcha"])

                if is_rate_limit:
                    # Update rate limit tracker
                    self._rate_limit_tracker['last_rate_limit'] = current_time
                    self._rate_limit_tracker['consecutive_failures'] += 1

                    # Implement exponential backoff for rate limiting
                    backoff_time = min(60 * (2 ** (self._rate_limit_tracker['consecutive_failures'] - 1)), 3600)

                    # If we've hit rate limits multiple times, implement a longer cooldown
                    if self._rate_limit_tracker['consecutive_failures'] >= 3:
                        cooldown_period = min(1800 * (self._rate_limit_tracker['consecutive_failures'] - 2), 7200)  # Up to 2 hours
                        self._rate_limit_tracker['cooldown_until'] = current_time + cooldown_period
                        print(f"⚠️ Startpage search rate limited too many times. Cooling down for {cooldown_period//60} minutes.")

                    print(f"⚠️ Rate limit detected. Waiting {backoff_time} seconds before retry.")
                    time.sleep(backoff_time)

        # If we get here, all retries failed
        error_result = [{"error": f"Search failed after {max_retries} attempts: {last_error}"}]

        # Cache the error too to prevent repeated failures (but with a shorter TTL)
        self._search_cache[cache_key] = (time.time(), error_result)

        # Try to provide a helpful message if we're rate limited
        if self._rate_limit_tracker['consecutive_failures'] > 0:
            error_result = [{"error": "Search temporarily unavailable due to rate limiting. Please try again later or use a different search method."}]

        return error_result

    def search_news(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """
        Search for news articles related to the query

        Args:
            query: The search query
            max_results: Maximum number of results to return

        Returns:
            List of news results with title, body, and url
        """
        if not WEB_SEARCH_AVAILABLE:
            return [{"error": "Web search not available. Install duckduckgo_search package."}]

        try:
            results = []
            for r in self.search_engine.news(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "body": r.get("body", ""),
                    "url": r.get("url", ""),
                    "date": r.get("date", "")
                })
            return results
        except Exception as e:
            return [{"error": f"News search failed: {str(e)}"}]

    def search_images(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """
        Search for images related to the query

        Args:
            query: The search query
            max_results: Maximum number of results to return

        Returns:
            List of image results with title and image_url
        """
        if not WEB_SEARCH_AVAILABLE:
            return [{"error": "Web search not available. Install duckduckgo_search package."}]

        try:
            results = []
            for r in self.search_engine.images(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "image_url": r.get("image", ""),
                    "source": r.get("source", "")
                })
            return results
        except Exception as e:
            return [{"error": f"Image search failed: {str(e)}"}]

    def format_results_as_text(self, results: List[Dict[str, str]]) -> str:
        """
        Format search results as a readable text

        Args:
            results: List of search results

        Returns:
            Formatted text of search results
        """
        if not results:
            return "No results found."

        if "error" in results[0]:
            return results[0]["error"]

        formatted_text = "Search Results:\n\n"

        for i, result in enumerate(results, 1):
            formatted_text += f"{i}. {result.get('title', 'No title')}\n"
            formatted_text += f"   {result.get('body', result.get('snippet', 'No description'))}\n"
            formatted_text += f"   URL: {result.get('href', result.get('url', 'No URL'))}\n\n"

        return formatted_text


class SystemAccess:
    """Provides secure access to system commands with proper permissions"""

    def __init__(self):
        self.sudo_available = self._check_sudo_available()

    def _check_sudo_available(self) -> bool:
        """Check if sudo is available on the system"""
        try:
            result = subprocess.run(
                ["which", "sudo"], 
                capture_output=True, 
                text=True, 
                timeout=2
            )
            return result.returncode == 0
        except Exception:
            return False

    def execute_command(self, command: str, use_root: bool = False, timeout: int = 30) -> Dict[str, Any]:
        """
        Execute a system command with optional root privileges

        Args:
            command: The command to execute
            use_root: Whether to use root privileges (sudo)
            timeout: Command timeout in seconds

        Returns:
            Dictionary with stdout, stderr, and return code
        """
        # No security checks - allow all commands to be executed without restrictions

        try:
            # Prepare the command
            cmd_list = command.split()
            if use_root and self.sudo_available:
                cmd_list = ["sudo"] + cmd_list

            # Execute the command
            result = subprocess.run(
                cmd_list,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": f"Command timed out after {timeout} seconds",
                "returncode": -1,
                "error": "timeout"
            }
        except Exception as e:
            return {
                "stdout": "",
                "stderr": str(e),
                "returncode": -1,
                "error": str(e)
            }

    def _is_dangerous_command(self, command: str) -> bool:
        """
        Check if a command is potentially dangerous

        Args:
            command: The command to check

        Returns:
            Always returns False to allow all commands without restrictions
        """
        # No command is considered dangerous - allow all commands to be executed
        return False


class ComputerControl:
    """Provides keyboard and mouse control capabilities with multi-monitor support"""

    def __init__(self):
        self.available = CONTROL_AVAILABLE
        self.multi_monitor_available = MULTI_MONITOR_AVAILABLE
        self.multi_monitor = None

    def _init_multi_monitor(self):
        """Initialize multi-monitor support if available"""
        if self.multi_monitor_available and self.multi_monitor is None:
            from doba_extensions import MultiMonitorSupport
            self.multi_monitor = MultiMonitorSupport()

    def move_mouse(self, x: int, y: int, monitor_id: int = None, human_like: bool = True) -> bool:
        """
        Move the mouse to the specified coordinates with enhanced human-like movement patterns

        Args:
            x: X coordinate
            y: Y coordinate
            monitor_id: Optional monitor ID (1-based) for multi-monitor support
            human_like: Whether to use human-like movement patterns (non-linear path, variable speed)

        Returns:
            True if successful, False otherwise
        """
        if not self.available:
            print("⚠️ Computer control not available. Install pyautogui.")
            return False

        try:
            # Handle multi-monitor coordinates if monitor_id is specified
            target_x, target_y = x, y
            if monitor_id is not None:
                self._init_multi_monitor()
                if self.multi_monitor and self.multi_monitor.available:
                    # Convert monitor-relative coordinates to global coordinates
                    target_x, target_y = self.multi_monitor.convert_to_global_coordinates(x, y, monitor_id)
                    print(f"🖱️ Moving mouse to monitor {monitor_id} coordinates ({x}, {y}) -> global ({target_x}, {target_y})")
                else:
                    print("⚠️ Multi-monitor support not available. Using coordinates as global.")
            else:
                print(f"🖱️ Moving mouse to global coordinates ({target_x}, {target_y})")

            if not human_like:
                # Use standard pyautogui movement
                pyautogui.moveTo(target_x, target_y)
                return True

            # Enhanced human-like mouse movement with non-linear path, variable speed,
            # occasional pauses, and overshooting with correction
            import random
            import math

            # Get current mouse position
            current_x, current_y = pyautogui.position()

            # Calculate distance to target
            distance = math.sqrt((target_x - current_x) ** 2 + (target_y - current_y) ** 2)

            # If distance is very small, just move directly
            if distance < 10:
                pyautogui.moveTo(target_x, target_y)
                return True

            # Determine if we should overshoot and correct (more likely for longer distances)
            should_overshoot = random.random() < min(0.3, distance / 1000)

            # Calculate overshoot amount if needed (5-15% past the target)
            overshoot_factor = 0
            if should_overshoot:
                overshoot_factor = random.uniform(0.05, 0.15)
                # Calculate overshoot target
                overshoot_x = target_x + (target_x - current_x) * overshoot_factor
                overshoot_y = target_y + (target_y - current_y) * overshoot_factor

            # Calculate number of steps based on distance with more variability
            # More steps for longer distances, but with diminishing returns and some randomness
            base_steps = min(60, max(10, int(distance / 15)))
            steps = int(base_steps * random.uniform(0.8, 1.2))

            # Generate a more natural path with multiple control points for longer distances
            if distance > 200:
                # Use a cubic Bezier curve for longer distances (more control points)
                # This creates a more complex and natural path

                # Generate control points with more variability
                # First control point - closer to start
                control1_x = current_x + (target_x - current_x) * random.uniform(0.2, 0.4)
                control1_y = current_y + (target_y - current_y) * random.uniform(0.2, 0.4)

                # Add some perpendicular offset for more natural curve
                perpendicular_x = -(target_y - current_y) * random.uniform(0.05, 0.15)
                perpendicular_y = (target_x - current_x) * random.uniform(0.05, 0.15)

                control1_x += perpendicular_x
                control1_y += perpendicular_y

                # Second control point - closer to end
                control2_x = current_x + (target_x - current_x) * random.uniform(0.6, 0.8)
                control2_y = current_y + (target_y - current_y) * random.uniform(0.6, 0.8)

                # Add opposite perpendicular offset for second control point
                control2_x -= perpendicular_x
                control2_y -= perpendicular_y

                # Move along the cubic Bezier path with variable speed
                for i in range(steps + 1):
                    # Parameter t goes from 0 to 1
                    t = i / steps

                    # Occasionally add a small pause to simulate human thinking or distraction
                    if random.random() < 0.02:  # 2% chance
                        time.sleep(random.uniform(0.1, 0.3))

                    # Cubic Bezier curve formula
                    # B(t) = (1-t)³P₀ + 3(1-t)²tP₁ + 3(1-t)t²P₂ + t³P₃
                    t_inv = 1 - t

                    # Calculate position along the curve
                    pos_x = (t_inv**3 * current_x + 
                             3 * t_inv**2 * t * control1_x + 
                             3 * t_inv * t**2 * control2_x + 
                             t**3 * (overshoot_x if should_overshoot else target_x))

                    pos_y = (t_inv**3 * current_y + 
                             3 * t_inv**2 * t * control1_y + 
                             3 * t_inv * t**2 * control2_y + 
                             t**3 * (overshoot_y if should_overshoot else target_y))

                    # Add variable jitter based on speed (more jitter when moving faster)
                    # This simulates less control during faster movements
                    speed_factor = 3 * t * (1 - t)  # Peaks at t=0.5
                    jitter_amount = 1 + speed_factor * 2
                    jitter_x = random.uniform(-jitter_amount, jitter_amount) if t > 0 and t < 1 else 0
                    jitter_y = random.uniform(-jitter_amount, jitter_amount) if t > 0 and t < 1 else 0

                    # Move to the position
                    pyautogui.moveTo(pos_x + jitter_x, pos_y + jitter_y)

                    # Enhanced variable speed: slower at start and end, faster in the middle
                    # Use a more natural acceleration/deceleration curve
                    if t < 0.2:
                        # Initial acceleration phase - gradually speed up
                        delay = 0.01 * (1 - t/0.2 * 0.7)
                    elif t > 0.8:
                        # Final deceleration phase - gradually slow down
                        delay = 0.01 * (0.3 + (t-0.8)/0.2 * 0.7)
                    else:
                        # Middle phase - maintain higher speed with some variability
                        delay = 0.01 * (0.3 + random.uniform(-0.1, 0.1))

                    time.sleep(delay)
            else:
                # Use quadratic Bezier for shorter distances
                # Generate control points for the curve with more variability
                control_x = (current_x + target_x) / 2
                control_y = (current_y + target_y) / 2

                # Add perpendicular offset for more natural curve
                perpendicular_x = -(target_y - current_y) * random.uniform(0.1, 0.3)
                perpendicular_y = (target_x - current_x) * random.uniform(0.1, 0.3)

                control_x += perpendicular_x
                control_y += perpendicular_y

                # Move along the path with variable speed
                for i in range(steps + 1):
                    # Parameter t goes from 0 to 1
                    t = i / steps

                    # Occasionally add a small pause to simulate human thinking
                    if random.random() < 0.02:  # 2% chance
                        time.sleep(random.uniform(0.1, 0.3))

                    # Quadratic Bezier curve formula
                    # B(t) = (1-t)²P₀ + 2(1-t)tP₁ + t²P₂
                    t_inv = 1 - t

                    # Calculate position along the curve
                    pos_x = (t_inv**2 * current_x + 
                             2 * t_inv * t * control_x + 
                             t**2 * (overshoot_x if should_overshoot else target_x))

                    pos_y = (t_inv**2 * current_y + 
                             2 * t_inv * t * control_y + 
                             t**2 * (overshoot_y if should_overshoot else target_y))

                    # Add variable jitter based on speed
                    speed_factor = 4 * t * (1 - t)  # Peaks at t=0.5
                    jitter_amount = 1 + speed_factor
                    jitter_x = random.uniform(-jitter_amount, jitter_amount) if t > 0 and t < 1 else 0
                    jitter_y = random.uniform(-jitter_amount, jitter_amount) if t > 0 and t < 1 else 0

                    # Move to the position
                    pyautogui.moveTo(pos_x + jitter_x, pos_y + jitter_y)

                    # Enhanced variable speed
                    if t < 0.2:
                        # Initial acceleration phase
                        delay = 0.01 * (1 - t/0.2 * 0.7)
                    elif t > 0.8:
                        # Final deceleration phase
                        delay = 0.01 * (0.3 + (t-0.8)/0.2 * 0.7)
                    else:
                        # Middle phase
                        delay = 0.01 * (0.3 + random.uniform(-0.1, 0.1))

                    time.sleep(delay)

            # If we overshot, now move back to the actual target with a correction movement
            if should_overshoot:
                # Pause briefly before correction (as if noticing the overshoot)
                time.sleep(random.uniform(0.1, 0.3))

                # Get current position after overshoot
                current_x, current_y = pyautogui.position()

                # Calculate a shorter, more direct path for the correction
                correction_steps = max(5, int(steps / 3))

                # Move directly to target with slight curve
                for i in range(correction_steps + 1):
                    t = i / correction_steps

                    # Simple curve for correction
                    t_inv = 1 - t
                    pos_x = t_inv * current_x + t * target_x
                    pos_y = t_inv * current_y + t * target_y

                    # Smaller jitter for correction movement (more precise)
                    jitter_x = random.uniform(-0.5, 0.5) if t > 0 and t < 1 else 0
                    jitter_y = random.uniform(-0.5, 0.5) if t > 0 and t < 1 else 0

                    pyautogui.moveTo(pos_x + jitter_x, pos_y + jitter_y)

                    # Faster, more consistent movement for correction
                    time.sleep(0.01)

            # Ensure we end exactly at the target
            pyautogui.moveTo(target_x, target_y)
            return True
        except Exception as e:
            print(f"⚠️ Mouse movement failed: {e}")
            return False

    def click(self, x: int = None, y: int = None, button: str = 'left', monitor_id: int = None) -> bool:
        """
        Click at the current or specified position with human-like behavior

        Args:
            x: Optional X coordinate
            y: Optional Y coordinate
            button: Mouse button to click ('left', 'right', 'middle')
            monitor_id: Optional monitor ID (1-based) for multi-monitor support

        Returns:
            True if successful, False otherwise
        """
        if not self.available:
            print("⚠️ Computer control not available. Install pyautogui.")
            return False

        try:
            import random
            import time

            # Determine target coordinates
            target_x, target_y = None, None

            if x is not None and y is not None:
                # Handle multi-monitor coordinates if monitor_id is specified
                if monitor_id is not None:
                    self._init_multi_monitor()
                    if self.multi_monitor and self.multi_monitor.available:
                        # Convert monitor-relative coordinates to global coordinates
                        target_x, target_y = self.multi_monitor.convert_to_global_coordinates(x, y, monitor_id)
                        print(f"🖱️ Clicking {button} at monitor {monitor_id} coordinates ({x}, {y}) -> global ({target_x}, {target_y})")
                    else:
                        print("⚠️ Multi-monitor support not available. Using coordinates as global.")
                        target_x, target_y = x, y
                else:
                    # Use coordinates as global if no monitor_id
                    print(f"🖱️ Clicking {button} at global coordinates ({x}, {y})")
                    target_x, target_y = x, y

                # Get current mouse position
                current_x, current_y = pyautogui.position()

                # If we're already very close to the target, add a tiny movement before clicking
                # This simulates the slight movement humans make even when trying to hold still
                if abs(current_x - target_x) < 5 and abs(current_y - target_y) < 5:
                    # Add a very small random movement
                    jitter_x = random.uniform(-2, 2)
                    jitter_y = random.uniform(-2, 2)
                    pyautogui.moveTo(current_x + jitter_x, current_y + jitter_y, duration=0.05)
                    time.sleep(random.uniform(0.05, 0.1))

                # If we're not already at the target position, move there with human-like movement
                if abs(current_x - target_x) > 3 or abs(current_y - target_y) > 3:
                    # Move to the position with human-like movement
                    self.move_mouse(target_x, target_y, human_like=True)

                # Add a small pause before clicking (simulating human verification)
                hover_time = random.uniform(0.1, 0.3)
                if button == 'right':  # Longer hover for right-click (context menu)
                    hover_time = random.uniform(0.2, 0.5)
                time.sleep(hover_time)

                # Determine if we should do a double-click (rare, only for left button)
                do_double_click = button == 'left' and random.random() < 0.02  # 2% chance

                if do_double_click:
                    # Perform a double-click with a human-like interval
                    pyautogui.click(target_x, target_y, clicks=2, interval=random.uniform(0.08, 0.15), button=button)
                    print(f"🖱️ Double-clicked {button} at coordinates ({target_x}, {target_y})")
                else:
                    # Occasionally add a small movement right before clicking (last-moment adjustment)
                    if random.random() < 0.1:  # 10% chance
                        micro_adjust_x = random.uniform(-2, 2)
                        micro_adjust_y = random.uniform(-2, 2)
                        pyautogui.moveTo(target_x + micro_adjust_x, target_y + micro_adjust_y, duration=0.05)
                        time.sleep(random.uniform(0.02, 0.08))
                        # Move back to target
                        pyautogui.moveTo(target_x, target_y, duration=0.05)
                        time.sleep(random.uniform(0.02, 0.05))

                    # Perform the click
                    pyautogui.click(button=button)
            else:
                # Clicking at current position
                print(f"🖱️ Clicking {button} at current position")

                # Add a small pause before clicking
                time.sleep(random.uniform(0.05, 0.2))

                # Determine if we should do a double-click (rare, only for left button)
                do_double_click = button == 'left' and random.random() < 0.02  # 2% chance

                if do_double_click:
                    # Perform a double-click with a human-like interval
                    pyautogui.click(clicks=2, interval=random.uniform(0.08, 0.15), button=button)
                    print(f"🖱️ Double-clicked {button} at current position")
                else:
                    # Perform the click
                    pyautogui.click(button=button)

            # Add a small pause after clicking (simulating human reaction time)
            time.sleep(random.uniform(0.05, 0.2))

            return True
        except Exception as e:
            print(f"⚠️ Mouse click failed: {e}")
            return False

    def type_text(self, text: str, interval: float = 0.0, human_like: bool = True) -> bool:
        """
        Type text at the current cursor position with enhanced human-like typing patterns

        Args:
            text: Text to type
            interval: Base time between keypresses in seconds
            human_like: Whether to use human-like typing patterns (randomized delays, occasional mistakes)

        Returns:
            True if successful, False otherwise
        """
        if not self.available:
            print("⚠️ Computer control not available. Install pyautogui.")
            return False

        try:
            if not human_like:
                # Use standard pyautogui typing
                pyautogui.write(text, interval=interval)
                return True

            # Enhanced human-like typing with variable speed, realistic pauses, and occasional mistakes
            import random
            import re

            # Define typing characteristics
            base_interval = max(0.05, interval)  # Ensure a minimum delay for human-like typing

            # Split text into words for more realistic word-based timing
            # This regex handles various punctuation and whitespace
            words = re.findall(r'\S+|\s+', text)

            # Define typing skill level (affects error rate and speed consistency)
            # Values between 0.5 (poor typist) and 0.95 (expert typist)
            typing_skill = random.uniform(0.75, 0.92)

            # Calculate error probability based on skill (lower skill = higher error rate)
            typo_probability = 0.04 * (1 - typing_skill)

            # Track if we're at the start of a sentence for capitalization errors
            sentence_start = True

            # Process each word
            for word in words:
                # If this is just whitespace, handle it specially
                if word.isspace():
                    # Type the whitespace
                    pyautogui.write(word)

                    # Add a pause after periods, commas, etc. (if previous word ended with punctuation)
                    if len(words) > 0 and words[-1][-1:] in ['.', '!', '?']:
                        # Longer pause after end of sentence
                        time.sleep(random.uniform(0.5, 1.0))
                        sentence_start = True
                    elif len(words) > 0 and words[-1][-1:] in [',', ';', ':']:
                        # Medium pause after clause
                        time.sleep(random.uniform(0.3, 0.7))
                    else:
                        # Short pause after word
                        time.sleep(random.uniform(0.1, 0.3))

                    continue

                # Check if this word is complex (longer or contains unusual characters)
                is_complex = len(word) > 7 or any(c not in 'abcdefghijklmnopqrstuvwxyz ' for c in word.lower())

                # Add a thinking pause before complex words
                if is_complex and random.random() < 0.4:  # 40% chance for complex words
                    time.sleep(random.uniform(0.3, 0.8))

                # Type each character in the word with variable timing
                i = 0
                while i < len(word):
                    char = word[i]

                    # Determine if this character should be typed with an error
                    make_error = random.random() < typo_probability

                    # Special case for capitalization errors at sentence start
                    if sentence_start and i == 0 and char.isalpha():
                        # Occasionally forget to capitalize first letter of sentence
                        if char.isupper() and random.random() < 0.05:  # 5% chance
                            char = char.lower()
                            make_error = False  # Don't make additional errors
                        # Occasionally capitalize when not needed
                        elif char.islower() and random.random() < 0.03:  # 3% chance
                            char = char.upper()
                            make_error = False  # Don't make additional errors

                        sentence_start = False

                    # Calculate typing speed based on character context
                    # Type slower for numbers, symbols, and capital letters
                    if char.isdigit() or char in '!@#$%^&*()_+-=[]{}|;:\'",.<>/?\\':
                        # Slower for special characters
                        current_interval = base_interval * random.uniform(1.1, 1.8)
                    elif char.isupper():
                        # Slower for uppercase (shift key)
                        current_interval = base_interval * random.uniform(1.0, 1.6)
                    else:
                        # Normal speed for lowercase with some variability based on skill
                        # Higher skill = more consistent speed
                        variability = (1 - typing_skill) * 0.8 + 0.2  # Range from 0.2 to 1.0
                        current_interval = base_interval * random.uniform(
                            max(0.6, 1.0 - variability),
                            min(1.6, 1.0 + variability)
                        )

                    # Occasionally add a longer pause (as if thinking or distracted)
                    if random.random() < 0.01:  # 1% chance
                        time.sleep(random.uniform(0.7, 1.5))

                    # Make a typo if determined earlier
                    if make_error:
                        error_type = random.random()

                        if error_type < 0.6:  # 60% of errors: adjacent key
                            # Type a wrong character (usually adjacent to the intended one)
                            adjacent_chars = self._get_adjacent_keys(char)
                            if adjacent_chars:
                                wrong_char = random.choice(adjacent_chars)
                                pyautogui.write(wrong_char)
                                time.sleep(random.uniform(0.1, 0.4))  # Pause before noticing error

                                # Sometimes make multiple backspaces (as if deleting more text)
                                if random.random() < 0.2:  # 20% chance of multiple backspaces
                                    for _ in range(random.randint(2, 3)):
                                        pyautogui.press('backspace')
                                        time.sleep(random.uniform(0.05, 0.15))
                                else:
                                    # Single backspace
                                    pyautogui.press('backspace')

                                time.sleep(random.uniform(0.1, 0.3))  # Pause after correction

                        elif error_type < 0.8:  # 20% of errors: double character
                            # Type the character twice
                            pyautogui.write(char + char)
                            time.sleep(random.uniform(0.1, 0.3))  # Pause before noticing error
                            pyautogui.press('backspace')
                            time.sleep(random.uniform(0.05, 0.2))  # Pause after correction

                        else:  # 20% of errors: skip character then add it
                            # Skip this character and type the next one
                            if i + 1 < len(word):
                                pyautogui.write(word[i+1])
                                time.sleep(random.uniform(0.1, 0.3))  # Pause before noticing error
                                pyautogui.press('backspace')
                                time.sleep(random.uniform(0.1, 0.3))  # Pause after correction
                                # Now type the correct sequence
                                pyautogui.write(char)
                                i += 1  # Skip the next character since we already typed it
                            else:
                                # If we're at the last character, just type it correctly
                                pyautogui.write(char)
                    else:
                        # Type the correct character
                        pyautogui.write(char)

                    # Wait before typing the next character
                    time.sleep(current_interval)
                    i += 1

                # Add a slight pause between words (already handled for whitespace)
                if not word.isspace() and i < len(text) - 1:
                    time.sleep(random.uniform(0.05, 0.15))

            return True
        except Exception as e:
            print(f"⚠️ Typing failed: {e}")
            return False

    def _get_adjacent_keys(self, char: str) -> list:
        """
        Get adjacent keys on a QWERTY keyboard for realistic typos

        Args:
            char: The character to find adjacent keys for

        Returns:
            List of adjacent characters
        """
        # Define a simplified keyboard layout
        keyboard = {
            'q': ['w', 'a', '1'],
            'w': ['q', 'e', 'a', 's', '2'],
            'e': ['w', 'r', 's', 'd', '3'],
            'r': ['e', 't', 'd', 'f', '4'],
            't': ['r', 'y', 'f', 'g', '5'],
            'y': ['t', 'u', 'g', 'h', '6'],
            'u': ['y', 'i', 'h', 'j', '7'],
            'i': ['u', 'o', 'j', 'k', '8'],
            'o': ['i', 'p', 'k', 'l', '9'],
            'p': ['o', '[', 'l', ';', '0'],
            'a': ['q', 'w', 's', 'z'],
            's': ['w', 'e', 'a', 'd', 'z', 'x'],
            'd': ['e', 'r', 's', 'f', 'x', 'c'],
            'f': ['r', 't', 'd', 'g', 'c', 'v'],
            'g': ['t', 'y', 'f', 'h', 'v', 'b'],
            'h': ['y', 'u', 'g', 'j', 'b', 'n'],
            'j': ['u', 'i', 'h', 'k', 'n', 'm'],
            'k': ['i', 'o', 'j', 'l', 'm', ','],
            'l': ['o', 'p', 'k', ';', ',', '.'],
            'z': ['a', 's', 'x'],
            'x': ['z', 's', 'd', 'c'],
            'c': ['x', 'd', 'f', 'v'],
            'v': ['c', 'f', 'g', 'b'],
            'b': ['v', 'g', 'h', 'n'],
            'n': ['b', 'h', 'j', 'm'],
            'm': ['n', 'j', 'k', ','],
            ',': ['m', 'k', 'l', '.'],
            '.': [',', 'l', ';', '/'],
            '1': ['q', '2'],
            '2': ['1', 'q', 'w', '3'],
            '3': ['2', 'w', 'e', '4'],
            '4': ['3', 'e', 'r', '5'],
            '5': ['4', 'r', 't', '6'],
            '6': ['5', 't', 'y', '7'],
            '7': ['6', 'y', 'u', '8'],
            '8': ['7', 'u', 'i', '9'],
            '9': ['8', 'i', 'o', '0'],
            '0': ['9', 'o', 'p', '-'],
            ' ': ['c', 'v', 'b', 'n', 'm']  # Space bar is below these keys
        }

        char_lower = char.lower()
        if char_lower in keyboard:
            return keyboard[char_lower]
        return []

    def press_key(self, key: str) -> bool:
        """
        Press a key

        Args:
            key: Key to press (e.g., 'enter', 'tab', 'a', 'ctrl')

        Returns:
            True if successful, False otherwise
        """
        if not self.available:
            print("⚠️ Computer control not available. Install pyautogui.")
            return False

        try:
            pyautogui.press(key)
            return True
        except Exception as e:
            print(f"⚠️ Key press failed: {e}")
            return False

    def hotkey(self, *keys) -> bool:
        """
        Press a combination of keys

        Args:
            *keys: Keys to press together (e.g., 'ctrl', 'c')

        Returns:
            True if successful, False otherwise
        """
        if not self.available:
            print("⚠️ Computer control not available. Install pyautogui.")
            return False

        try:
            pyautogui.hotkey(*keys)
            return True
        except Exception as e:
            print(f"⚠️ Hotkey failed: {e}")
            return False

    def get_screen_size(self) -> Tuple[int, int]:
        """
        Get the screen size

        Returns:
            Tuple of (width, height)
        """
        if not self.available:
            print("⚠️ Computer control not available. Install pyautogui.")
            return (0, 0)

        try:
            return pyautogui.size()
        except Exception as e:
            print(f"⚠️ Failed to get screen size: {e}")
            return (0, 0)

    def get_mouse_position(self) -> Tuple[int, int]:
        """
        Get the current mouse position

        Returns:
            Tuple of (x, y)
        """
        if not self.available:
            print("⚠️ Computer control not available. Install pyautogui.")
            return (0, 0)

        try:
            return pyautogui.position()
        except Exception as e:
            print(f"⚠️ Failed to get mouse position: {e}")
            return (0, 0)


class OCRCapability:
    """Provides OCR capabilities for reading text from screen or images"""

    def __init__(self):
        self.available = OCR_AVAILABLE
        if self.available:
            # Configure pytesseract
            if sys.platform.startswith('win'):
                # For Windows, pytesseract needs the path to the executable
                pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    def capture_screen(self) -> Optional[PIL.Image.Image]:
        """
        Capture the entire screen

        Returns:
            PIL Image object of the screen or None if failed
        """
        if not self.available:
            print("⚠️ OCR not available. Install pytesseract and Pillow.")
            return None

        try:
            screenshot = PIL.ImageGrab.grab()
            return screenshot
        except Exception as e:
            print(f"⚠️ Screen capture failed: {e}")
            return None

    def capture_region(self, bbox: Tuple[int, int, int, int]) -> Optional[PIL.Image.Image]:
        """
        Capture a specific region of the screen

        Args:
            bbox: Bounding box (left, top, right, bottom)

        Returns:
            PIL Image object of the region or None if failed
        """
        if not self.available:
            print("⚠️ OCR not available. Install pytesseract and Pillow.")
            return None

        try:
            screenshot = PIL.ImageGrab.grab(bbox=bbox)
            return screenshot
        except Exception as e:
            print(f"⚠️ Region capture failed: {e}")
            return None

    def image_to_text(self, image: PIL.Image.Image, lang: str = 'eng') -> str:
        """
        Extract text from an image using OCR with advanced configuration and post-processing

        This enhanced version:
        1. Uses multiple OCR configurations and combines results
        2. Applies page segmentation modes appropriate for different content types
        3. Uses the best OCR engine mode available
        4. Includes advanced text post-processing for better readability
        5. Applies language model-based correction for common OCR errors
        6. Handles UI elements and structured content better

        Args:
            image: PIL Image object
            lang: Language for OCR (default: English)

        Returns:
            Extracted text from the image, with improved accuracy and readability
        """
        if not self.available:
            return "OCR not available. Install pytesseract and Pillow."

        try:
            # Try multiple OCR configurations for best accuracy

            # Configuration 1: Optimized for single column text
            # PSM 6: Assume a single uniform block of text
            config1 = f"--psm 6 --oem 3 -l {lang}"
            text1 = pytesseract.image_to_string(image, lang=lang, config=config1)

            # Configuration 2: Optimized for detecting text in UI elements
            # PSM 11: Sparse text. Find as much text as possible in no particular order
            config2 = f"--psm 11 --oem 3 -l {lang}"
            text2 = pytesseract.image_to_string(image, lang=lang, config=config2)

            # Configuration 3: Optimized for mixed content
            # PSM 3: Fully automatic page segmentation, but no OSD (default)
            config3 = f"--psm 3 --oem 3 -l {lang}"
            text3 = pytesseract.image_to_string(image, lang=lang, config=config3)

            # Configuration 4: Optimized for single line text (good for UI elements)
            # PSM 7: Treat the image as a single line of text
            config4 = f"--psm 7 --oem 3 -l {lang}"
            text4 = pytesseract.image_to_string(image, lang=lang, config=config4)

            # Configuration 5: Optimized for single word text (good for buttons, labels)
            # PSM 8: Treat the image as a single word
            config5 = f"--psm 8 --oem 3 -l {lang}"
            text5 = pytesseract.image_to_string(image, lang=lang, config=config5)

            # Advanced text cleaning and correction
            def advanced_clean_text(text):
                if not text:
                    return ""

                # Basic cleaning
                # Replace multiple newlines with a single one
                text = re.sub(r'\n+', '\n', text)
                # Replace multiple spaces with a single one
                text = re.sub(r' +', ' ', text)
                # Remove non-printable characters
                text = re.sub(r'[^\x20-\x7E\n]', '', text)

                # Advanced corrections for common OCR errors
                # Fix common character confusions
                replacements = {
                    # Common OCR errors
                    'l1': 'li', 'Il': 'Il', '0O': 'OO', '0o': 'oo',
                    # Fix broken words (examples)
                    'i ng': 'ing', 't he': 'the', 'a nd': 'and',
                    # Fix common UI element text
                    'S ubmit': 'Submit', 'C ancel': 'Cancel', 'O pen': 'Open',
                    'S ave': 'Save', 'F ile': 'File', 'E dit': 'Edit',
                    'S ettings': 'Settings', 'P references': 'Preferences'
                }

                # Apply replacements
                for error, correction in replacements.items():
                    # Only replace if the error is a standalone word or part of a word
                    text = re.sub(r'\b' + error + r'\b', correction, text)

                # Fix spacing around punctuation
                text = re.sub(r'\s+([.,;:!?)])', r'\1', text)
                text = re.sub(r'([({[])\s+', r'\1', text)

                # Normalize whitespace around newlines
                text = re.sub(r'\s*\n\s*', '\n', text)

                return text.strip()

            # Apply advanced cleaning to all texts
            text1 = advanced_clean_text(text1)
            text2 = advanced_clean_text(text2)
            text3 = advanced_clean_text(text3)
            text4 = advanced_clean_text(text4)
            text5 = advanced_clean_text(text5)

            # Enhanced scoring system for text quality
            def enhanced_score_text(text):
                if not text:
                    return 0

                # Basic metrics
                length_score = len(text) * 0.1

                # Character quality metrics
                alpha_ratio = sum(c.isalnum() for c in text) / max(1, len(text))
                alpha_score = alpha_ratio * 10

                # Word quality metrics
                words = text.split()
                if not words:
                    return length_score + alpha_score

                avg_word_length = sum(len(w) for w in words) / len(words)
                word_length_score = 5 if 3 <= avg_word_length <= 10 else 0

                # Check for dictionary words (simple approximation)
                common_words = {'the', 'and', 'a', 'to', 'of', 'in', 'is', 'it', 'you', 'that', 'was', 'for', 'on', 'are', 'with', 'as', 'this', 'at', 'be', 'by', 'have', 'from', 'or', 'one', 'had', 'not', 'but', 'what', 'all', 'were', 'when', 'we', 'there', 'can', 'an', 'your', 'which', 'their', 'said', 'if', 'do', 'will', 'each', 'about', 'how', 'up', 'out', 'them', 'then', 'she', 'many', 'some', 'so', 'these', 'would', 'other', 'into', 'has', 'more', 'her', 'two', 'like', 'him', 'see', 'time', 'could', 'no', 'make', 'than', 'first', 'been', 'its', 'who', 'now', 'people', 'my', 'made', 'over', 'did', 'down', 'only', 'way', 'find', 'use', 'may', 'water', 'long', 'little', 'very', 'after', 'words', 'called', 'just', 'where', 'most', 'know'}

                # Count common words
                common_word_count = sum(1 for word in words if word.lower() in common_words)
                common_word_ratio = common_word_count / len(words)
                dictionary_score = common_word_ratio * 15

                # UI element detection (buttons, menus, etc.)
                ui_elements = {'ok', 'cancel', 'submit', 'save', 'open', 'close', 'file', 'edit', 'view', 'help', 'tools', 'options', 'settings', 'preferences', 'menu', 'start', 'search', 'find', 'new', 'delete', 'copy', 'paste', 'cut', 'undo', 'redo', 'print', 'exit', 'quit', 'yes', 'no'}
                ui_element_count = sum(1 for word in words if word.lower() in ui_elements)
                ui_score = ui_element_count * 3  # Boost UI element detection

                # Coherence score - check if words make sense together
                coherence_score = 0
                if len(words) > 3:
                    # Simple bigram check
                    common_bigrams = {('in', 'the'), ('of', 'the'), ('to', 'the'), ('on', 'the'), ('for', 'the'), ('with', 'the'), ('at', 'the'), ('is', 'a'), ('this', 'is'), ('there', 'is'), ('it', 'is')}
                    bigram_count = sum(1 for i in range(len(words)-1) if (words[i].lower(), words[i+1].lower()) in common_bigrams)
                    coherence_score = bigram_count * 2

                return length_score + alpha_score + word_length_score + dictionary_score + ui_score + coherence_score

            # Score all texts
            score1 = enhanced_score_text(text1)
            score2 = enhanced_score_text(text2)
            score3 = enhanced_score_text(text3)
            score4 = enhanced_score_text(text4)
            score5 = enhanced_score_text(text5)

            # Log scores for debugging
            print(f"👁️ OCR scores - PSM6: {score1:.1f}, PSM11: {score2:.1f}, PSM3: {score3:.1f}, PSM7: {score4:.1f}, PSM8: {score5:.1f}")

            # Choose the best result based on scores
            all_texts = [(text1, score1), (text2, score2), (text3, score3), (text4, score4), (text5, score5)]
            all_texts.sort(key=lambda x: x[1], reverse=True)
            best_text = all_texts[0][0]

            # If the best text is very short but another text is much longer and has a decent score,
            # prefer the longer text (helps with context)
            if len(best_text) < 20:  # Very short text
                for text, score in all_texts[1:]:
                    if len(text) > 3 * len(best_text) and score > all_texts[0][1] * 0.7:
                        best_text = text
                        break

            # If all results are poor, try a hybrid approach
            if max(score1, score2, score3, score4, score5) < 10:
                # Combine unique lines from all texts with reasonable scores
                all_lines = []
                seen_lines = set()

                # Sort texts by score
                for text, score in all_texts:
                    if score > 3:  # Only use texts with reasonable scores
                        for line in text.split('\n'):
                            line = line.strip()
                            # Use a normalized version for deduplication
                            normalized = re.sub(r'\s+', ' ', line.lower())
                            if normalized and normalized not in seen_lines and len(normalized) > 1:
                                all_lines.append(line)
                                seen_lines.add(normalized)

                if all_lines:
                    hybrid_text = '\n'.join(all_lines)

                    # If hybrid text is significantly better, use it
                    if len(hybrid_text) > len(best_text) * 1.2:
                        best_text = hybrid_text

            # Final post-processing for readability
            best_text = self._post_process_ocr_text(best_text)

            return best_text
        except Exception as e:
            return f"OCR processing failed: {e}"

    def screen_to_text(self, region: Optional[Tuple[int, int, int, int]] = None, lang: str = 'eng', save_screenshot: bool = False) -> str:
        """
        Capture screen or region and extract text with advanced processing and screenshot saving

        This enhanced version:
        1. Tries multiple enhancement techniques
        2. Uses a sophisticated approach to select the best result
        3. Provides better error handling and fallback options
        4. Includes detailed diagnostic information
        5. Can save screenshots for future reference
        6. Uses advanced post-processing for better readability

        Args:
            region: Optional bounding box (left, top, right, bottom)
            lang: Language for OCR (default: English)
            save_screenshot: Whether to save the screenshot for future reference

        Returns:
            Extracted text from the screen or region with improved accuracy and readability
        """
        if not self.available:
            return "OCR not available. Install pytesseract and Pillow."

        try:
            # Capture the screen or region
            if region:
                print(f"👁️ Capturing screen region: {region}")
                screenshot = self.capture_region(region)
            else:
                print(f"👁️ Capturing full screen")
                screenshot = self.capture_screen()

            if screenshot:
                # Save the screenshot if requested
                if save_screenshot:
                    screenshot_path = self.save_screenshot(screenshot, "ocr")
                    print(f"📸 Screenshot saved for OCR reference: {screenshot_path}")

                # Try multiple approaches and use the best result
                results = []

                # Approach 1: Enhanced image with our advanced algorithm
                enhanced_image = self.enhance_image_for_ocr(screenshot)
                enhanced_text = self.image_to_text(enhanced_image, lang)
                if enhanced_text.strip():
                    results.append(("enhanced", enhanced_text))

                # Approach 2: Original image without enhancement
                # Sometimes the original image works better, especially for colored text
                original_text = self.image_to_text(screenshot, lang)
                if original_text.strip():
                    results.append(("original", original_text))

                # Approach 3: Grayscale with moderate contrast adjustment
                # This can work well for low-contrast text
                from PIL import ImageEnhance
                gray = screenshot.convert('L')
                contrast_enhancer = ImageEnhance.Contrast(gray)
                moderate_contrast = contrast_enhancer.enhance(1.5)
                moderate_text = self.image_to_text(moderate_contrast, lang)
                if moderate_text.strip():
                    results.append(("moderate", moderate_text))

                # Approach 4: Inverted colors (can help with dark backgrounds)
                from PIL import ImageOps
                inverted = ImageOps.invert(screenshot.convert('RGB'))
                inverted_text = self.image_to_text(inverted, lang)
                if inverted_text.strip():
                    results.append(("inverted", inverted_text))

                # If we have results, select the best one based on quality metrics
                if results:
                    # First, try to identify if this is UI text or regular text
                    # UI text often contains buttons, menus, etc.
                    ui_elements = {'ok', 'cancel', 'submit', 'save', 'open', 'close', 'file', 'edit', 
                                  'view', 'help', 'tools', 'options', 'settings', 'preferences', 
                                  'menu', 'start', 'search', 'find', 'new', 'delete', 'copy', 'paste'}

                    # Check if any result contains UI elements
                    contains_ui = False
                    for _, text in results:
                        words = set(word.lower() for word in re.findall(r'\b\w+\b', text))
                        if any(ui_word in words for ui_word in ui_elements):
                            contains_ui = True
                            break

                    if contains_ui:
                        # For UI text, prefer approaches that work well with UI elements
                        # PSM 11 (sparse text) often works better for UI
                        print("👁️ Detected UI elements in text, optimizing for UI recognition")

                        # Re-sort results, giving preference to approaches that work well with UI
                        ui_preference = {"enhanced": 1.2, "inverted": 1.1, "original": 1.0, "moderate": 0.9}
                        results.sort(key=lambda x: len(x[1]) * ui_preference.get(x[0], 1.0), reverse=True)
                    else:
                        # For regular text, sort by text length as before
                        results.sort(key=lambda x: len(x[1]), reverse=True)

                    best_approach, best_text = results[0]

                    # Log which approach worked best
                    print(f"👁️ Best OCR result from {best_approach} approach")

                    # Apply final post-processing for readability
                    final_text = self._post_process_ocr_text(best_text)

                    # If the text is very short, try to combine results
                    if len(final_text) < 20 and len(results) > 1:
                        print("👁️ Text is very short, attempting to combine results")
                        combined_text = self._combine_ocr_results([text for _, text in results])
                        if len(combined_text) > len(final_text) * 1.5:
                            final_text = combined_text
                            print("👁️ Using combined OCR results for better coverage")

                    return final_text
                else:
                    # If all approaches failed, provide a more informative message
                    return "OCR completed but no text was detected in the image. The screen may not contain readable text or the text may be in a format that's difficult for OCR to recognize."
            else:
                return "Screen capture failed. Please check if the screen capture functionality is working correctly."
        except Exception as e:
            print(f"❌ OCR Error: {str(e)}")
            return f"Screen to text conversion failed: {e}. Please check if the OCR functionality is working correctly."

    def _combine_ocr_results(self, texts: List[str]) -> str:
        """
        Combine multiple OCR results into a single coherent text

        Args:
            texts: List of OCR results to combine

        Returns:
            Combined text with duplicates removed
        """
        if not texts:
            return ""

        # Remove empty texts
        texts = [t for t in texts if t.strip()]
        if not texts:
            return ""

        # If only one text, return it
        if len(texts) == 1:
            return texts[0]

        # Split texts into lines and remove duplicates
        all_lines = []
        seen_lines = set()

        for text in texts:
            for line in text.split('\n'):
                line = line.strip()
                if not line:
                    continue

                # Normalize for deduplication
                normalized = re.sub(r'\s+', ' ', line.lower())
                if normalized and normalized not in seen_lines:
                    all_lines.append(line)
                    seen_lines.add(normalized)

        # Sort lines by length (longer lines first)
        all_lines.sort(key=len, reverse=True)

        # Join lines and post-process
        combined = '\n'.join(all_lines)
        return self._post_process_ocr_text(combined)

    def _post_process_ocr_text(self, text: str) -> str:
        """
        Perform final post-processing on OCR text to improve readability

        This method applies advanced language-based corrections and formatting
        to make the OCR text more readable and accurate.

        Args:
            text: The OCR text to post-process

        Returns:
            Post-processed text with improved readability
        """
        if not text:
            return ""

        try:
            # Fix common OCR errors that weren't caught in the basic cleaning

            # Fix common misrecognized characters
            char_replacements = {
                # Zero vs letter O
                r'\b0\b': 'O',  # Standalone zero to letter O
                r'\bO(?=\d)': '0',  # Letter O followed by digit to zero

                # One vs letter I/l
                r'\b1(?=[a-zA-Z])': 'I',  # Digit 1 followed by letter to I
                r'\bl(?=\d)': '1',  # Letter l followed by digit to 1

                # Fix common punctuation errors
                r'\.\.\.\.+': '...',  # Multiple periods to ellipsis
                r'\,\,': ',',  # Double commas
                r'\.\.': '.',  # Double periods

                # Fix common UI text errors
                r'(?i)\bsubrnit\b': 'Submit',
                r'(?i)\bcancei\b': 'Cancel',
                r'(?i)\bciose\b': 'Close',
                r'(?i)\bciick\b': 'Click',
                r'(?i)\boperi\b': 'Open'
            }

            for pattern, replacement in char_replacements.items():
                text = re.sub(pattern, replacement, text)

            # Fix broken words (more comprehensive than in basic cleaning)
            word_replacements = {
                r't he\b': 'the',
                r'a nd\b': 'and',
                r'o f\b': 'of',
                r'f or\b': 'for',
                r'w ith\b': 'with',
                r'i n\b': 'in',
                r'o n\b': 'on',
                r'a t\b': 'at',
                r'i s\b': 'is',
                r'a re\b': 'are',
                r'w as\b': 'was',
                r'b e\b': 'be',
                r'h ave\b': 'have',
                r'c an\b': 'can',
                r'w ill\b': 'will',
                r'd o\b': 'do',
                r'i f\b': 'if',
                r'b ut\b': 'but',
                r'n ot\b': 'not',
                r'w hat\b': 'what',
                r'w hen\b': 'when',
                r'w here\b': 'where',
                r'w ho\b': 'who',
                r'w hy\b': 'why',
                r'h ow\b': 'how'
            }

            for pattern, replacement in word_replacements.items():
                text = re.sub(pattern, replacement, text)

            # Fix capitalization in sentences
            sentences = re.split(r'([.!?]\s+)', text)
            for i in range(0, len(sentences), 2):
                if i < len(sentences) and sentences[i]:
                    sentences[i] = sentences[i][0].upper() + sentences[i][1:]
            text = ''.join(sentences)

            # Fix UI element capitalization (buttons, menus, etc.)
            ui_elements = ['ok', 'cancel', 'submit', 'save', 'open', 'close', 'file', 'edit', 
                          'view', 'help', 'tools', 'options', 'settings', 'preferences',
                          'menu', 'start', 'search', 'find', 'new', 'delete', 'copy', 'paste',
                          'cut', 'undo', 'redo', 'print', 'exit', 'quit', 'yes', 'no']

            for element in ui_elements:
                # Only capitalize standalone UI elements
                text = re.sub(r'\b' + element + r'\b', element.capitalize(), text, flags=re.IGNORECASE)

            # Ensure proper spacing around special characters
            text = re.sub(r'\s+([.,;:!?)])', r'\1', text)  # No space before punctuation
            text = re.sub(r'([.,;:!?])(?=[a-zA-Z])', r'\1 ', text)  # Space after punctuation if followed by letter
            text = re.sub(r'([({[])\s+', r'\1', text)  # No space after opening brackets

            # Remove any remaining excessive whitespace
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'\s*\n\s*', '\n', text)

            # Final cleanup
            text = text.strip()

            return text

        except Exception as e:
            print(f"⚠️ Error in OCR post-processing: {e}")
            return text  # Return original text if post-processing fails

    def save_screenshot(self, image: PIL.Image.Image, prefix: str = "screenshot") -> str:
        """
        Save a screenshot to disk for future reference

        Args:
            image: The PIL Image to save
            prefix: Prefix for the filename

        Returns:
            Path to the saved screenshot
        """
        try:
            # Create screenshots directory if it doesn't exist
            screenshots_dir = os.path.join(os.path.expanduser("~"), "doba_screenshots")
            os.makedirs(screenshots_dir, exist_ok=True)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}_{timestamp}.png"
            filepath = os.path.join(screenshots_dir, filename)

            # Save the image
            image.save(filepath)
            print(f"📸 Screenshot saved to {filepath}")
            return filepath
        except Exception as e:
            print(f"❌ Error saving screenshot: {e}")
            return ""

    def find_ui_element_by_image(self, template_path: str, threshold: float = 0.8) -> Optional[Tuple[int, int]]:
        """
        Find a UI element on screen using template matching

        This method uses OpenCV template matching to find UI elements like icons

        Args:
            template_path: Path to the template image (e.g., Firefox icon)
            threshold: Matching threshold (0.0-1.0, higher is stricter)

        Returns:
            Tuple of (x, y) coordinates of the center of the matched element, or None if not found
        """
        try:
            import cv2
            import numpy as np

            # Capture the screen
            screenshot = self.capture_screen()
            if screenshot is None:
                return None

            # Convert PIL image to OpenCV format
            screenshot_cv = np.array(screenshot)
            screenshot_cv = cv2.cvtColor(screenshot_cv, cv2.COLOR_RGB2BGR)

            # Load the template
            template = cv2.imread(template_path)
            if template is None:
                print(f"❌ Could not load template image: {template_path}")
                return None

            # Get template dimensions
            h, w = template.shape[:2]

            # Perform template matching
            result = cv2.matchTemplate(screenshot_cv, template, cv2.TM_CCOEFF_NORMED)

            # Find locations where the matching exceeds the threshold
            locations = np.where(result >= threshold)

            if len(locations[0]) > 0:
                # Get the best match
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                # Calculate center of the match
                center_x = max_loc[0] + w // 2
                center_y = max_loc[1] + h // 2

                print(f"🔍 Found UI element at ({center_x}, {center_y}) with confidence {max_val:.2f}")
                return (center_x, center_y)
            else:
                print(f"⚠️ UI element not found on screen (threshold: {threshold})")
                return None

        except Exception as e:
            print(f"❌ Error finding UI element: {e}")
            return None

    def enhance_image_for_ocr(self, image: PIL.Image.Image) -> PIL.Image.Image:
        """
        Enhance image for better OCR results using advanced image processing techniques

        This improved version uses more sophisticated techniques to enhance text readability:
        1. Resize the image to improve resolution
        2. Use more moderate contrast enhancement
        3. Apply adaptive thresholding instead of fixed threshold
        4. Provide multiple processing paths for different types of content

        Args:
            image: PIL Image object

        Returns:
            Enhanced PIL Image object
        """
        if not self.available:
            return image

        try:
            # Import necessary modules
            from PIL import ImageEnhance, ImageFilter, Image

            # Make a copy of the original image
            original = image.copy()

            # Step 1: Resize the image to improve resolution (if it's small)
            width, height = image.size
            if width < 1000 or height < 1000:
                scale_factor = 2.0
                image = image.resize((int(width * scale_factor), int(height * scale_factor)), Image.LANCZOS)

            # Step 2: Convert to grayscale
            gray = image.convert('L')

            try:
                import numpy as np
                import cv2

                # Convert PIL image to numpy array for OpenCV processing
                img_array = np.array(gray)

                # Apply multiple enhancement techniques and combine results

                # Path 1: Adaptive thresholding for text-heavy content
                # This works well for documents, UI elements, etc.
                adaptive_thresh = cv2.adaptiveThreshold(
                    img_array, 
                    255, 
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY, 
                    11,  # Block size
                    2    # Constant subtracted from mean
                )

                # Path 2: Moderate contrast enhancement + Otsu thresholding
                # This works well for mixed content
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                clahe_img = clahe.apply(img_array)
                _, otsu_thresh = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # Path 3: Edge enhancement for detecting boundaries
                # This helps with text that's part of graphics or has colored backgrounds
                edges = cv2.Canny(img_array, 100, 200)
                dilated_edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)

                # Combine the results (weighted average)
                # We give more weight to the adaptive threshold result as it's usually best for text
                combined = cv2.addWeighted(adaptive_thresh, 0.6, otsu_thresh, 0.4, 0)

                # Convert back to PIL Image
                enhanced_image = Image.fromarray(combined)

                print("👁️ Applied advanced image enhancement for OCR with adaptive thresholding")
                return enhanced_image

            except (ImportError, Exception) as cv_error:
                # Fallback to PIL-only processing if OpenCV is not available
                print(f"⚠️ Advanced enhancement failed: {str(cv_error)}, using PIL-only enhancement")

                # Step 3: Apply a very slight blur to reduce noise (less aggressive)
                denoised = gray.filter(ImageFilter.GaussianBlur(radius=0.5))

                # Step 4: Increase contrast (more moderate than before)
                contrast_enhancer = ImageEnhance.Contrast(denoised)
                high_contrast = contrast_enhancer.enhance(1.8)  # Reduced from 2.5

                # Step 5: Increase brightness slightly
                brightness_enhancer = ImageEnhance.Brightness(high_contrast)
                brightened = brightness_enhancer.enhance(1.3)  # Increased from 1.2

                # Step 6: Increase sharpness
                sharpness_enhancer = ImageEnhance.Sharpness(brightened)
                sharpened = sharpness_enhancer.enhance(1.8)  # Increased from 1.5

                try:
                    # If numpy is available, use a more sophisticated thresholding approach
                    import numpy as np
                    img_array = np.array(sharpened)

                    # Calculate a dynamic threshold based on image statistics
                    mean_val = np.mean(img_array)
                    std_val = np.std(img_array)
                    threshold_value = max(min(mean_val - 0.5 * std_val, 180), 100)  # Keep in reasonable range

                    binary_array = (img_array > threshold_value) * 255
                    enhanced_image = Image.fromarray(binary_array.astype(np.uint8))

                    print("👁️ Applied improved PIL-based enhancement for OCR")
                    return enhanced_image
                except ImportError:
                    # If numpy is not available, return the sharpened image
                    print("👁️ Applied basic image enhancement for OCR (numpy not available)")
                    return sharpened

        except Exception as e:
            print(f"⚠️ Image enhancement failed: {str(e)}, using original image")
            return image


# Main class that integrates all extensions
class FileOperations:
    """Provides file operation capabilities"""

    def __init__(self):
        self.available = True

    def read_file(self, file_path: str) -> Dict[str, Any]:
        """
        Read content from a file

        Args:
            file_path: Path to the file to read

        Returns:
            Dictionary with content and status
        """
        try:
            with open(file_path, 'r') as file:
                content = file.read()
                return {
                    "success": True,
                    "content": content,
                    "error": None
                }
        except Exception as e:
            return {
                "success": False,
                "content": None,
                "error": str(e)
            }

    def write_file(self, file_path: str, content: str, append: bool = False) -> Dict[str, Any]:
        """
        Write content to a file

        Args:
            file_path: Path to the file to write
            content: Content to write
            append: Whether to append to the file

        Returns:
            Dictionary with status
        """
        try:
            mode = 'a' if append else 'w'
            with open(file_path, mode) as file:
                file.write(content)
                return {
                    "success": True,
                    "error": None
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def list_directory(self, directory_path: str = '.') -> Dict[str, Any]:
        """
        List contents of a directory

        Args:
            directory_path: Path to the directory

        Returns:
            Dictionary with directory contents and status
        """
        try:
            contents = os.listdir(directory_path)
            files = []
            directories = []

            for item in contents:
                full_path = os.path.join(directory_path, item)
                if os.path.isfile(full_path):
                    files.append(item)
                elif os.path.isdir(full_path):
                    directories.append(item)

            return {
                "success": True,
                "files": files,
                "directories": directories,
                "error": None
            }
        except Exception as e:
            return {
                "success": False,
                "files": [],
                "directories": [],
                "error": str(e)
            }

    def file_exists(self, file_path: str) -> bool:
        """
        Check if a file exists

        Args:
            file_path: Path to the file

        Returns:
            True if the file exists, False otherwise
        """
        return os.path.isfile(file_path)

    def directory_exists(self, directory_path: str) -> bool:
        """
        Check if a directory exists

        Args:
            directory_path: Path to the directory

        Returns:
            True if the directory exists, False otherwise
        """
        return os.path.isdir(directory_path)


class CodeAnalysis:
    """Provides code analysis capabilities for reading, analyzing, and critiquing code"""

    def __init__(self):
        self.available = True
        # Common programming file extensions
        self.code_extensions = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.html': 'HTML',
            '.css': 'CSS',
            '.java': 'Java',
            '.c': 'C',
            '.cpp': 'C++',
            '.h': 'C/C++ Header',
            '.cs': 'C#',
            '.php': 'PHP',
            '.rb': 'Ruby',
            '.go': 'Go',
            '.rs': 'Rust',
            '.ts': 'TypeScript',
            '.swift': 'Swift',
            '.kt': 'Kotlin',
            '.sh': 'Shell',
            '.bat': 'Batch',
            '.ps1': 'PowerShell',
            '.sql': 'SQL',
            '.json': 'JSON',
            '.xml': 'XML',
            '.yaml': 'YAML',
            '.yml': 'YAML',
            '.md': 'Markdown',
            '.txt': 'Text'
        }
        # Common code quality metrics
        self.quality_metrics = [
            'readability',
            'maintainability',
            'complexity',
            'documentation',
            'error handling',
            'performance',
            'security',
            'testability'
        ]

    def read_code_file(self, file_path: str) -> Dict[str, Any]:
        """
        Read a code file and return its contents

        Args:
            file_path: Path to the code file

        Returns:
            Dictionary with file contents, language, and status
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            # Determine the language based on file extension
            _, ext = os.path.splitext(file_path)
            language = self.code_extensions.get(ext.lower(), 'Unknown')

            return {
                "success": True,
                "content": content,
                "language": language,
                "error": None
            }
        except Exception as e:
            return {
                "success": False,
                "content": "",
                "language": "Unknown",
                "error": str(e)
            }

    def analyze_code_structure(self, code: str, language: str) -> Dict[str, Any]:
        """
        Analyze the structure of code (classes, functions, imports, etc.)

        Args:
            code: The code to analyze
            language: The programming language of the code

        Returns:
            Dictionary with code structure information
        """
        try:
            result = {
                "success": True,
                "language": language,
                "imports": [],
                "classes": [],
                "functions": [],
                "variables": [],
                "lines_of_code": len(code.splitlines()),
                "error": None
            }

            # Basic analysis based on language
            if language == 'Python':
                # Extract imports
                import_pattern = r'^import\s+(\w+)|^from\s+(\w+).*import'
                result["imports"] = list(set(re.findall(import_pattern, code, re.MULTILINE)))

                # Extract classes
                class_pattern = r'class\s+(\w+)'
                result["classes"] = list(set(re.findall(class_pattern, code)))

                # Extract functions
                function_pattern = r'def\s+(\w+)'
                result["functions"] = list(set(re.findall(function_pattern, code)))

                # Extract global variables
                variable_pattern = r'^(\w+)\s*='
                result["variables"] = list(set(re.findall(variable_pattern, code, re.MULTILINE)))

            elif language in ['JavaScript', 'TypeScript']:
                # Extract imports
                import_pattern = r'import\s+.*from\s+[\'"](.+)[\'"]|require\([\'"](.+)[\'"]\)'
                result["imports"] = list(set(re.findall(import_pattern, code)))

                # Extract classes
                class_pattern = r'class\s+(\w+)'
                result["classes"] = list(set(re.findall(class_pattern, code)))

                # Extract functions
                function_pattern = r'function\s+(\w+)|const\s+(\w+)\s*=\s*\(.*\)\s*=>'
                result["functions"] = list(set(re.findall(function_pattern, code)))

                # Extract global variables
                variable_pattern = r'(const|let|var)\s+(\w+)\s*='
                result["variables"] = [match[1] for match in re.findall(variable_pattern, code)]

            # Add more language-specific analysis as needed

            return result
        except Exception as e:
            return {
                "success": False,
                "language": language,
                "error": str(e)
            }

    def critique_code_quality(self, code: str, language: str) -> Dict[str, Any]:
        """
        Critique the quality of code based on best practices

        Args:
            code: The code to critique
            language: The programming language of the code

        Returns:
            Dictionary with code quality critique
        """
        try:
            result = {
                "success": True,
                "language": language,
                "overall_score": 0,
                "metrics": {},
                "suggestions": [],
                "error": None
            }

            # Calculate lines of code
            lines = code.splitlines()
            line_count = len(lines)

            # Initialize metrics with default scores
            for metric in self.quality_metrics:
                result["metrics"][metric] = {
                    "score": 0,
                    "comments": []
                }

            # Basic quality analysis

            # 1. Readability
            readability = result["metrics"]["readability"]

            # Check line length
            long_lines = [i+1 for i, line in enumerate(lines) if len(line) > 100]
            if long_lines:
                readability["comments"].append(f"Found {len(long_lines)} lines longer than 100 characters")
                readability["score"] -= len(long_lines) * 0.1

            # Check indentation consistency
            indent_sizes = set()
            for line in lines:
                if line.strip() and line.startswith(' '):
                    indent_size = len(line) - len(line.lstrip(' '))
                    if indent_size > 0:
                        indent_sizes.add(indent_size)

            if len(indent_sizes) > 1:
                readability["comments"].append(f"Inconsistent indentation: found {indent_sizes} different indent sizes")
                readability["score"] -= 0.5

            # 2. Documentation
            documentation = result["metrics"]["documentation"]

            # Check for docstrings or comments
            comment_lines = sum(1 for line in lines if line.strip().startswith('#') or line.strip().startswith('//'))
            comment_ratio = comment_lines / line_count if line_count > 0 else 0

            if comment_ratio < 0.1:
                documentation["comments"].append("Low comment ratio (less than 10% of code)")
                documentation["score"] -= 0.5

            # 3. Complexity
            complexity = result["metrics"]["complexity"]

            # Check for nested loops and conditionals
            nested_pattern = r'(if|for|while).*:.*\n\s+(if|for|while)'
            nested_count = len(re.findall(nested_pattern, code))
            if nested_count > 5:
                complexity["comments"].append(f"Found {nested_count} nested control structures")
                complexity["score"] -= nested_count * 0.1

            # 4. Error handling
            error_handling = result["metrics"]["error handling"]

            # Check for try-except blocks
            if language == 'Python':
                try_count = len(re.findall(r'try:', code))
                except_count = len(re.findall(r'except', code))

                if try_count == 0:
                    error_handling["comments"].append("No error handling (try-except) found")
                    error_handling["score"] -= 0.5
                elif try_count != except_count:
                    error_handling["comments"].append("Mismatched try-except blocks")
                    error_handling["score"] -= 0.3

            # Calculate overall score based on individual metrics
            # Start with a perfect score and subtract based on issues
            base_score = 10.0
            penalty = sum(abs(metric["score"]) for metric in result["metrics"].values())
            result["overall_score"] = max(0, min(10, base_score - penalty))

            # Generate suggestions based on findings
            for metric, data in result["metrics"].items():
                if data["comments"]:
                    for comment in data["comments"]:
                        result["suggestions"].append(f"{metric.capitalize()}: {comment}")

            return result
        except Exception as e:
            return {
                "success": False,
                "language": language,
                "error": str(e)
            }

    def find_code_files(self, directory_path: str, recursive: bool = True) -> Dict[str, Any]:
        """
        Find code files in a directory

        Args:
            directory_path: Path to the directory
            recursive: Whether to search recursively

        Returns:
            Dictionary with list of code files and their languages
        """
        try:
            code_files = []

            if recursive:
                for root, _, files in os.walk(directory_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        _, ext = os.path.splitext(file)
                        if ext.lower() in self.code_extensions:
                            code_files.append({
                                "path": file_path,
                                "language": self.code_extensions.get(ext.lower(), 'Unknown')
                            })
            else:
                for file in os.listdir(directory_path):
                    file_path = os.path.join(directory_path, file)
                    if os.path.isfile(file_path):
                        _, ext = os.path.splitext(file)
                        if ext.lower() in self.code_extensions:
                            code_files.append({
                                "path": file_path,
                                "language": self.code_extensions.get(ext.lower(), 'Unknown')
                            })

            return {
                "success": True,
                "code_files": code_files,
                "count": len(code_files),
                "error": None
            }
        except Exception as e:
            return {
                "success": False,
                "code_files": [],
                "count": 0,
                "error": str(e)
            }

    def analyze_project(self, project_path: str) -> Dict[str, Any]:
        """
        Analyze an entire project directory

        Args:
            project_path: Path to the project directory

        Returns:
            Dictionary with project analysis
        """
        try:
            # Find all code files
            files_result = self.find_code_files(project_path)
            if not files_result["success"]:
                return {
                    "success": False,
                    "error": files_result["error"]
                }

            code_files = files_result["code_files"]

            # Analyze each file
            file_analyses = []
            total_lines = 0
            languages = {}

            for file_info in code_files:
                file_path = file_info["path"]
                language = file_info["language"]

                # Read the file
                read_result = self.read_code_file(file_path)
                if not read_result["success"]:
                    continue

                # Count lines
                lines = len(read_result["content"].splitlines())
                total_lines += lines

                # Track languages
                languages[language] = languages.get(language, 0) + 1

                # Analyze structure
                structure = self.analyze_code_structure(read_result["content"], language)

                # Add to results
                file_analyses.append({
                    "path": file_path,
                    "language": language,
                    "lines": lines,
                    "structure": structure
                })

            return {
                "success": True,
                "project_path": project_path,
                "file_count": len(code_files),
                "total_lines": total_lines,
                "languages": languages,
                "files": file_analyses,
                "error": None
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


class BrowserAutomation:
    """Provides browser automation capabilities for web search and page reading"""

    def __init__(self):
        self.available = BROWSER_AUTOMATION_AVAILABLE
        self.driver = None
        self.current_url = None
        self.search_engine = "https://www.duckduckgo.com"  # Changed from Google to DuckDuckGo to avoid detection
        self.consecutive_searches = 0
        self.last_search_time = 0
        self.request_queue = []
        self.max_queue_size = 50  # Limit queue size to prevent overload
        self.user_data_dir = "./chrome-data"  # Directory for persistent session data

    def initialize_browser(self, headless: bool = False) -> bool:
        """
        Initialize the browser with enhanced error handling and fallback mechanisms.

        Args:
            headless: Whether to run in headless mode

        Returns:
            True if successful, False otherwise
        """
        if not self.available:
            print("⚠️ Browser automation not available. Install selenium and webdriver_manager.")
            return False

        # Track created directories for cleanup in case of failure
        created_dirs = []

        # Maximum retry attempts
        max_retries = 3

        # Timeout for browser initialization (seconds)
        init_timeout = 30

        # Try Chrome first, then Firefox, then Edge
        browsers_to_try = ["chrome", "firefox", "edge"]

        for browser_type in browsers_to_try:
            for attempt in range(max_retries):
                try:
                    print(f"🔄 Attempting to initialize {browser_type} browser (attempt {attempt+1}/{max_retries})")

                    if browser_type == "chrome":
                        # Set up Chrome options
                        browser_options = Options()
                        if headless:
                            browser_options.add_argument("--headless=new")  # Updated headless mode for newer Chrome
                        browser_options.add_argument("--no-sandbox")
                        browser_options.add_argument("--disable-dev-shm-usage")
                        # Additional options to improve stability
                        browser_options.add_argument("--disable-extensions")
                        browser_options.add_argument("--disable-gpu")
                        browser_options.add_argument("--disable-features=VizDisplayCompositor")
                        browser_options.add_argument("--disable-features=NetworkService")
                        browser_options.add_argument("--window-size=1920,1080")
                        # Add option to ensure browser window is visible
                        browser_options.add_argument("--start-maximized")

                        # Add user-agent rotation to appear as different browsers
                        user_agents = [
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
                            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36 Edg/94.0.992.47",
                            "Mozilla/5.0 (iPhone; CPU iPhone OS 14_7_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Mobile/15E148 Safari/604.1"
                        ]
                        browser_options.add_argument(f"user-agent={random.choice(user_agents)}")

                        # Create a unique user data directory for this browser instance
                        import os
                        import shutil
                        unique_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
                        unique_user_data_dir = f"{self.user_data_dir}_{browser_type}_{unique_id}"

                        if not os.path.exists(unique_user_data_dir):
                            os.makedirs(unique_user_data_dir)
                            created_dirs.append(unique_user_data_dir)

                        browser_options.add_argument(f"--user-data-dir={unique_user_data_dir}")
                        browser_options.add_argument("--profile-directory=Default")

                        # Initialize the Chrome driver with timeout
                        try:
                            from webdriver_manager.chrome import ChromeDriverManager
                            from selenium.webdriver.support.ui import WebDriverWait
                            from selenium.webdriver.support import expected_conditions as EC
                            from selenium.common.exceptions import TimeoutException

                            service = Service(ChromeDriverManager().install())

                            # Use threading to implement timeout for browser initialization
                            driver_initialized = False
                            driver_error = None

                            def init_driver():
                                nonlocal driver_initialized, driver_error
                                try:
                                    self.driver = webdriver.Chrome(service=service, options=browser_options)
                                    driver_initialized = True
                                except Exception as e:
                                    driver_error = e

                            # Start initialization in a separate thread
                            init_thread = threading.Thread(target=init_driver)
                            init_thread.daemon = True
                            init_thread.start()

                            # Wait for initialization with timeout
                            init_thread.join(timeout=init_timeout)

                            if not driver_initialized:
                                if driver_error:
                                    raise Exception(f"Chrome initialization failed: {str(driver_error)}")
                                else:
                                    raise TimeoutException(f"Chrome initialization timed out after {init_timeout} seconds")

                            # Verify browser is responsive
                            WebDriverWait(self.driver, 10).until(
                                EC.presence_of_element_located((By.TAG_NAME, "html"))
                            )

                        except Exception as chrome_error:
                            print(f"⚠️ Error with ChromeDriverManager: {chrome_error}")
                            # Try direct path as fallback
                            try:
                                self.driver = webdriver.Chrome(options=browser_options)
                                # Verify browser is responsive
                                WebDriverWait(self.driver, 10).until(
                                    EC.presence_of_element_located((By.TAG_NAME, "html"))
                                )
                            except Exception as direct_error:
                                # Clean up created directories
                                for dir_path in created_dirs:
                                    try:
                                        if os.path.exists(dir_path):
                                            shutil.rmtree(dir_path)
                                    except Exception as cleanup_error:
                                        print(f"⚠️ Error cleaning up directory {dir_path}: {cleanup_error}")
                                raise Exception(f"Both ChromeDriverManager and direct path failed: {direct_error}")

                    elif browser_type == "firefox":
                        # Try Firefox as fallback
                        try:
                            from selenium.webdriver.firefox.options import Options as FirefoxOptions
                            from selenium.webdriver.firefox.service import Service as FirefoxService
                            from webdriver_manager.firefox import GeckoDriverManager
                            from selenium.webdriver.support.ui import WebDriverWait
                            from selenium.webdriver.support import expected_conditions as EC
                            from selenium.common.exceptions import TimeoutException

                            firefox_options = FirefoxOptions()
                            if headless:
                                firefox_options.add_argument("--headless")

                            # Create a unique Firefox profile directory
                            unique_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
                            unique_profile_dir = f"{self.user_data_dir}_{browser_type}_{unique_id}"

                            if not os.path.exists(unique_profile_dir):
                                os.makedirs(unique_profile_dir)
                                created_dirs.append(unique_profile_dir)

                            firefox_options.set_preference("profile", unique_profile_dir)

                            # Initialize Firefox driver with timeout
                            service = FirefoxService(GeckoDriverManager().install())

                            # Use threading to implement timeout for browser initialization
                            driver_initialized = False
                            driver_error = None

                            def init_driver():
                                nonlocal driver_initialized, driver_error
                                try:
                                    self.driver = webdriver.Firefox(service=service, options=firefox_options)
                                    driver_initialized = True
                                except Exception as e:
                                    driver_error = e

                            # Start initialization in a separate thread
                            init_thread = threading.Thread(target=init_driver)
                            init_thread.daemon = True
                            init_thread.start()

                            # Wait for initialization with timeout
                            init_thread.join(timeout=init_timeout)

                            if not driver_initialized:
                                if driver_error:
                                    raise Exception(f"Firefox initialization failed: {str(driver_error)}")
                                else:
                                    raise TimeoutException(f"Firefox initialization timed out after {init_timeout} seconds")

                            # Verify browser is responsive
                            WebDriverWait(self.driver, 10).until(
                                EC.presence_of_element_located((By.TAG_NAME, "html"))
                            )
                        except ImportError:
                            raise Exception("Firefox webdriver not available. Install geckodriver and firefox-webdriver.")

                    elif browser_type == "edge":
                        # Try Edge as last resort
                        try:
                            from selenium.webdriver.edge.options import Options as EdgeOptions
                            from selenium.webdriver.edge.service import Service as EdgeService
                            from webdriver_manager.microsoft import EdgeChromiumDriverManager

                            edge_options = EdgeOptions()
                            if headless:
                                edge_options.add_argument("--headless")
                            edge_options.add_argument("--no-sandbox")

                            # Create a unique Edge profile directory
                            unique_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
                            unique_profile_dir = f"{self.user_data_dir}_{browser_type}_{unique_id}"

                            if not os.path.exists(unique_profile_dir):
                                os.makedirs(unique_profile_dir)
                                created_dirs.append(unique_profile_dir)

                            edge_options.add_argument(f"--user-data-dir={unique_profile_dir}")

                            # Initialize Edge driver
                            service = EdgeService(EdgeChromiumDriverManager().install())
                            self.driver = webdriver.Edge(service=service, options=edge_options)
                        except ImportError:
                            raise Exception("Edge webdriver not available. Install msedgedriver.")

                    # Test if browser is working by loading a simple page
                    self.driver.set_page_load_timeout(30)
                    self.driver.get("about:blank")

                    print(f"✅ Browser ({browser_type}) initialized successfully on attempt {attempt+1}")
                    return True

                except Exception as e:
                    print(f"⚠️ Failed to initialize {browser_type} browser on attempt {attempt+1}: {e}")

                    # Close driver if it was created
                    if hasattr(self, 'driver') and self.driver is not None:
                        try:
                            self.driver.quit()
                        except:
                            pass
                        self.driver = None

                    # If this is the last attempt for this browser type, clean up and try next browser
                    if attempt == max_retries - 1:
                        # Clean up created directories
                        for dir_path in created_dirs:
                            try:
                                if os.path.exists(dir_path):
                                    shutil.rmtree(dir_path)
                                    print(f"🧹 Cleaned up directory: {dir_path}")
                            except Exception as cleanup_error:
                                print(f"⚠️ Error cleaning up directory {dir_path}: {cleanup_error}")

                        # Clear the list for the next browser type
                        created_dirs = []
                    else:
                        # Wait before retrying
                        time.sleep(2)

        # If we get here, all browsers failed
        print("❌ All browser initialization attempts failed")
        return False

    def close_browser(self) -> bool:
        """
        Close the browser

        Returns:
            True if successful, False otherwise
        """
        if self.driver is None:
            return True

        try:
            self.driver.quit()
            self.driver = None
            self.current_url = None
            print("✅ Browser closed successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to close browser: {e}")
            return False

    def search(self, query: str, max_results: int = 5, priority: int = 1) -> List[Dict[str, str]]:
        """
        Search the web using a browser with human-like behavior

        This method adds the request to a priority queue and processes it based on priority.
        High priority requests are processed immediately, while lower priority requests
        may be batched or deferred.

        Args:
            query: The search query
            max_results: Maximum number of results to return
            priority: Priority of the search request (1-10, higher is more important)

        Returns:
            List of search results with title, snippet, and url
        """
        if not self.available:
            return [{"error": "Browser automation not available. Install selenium and webdriver_manager."}]

        # Add request to queue with priority and context
        request_id = str(uuid.uuid4())
        request_data = {
            "id": request_id,
            "query": query,
            "max_results": max_results,
            "priority": priority,
            "timestamp": time.time(),
            "context": self._get_request_context(query),
            "status": "pending"
        }

        # Add to queue and manage queue size
        self._add_to_queue(request_data)

        # Initialize the processing history if it doesn't exist
        if not hasattr(self, 'processing_history'):
            self.processing_history = []

        # Check for repetitive patterns and adjust behavior if needed
        if hasattr(self, 'browsing_history') and len(self.browsing_history) >= 3:
            patterns = self.analyze_browsing_patterns()
            repetitiveness_score = patterns.get("repetitiveness_score", 0)

            # If we detect high repetitiveness, take corrective action
            if repetitiveness_score > 60:
                print(f"⚠️ High repetitiveness detected (score: {repetitiveness_score}). Adjusting behavior.")

                # Log the recommendations
                for recommendation in patterns.get("recommendations", []):
                    print(f"🔄 {recommendation}")

                # If we're repeatedly searching without clicking, force a higher click probability
                if patterns.get("search_only_behavior", False):
                    print("🖱️ Encouraging more clicking on search results")
                    # Set a flag to increase click probability
                    self.encourage_clicking = True

                # If we're repeatedly visiting the same URLs, encourage exploration
                if patterns.get("repetitive_urls", []):
                    print("🔍 Encouraging exploration of new content")
                    # Set a flag to avoid previously visited URLs
                    self.avoid_repetitive_urls = patterns.get("repetitive_urls", [])

                # If we're repeatedly searching for the same queries, suggest alternatives
                if patterns.get("repetitive_queries", []):
                    # Generate alternative queries based on the original query
                    alt_queries = self._generate_alternative_queries(query)
                    if alt_queries:
                        alt_query = random.choice(alt_queries)
                        print(f"🔄 Suggesting alternative query: '{alt_query}' instead of '{query}'")
                        # Use the alternative query instead
                        query = alt_query
                        request_data["query"] = query
            else:
                # Reset behavior adjustment flags if repetitiveness is low
                self.encourage_clicking = False
                self.avoid_repetitive_urls = []

        # Check if we should process immediately or return cached results
        cached_results = self._check_for_cached_results(query)
        if cached_results:
            print(f"📋 Using cached results for query: '{query}'")
            return cached_results

        # For very high priority requests (8-10), process immediately
        if priority >= 8:
            print(f"🚀 High priority request ({priority}): Processing immediately")
            return self._process_search_request(request_data)

        # For medium-high priority requests (6-7), check if we can process now
        elif priority >= 6:
            # Check if we've processed too many requests recently
            recent_requests = sum(1 for h in self.processing_history 
                                if time.time() - h["timestamp"] < 60)

            if recent_requests < 3:  # If fewer than 3 requests in the last minute
                print(f"🔄 Medium-high priority request ({priority}): Processing now")
                return self._process_search_request(request_data)
            else:
                print(f"⏳ Medium-high priority request ({priority}): Too many recent requests, returning preliminary results")
                # Return preliminary results and process in background
                self._schedule_background_processing(request_data)
                return [{"title": f"Searching for '{query}'...", 
                         "snippet": "Your request is being processed. Results will be available soon.",
                         "url": "",
                         "status": "processing"}]

        # For medium-low priority requests (3-5), check queue size
        elif priority >= 3:
            # If queue is small, process now
            if len(self.request_queue) < 5:
                print(f"🔄 Medium priority request ({priority}): Queue is small, processing now")
                return self._process_search_request(request_data)
            else:
                print(f"⏳ Medium priority request ({priority}): Queue is busy, returning preliminary results")
                # Return preliminary results and process in background
                self._schedule_background_processing(request_data)
                return [{"title": f"Queued search for '{query}'", 
                         "snippet": f"Your request (priority {priority}) is queued. There are {len(self.request_queue)} requests in the queue.",
                         "url": "",
                         "status": "queued"}]

        # For low priority requests (1-2), almost always queue
        else:
            # Only process immediately if nothing else is happening
            if len(self.request_queue) <= 1 and not self.processing_history:
                print(f"🔄 Low priority request ({priority}): System is idle, processing now")
                return self._process_search_request(request_data)
            else:
                print(f"⏳ Low priority request ({priority}): Queued for later processing")
                # Return preliminary results and process in background
                self._schedule_background_processing(request_data)
                return [{"title": f"Queued search for '{query}'", 
                         "snippet": f"Your low priority request ({priority}) has been queued and will be processed when resources are available.",
                         "url": "",
                         "status": "queued"}]

    def _generate_alternative_queries(self, query: str) -> List[str]:
        """
        Generate alternative queries based on the original query

        Args:
            query: The original search query

        Returns:
            List of alternative queries
        """
        # Start with an empty list
        alternatives = []

        # Split the query into words
        words = query.split()

        if not words:
            return []

        # 1. Add more specific versions by adding relevant terms
        specificity_terms = ["tutorial", "guide", "examples", "how to", "explained", 
                            "best practices", "introduction", "advanced", "review"]

        for term in specificity_terms:
            if term not in query.lower():
                alternatives.append(f"{query} {term}")

        # 2. Add broader versions by removing specific terms
        if len(words) > 2:
            # Remove the last word
            alternatives.append(" ".join(words[:-1]))
            # Remove the first word
            alternatives.append(" ".join(words[1:]))

        # 3. Add related concepts if we can identify the main topic
        main_topic = words[0].lower()

        # Simple dictionary of related topics
        related_topics = {
            "python": ["programming", "coding", "software development", "scripting"],
            "javascript": ["web development", "frontend", "coding", "programming"],
            "machine": ["artificial intelligence", "deep learning", "neural networks"],
            "learning": ["education", "training", "courses", "tutorials"],
            "ai": ["artificial intelligence", "machine learning", "neural networks"],
            "web": ["website", "internet", "online", "browser"],
            "data": ["database", "analytics", "visualization", "statistics"],
            "science": ["research", "academic", "studies", "experiments"],
            "technology": ["tech", "innovation", "digital", "computing"],
            "business": ["company", "enterprise", "corporate", "management"],
            "health": ["medical", "wellness", "healthcare", "fitness"],
            "food": ["cooking", "recipes", "cuisine", "nutrition"],
            "travel": ["tourism", "vacation", "destinations", "trips"],
            "music": ["songs", "artists", "bands", "concerts"],
            "art": ["design", "creative", "visual", "artistic"],
            "history": ["historical", "past", "ancient", "events"],
            "news": ["current events", "headlines", "updates", "reports"]
        }

        # Check if we have related topics for the main topic
        for topic, related in related_topics.items():
            if topic in query.lower():
                for related_term in related:
                    # Replace the topic with the related term
                    new_query = query.lower().replace(topic, related_term)
                    if new_query != query.lower():
                        alternatives.append(new_query)
                break

        # 4. Add "how to" versions for practical queries
        if not any(w in query.lower() for w in ["how to", "tutorial", "guide"]):
            alternatives.append(f"how to {query}")

        # 5. Add "what is" versions for conceptual queries
        if not any(w in query.lower() for w in ["what is", "definition", "meaning"]):
            alternatives.append(f"what is {query}")

        # Remove duplicates and the original query
        alternatives = list(set(alternatives))
        if query in alternatives:
            alternatives.remove(query)

        # Limit to a reasonable number
        return alternatives[:5]

    def _check_for_cached_results(self, query: str) -> List[Dict[str, str]]:
        """
        Check if we have cached results for this query

        Args:
            query: The search query

        Returns:
            Cached results if available, otherwise None
        """
        if not hasattr(self, 'results_cache'):
            self.results_cache = {}
            return None

        # Check for exact match
        if query.lower() in self.results_cache:
            cache_entry = self.results_cache[query.lower()]
            # Check if cache is still fresh (less than 5 minutes old)
            if time.time() - cache_entry["timestamp"] < 300:
                return cache_entry["results"]

        # Check for similar queries
        query_keywords = set(word.lower() for word in query.split() if len(word) > 3)
        if query_keywords:
            for cached_query, cache_entry in self.results_cache.items():
                # Skip if cache is too old (more than 10 minutes)
                if time.time() - cache_entry["timestamp"] > 600:
                    continue

                cached_keywords = set(word.lower() for word in cached_query.split() if len(word) > 3)
                if cached_keywords:
                    # If there's significant overlap (>70% of keywords match)
                    overlap = len(query_keywords.intersection(cached_keywords))
                    smaller_set_size = min(len(query_keywords), len(cached_keywords))

                    if smaller_set_size > 0 and overlap / smaller_set_size > 0.7:
                        print(f"📋 Using similar cached results: '{cached_query}' for query: '{query}'")
                        return cache_entry["results"]

        return None

    def _schedule_background_processing(self, request_data: Dict[str, Any]) -> None:
        """
        Schedule a request for background processing

        Args:
            request_data: The request data
        """
        # Mark the request as scheduled
        request_data["status"] = "scheduled"

        # In a real implementation, this would start a background thread
        # For now, we'll just log that it's scheduled
        print(f"🔄 Scheduled background processing for query: '{request_data['query']}'")

        # In a real implementation, we would have a background thread that processes
        # the queue. For now, we'll just simulate it by processing the highest priority
        # request if the queue isn't too large
        if len(self.request_queue) <= 3:
            # Find the highest priority request that's scheduled
            scheduled_requests = [r for r in self.request_queue if r["status"] == "scheduled"]
            if scheduled_requests:
                # Sort by priority (higher first)
                scheduled_requests.sort(key=lambda x: -x["priority"])
                next_request = scheduled_requests[0]
                print(f"🔄 Processing highest priority scheduled request: '{next_request['query']}'")
                # Process the request (in a real implementation, this would be in a separate thread)
                try:
                    results = self._process_search_request(next_request)
                    # Cache the results
                    if not hasattr(self, 'results_cache'):
                        self.results_cache = {}
                    self.results_cache[next_request["query"].lower()] = {
                        "results": results,
                        "timestamp": time.time()
                    }
                    # Keep cache at a reasonable size
                    if len(self.results_cache) > 50:
                        # Remove oldest entries
                        oldest_queries = sorted(self.results_cache.keys(), 
                                              key=lambda q: self.results_cache[q]["timestamp"])
                        for old_query in oldest_queries[:len(self.results_cache) - 50]:
                            del self.results_cache[old_query]
                except Exception as e:
                    print(f"❌ Error processing scheduled request: {e}")

    def _add_to_queue(self, request_data: Dict[str, Any]) -> None:
        """
        Add a request to the queue and manage queue size with deduplication and batching

        Args:
            request_data: The request data to add
        """
        query = request_data["query"]
        priority = request_data["priority"]

        # Check for duplicate or similar requests
        duplicate_found = False
        for existing_request in self.request_queue:
            existing_query = existing_request["query"]

            # Check for exact duplicates
            if existing_query.lower() == query.lower():
                # Update priority if the new request has higher priority
                if priority > existing_request["priority"]:
                    existing_request["priority"] = priority
                    print(f"📊 Updated priority for duplicate request: '{query}' to {priority}")

                # Update timestamp to keep it fresh
                existing_request["timestamp"] = time.time()

                duplicate_found = True
                break

            # Check for similar requests (significant keyword overlap)
            query_keywords = set(word.lower() for word in query.split() if len(word) > 3)
            existing_keywords = set(word.lower() for word in existing_query.split() if len(word) > 3)

            # If there's significant overlap (>50% of keywords match)
            if query_keywords and existing_keywords:
                overlap = len(query_keywords.intersection(existing_keywords))
                smaller_set_size = min(len(query_keywords), len(existing_keywords))

                if smaller_set_size > 0 and overlap / smaller_set_size > 0.5:
                    # Combine the queries if they're similar but not identical
                    if priority >= existing_request["priority"]:
                        # Create a combined query that includes both
                        combined_query = f"{existing_query} {query}"
                        # Remove duplicate words
                        combined_words = []
                        for word in combined_query.split():
                            if word.lower() not in [w.lower() for w in combined_words]:
                                combined_words.append(word)

                        combined_query = " ".join(combined_words)

                        # Update the existing request
                        existing_request["query"] = combined_query
                        existing_request["priority"] = max(priority, existing_request["priority"])
                        existing_request["timestamp"] = time.time()

                        print(f"🔄 Combined similar requests into: '{combined_query}'")
                        duplicate_found = True
                        break

        # If no duplicate or similar request was found, add the new request
        if not duplicate_found:
            self.request_queue.append(request_data)
            print(f"➕ Added new request to queue: '{query}' with priority {priority}")

        # Sort queue by priority (higher first) and then by timestamp (older first)
        self.request_queue.sort(key=lambda x: (-x["priority"], x["timestamp"]))

        # Group related requests together for more efficient processing
        self._batch_related_requests()

        # If queue is too large, remove lowest priority items
        if len(self.request_queue) > self.max_queue_size:
            removed_count = len(self.request_queue) - self.max_queue_size
            print(f"⚠️ Queue size limit reached ({self.max_queue_size}). Removing {removed_count} lowest priority items.")
            # Keep only the top max_queue_size items
            self.request_queue = self.request_queue[:self.max_queue_size]

    def _batch_related_requests(self) -> None:
        """
        Group related requests together for more efficient processing
        """
        if len(self.request_queue) <= 1:
            return

        # Create a dictionary to group requests by their main topic
        topic_groups = {}

        # First pass: identify main topics and group requests
        for request in self.request_queue:
            query = request["query"]
            # Extract main topic (first few words or first keyword)
            words = query.split()
            main_topic = words[0].lower() if words else ""

            # For longer queries, try to find a more meaningful topic
            if len(words) > 1:
                # Look for longer words that might be more meaningful
                for word in words:
                    if len(word) > 5:  # Longer words are often more meaningful
                        main_topic = word.lower()
                        break

            # Add to appropriate group
            if main_topic not in topic_groups:
                topic_groups[main_topic] = []
            topic_groups[main_topic].append(request)

        # Second pass: reorder the queue to keep related requests together
        new_queue = []
        for topic, requests in topic_groups.items():
            # Sort each group by priority
            requests.sort(key=lambda x: (-x["priority"], x["timestamp"]))
            new_queue.extend(requests)

        # Update the queue
        self.request_queue = new_queue

    def _get_request_context(self, query: str) -> Dict[str, Any]:
        """
        Get context for a search request to ensure requests are contextually related

        Args:
            query: The search query

        Returns:
            Dict with context information
        """
        # Extract keywords from query
        keywords = [word.lower() for word in query.split() if len(word) > 3]

        # Find related previous queries
        related_queries = []
        for request in self.request_queue:
            prev_query = request["query"]
            prev_keywords = [word.lower() for word in prev_query.split() if len(word) > 3]

            # Check for keyword overlap
            if any(keyword in prev_keywords for keyword in keywords):
                related_queries.append(prev_query)

        return {
            "keywords": keywords,
            "related_queries": related_queries[:5]  # Keep only the 5 most recent related queries
        }

    def _process_search_request(self, request_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Process a search request

        Args:
            request_data: The request data

        Returns:
            List of search results
        """
        query = request_data["query"]
        max_results = request_data["max_results"]

        # Record this processing in the history
        if hasattr(self, 'processing_history'):
            # Add to processing history
            self.processing_history.append({
                "query": query,
                "priority": request_data.get("priority", 1),
                "timestamp": time.time(),
                "id": request_data.get("id", str(uuid.uuid4()))
            })

            # Keep history at a reasonable size
            if len(self.processing_history) > 100:
                self.processing_history = self.processing_history[-100:]

        # Update request status
        request_data["status"] = "processing"

        if self.driver is None:
            if not self.initialize_browser():
                return [{"error": "Failed to initialize browser."}]

        try:
            # Import computer control for human-like interactions
            from doba_extensions import ComputerControl
            computer_control = ComputerControl()

            # Implement priority-aware exponential backoff for search requests
            current_time = time.time()
            time_since_last_search = current_time - self.last_search_time
            priority = request_data.get("priority", 5)  # Default to medium priority if not specified

            # Calculate priority factor (higher priority = shorter wait time)
            # Priority range is 1-10, so priority_factor ranges from 0.1 to 1.0
            priority_factor = priority / 10.0

            # If searches are happening too quickly, add priority-aware exponential delay
            if time_since_last_search < 60:  # Less than a minute since last search
                self.consecutive_searches += 1

                # Base wait time with exponential backoff
                base_wait_time = min(300, 5 * (2 ** self.consecutive_searches))  # Cap at 5 minutes

                # Adjust wait time based on priority (higher priority = shorter wait)
                wait_time = base_wait_time * (1 - (priority_factor * 0.8))  # Reduce wait time by up to 80% for highest priority

                # Ensure minimum wait time of 1 second even for highest priority
                wait_time = max(1, wait_time)

                print(f"⚠️ Searches happening too quickly. Priority {priority}: Waiting {wait_time:.1f} seconds...")

                # For very high priority requests, check if we can skip the wait
                if priority >= 9 and self.consecutive_searches <= 2:
                    print(f"🚀 High priority request ({priority}): Skipping wait time")
                else:
                    time.sleep(wait_time)
            else:
                # Reset consecutive searches counter if enough time has passed
                self.consecutive_searches = 0

            # Record the time of this search
            self.last_search_time = time.time()

            # For debugging and learning
            if hasattr(self, 'request_timing_history'):
                self.request_timing_history.append({
                    "query": query,
                    "priority": priority,
                    "consecutive_searches": self.consecutive_searches,
                    "time_since_last_search": time_since_last_search,
                    "wait_time": wait_time if time_since_last_search < 60 else 0,
                    "timestamp": current_time
                })
                # Keep history at a reasonable size
                if len(self.request_timing_history) > 100:
                    self.request_timing_history = self.request_timing_history[-100:]
            else:
                self.request_timing_history = []

            # Navigate to search engine
            self.driver.get(self.search_engine)
            self.current_url = self.search_engine

            # Add a human-like delay after page load (simulate reading the page)
            time.sleep(random.uniform(2.5, 6.0))  # Increased delay

            # Wait for the search box to be available
            search_box = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.NAME, "q"))
            )

            # Get the position of the search box for mouse movement
            search_box_location = search_box.location
            search_box_size = search_box.size
            search_box_x = search_box_location['x'] + search_box_size['width'] // 2
            search_box_y = search_box_location['y'] + search_box_size['height'] // 2

            # Move mouse to search box with human-like movement
            computer_control.move_mouse(search_box_x, search_box_y, human_like=True)

            # Add a small delay before clicking (simulating human hesitation)
            time.sleep(random.uniform(0.3, 0.7))

            # Click on the search box
            computer_control.click(search_box_x, search_box_y)

            # Clear the search box if needed
            search_box.clear()

            # Add a small delay before typing (simulating human thinking)
            time.sleep(random.uniform(1.0, 2.5))  # Increased delay

            # Type the query with human-like typing (directly in Selenium)
            # We can't use computer_control.type_text here because it would type outside the browser
            # Instead, we'll simulate human typing within Selenium
            for char in query:
                # Type each character with variable delay
                search_box.send_keys(char)
                # Random delay between keystrokes (increased)
                time.sleep(random.uniform(0.1, 0.35))

                # Occasionally add a longer pause (as if thinking)
                if random.random() < 0.05:  # 5% chance (increased from 2%)
                    time.sleep(random.uniform(0.8, 1.5))  # Increased delay

                # Occasionally make a typo and correct it
                if random.random() < 0.05:  # 5% chance (increased from 3%)
                    # Backspace to delete the character
                    search_box.send_keys(Keys.BACKSPACE)
                    time.sleep(random.uniform(0.2, 0.5))  # Increased delay
                    # Type the correct character
                    search_box.send_keys(char)
                    time.sleep(random.uniform(0.2, 0.4))  # Increased delay

            # Add a pause before pressing Enter (simulating human review of query)
            time.sleep(random.uniform(1.5, 3.0))  # Increased delay

            # Press Enter to submit the search
            search_box.send_keys(Keys.RETURN)

            # Wait for search results with a variable timeout
            try:
                WebDriverWait(self.driver, random.uniform(10, 15)).until(  # Increased timeout
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div.g"))
                )
            except TimeoutException:
                # Try an alternative selector if the first one fails
                try:
                    WebDriverWait(self.driver, 8).until(  # Increased timeout
                        EC.presence_of_element_located((By.CSS_SELECTOR, "div[data-hveid]"))
                    )
                except TimeoutException:
                    # Try one more generic selector
                    WebDriverWait(self.driver, 5).until(
                        EC.presence_of_element_located((By.TAG_NAME, "a"))
                    )

            # Enhanced CAPTCHA detection with context analysis and verification
            # Define different types of indicators with weights
            captcha_indicators = {
                # Strong indicators (these are very likely to indicate a CAPTCHA)
                "strong": [
                    "please solve this captcha",
                    "complete the security check",
                    "prove you're not a robot",
                    "i'm not a robot checkbox",
                    "captcha challenge",
                    "security verification required"
                ],
                # Medium indicators (these might indicate a CAPTCHA in certain contexts)
                "medium": [
                    "unusual traffic from your computer",
                    "verify you're a human",
                    "automated requests",
                    "suspicious activity",
                    "security check"
                ],
                # Weak indicators (these alone are not enough to indicate a CAPTCHA)
                "weak": [
                    "robot",
                    "automated",
                    "captcha",
                    "verification",
                    "security"
                ]
            }

            page_source = self.driver.page_source.lower()

            # Calculate a CAPTCHA probability score
            captcha_score = 0
            captcha_matches = []

            # Check for strong indicators (any one is highly suspicious)
            for indicator in captcha_indicators["strong"]:
                if indicator in page_source:
                    captcha_score += 10
                    captcha_matches.append(f"Strong: {indicator}")

            # Check for medium indicators (need more context)
            for indicator in captcha_indicators["medium"]:
                if indicator in page_source:
                    # Look for nearby context that confirms it's a CAPTCHA
                    # For example, if "unusual traffic" is near "security check"
                    context_window = 200  # Characters to check before and after
                    indicator_index = page_source.find(indicator)
                    if indicator_index >= 0:
                        start = max(0, indicator_index - context_window)
                        end = min(len(page_source), indicator_index + len(indicator) + context_window)
                        context = page_source[start:end]

                        # Check if other indicators are in the same context window
                        context_score = 0
                        for other_indicator in captcha_indicators["medium"] + captcha_indicators["strong"]:
                            if other_indicator != indicator and other_indicator in context:
                                context_score += 5
                                captcha_matches.append(f"Context: {indicator} near {other_indicator}")

                        captcha_score += 3 + context_score
                    else:
                        captcha_score += 3
                        captcha_matches.append(f"Medium: {indicator}")

            # Check for weak indicators (only matter if there are multiple)
            weak_count = 0
            for indicator in captcha_indicators["weak"]:
                if indicator in page_source:
                    weak_count += 1
                    captcha_matches.append(f"Weak: {indicator}")

            # Add score based on number of weak indicators
            if weak_count >= 3:
                captcha_score += 5
            elif weak_count >= 2:
                captcha_score += 2
            elif weak_count == 1:
                captcha_score += 1

            # Check for CAPTCHA-specific HTML elements
            captcha_elements = [
                "iframe[src*='captcha']",
                "iframe[src*='recaptcha']",
                "div.g-recaptcha",
                "div[class*='captcha']",
                "input[name*='captcha']"
            ]

            for element_selector in captcha_elements:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, element_selector)
                    if elements:
                        captcha_score += 15  # Very strong indicator
                        captcha_matches.append(f"Element: {element_selector}")
                except:
                    pass

            # Determine if this is a CAPTCHA based on the score
            is_captcha = captcha_score >= 10  # Threshold for CAPTCHA detection

            if is_captcha:
                print(f"⚠️ CAPTCHA or unusual traffic detection encountered (score: {captcha_score})")
                print(f"⚠️ Matches: {', '.join(captcha_matches)}")

                # Log the detection for analysis
                with open("captcha_detection_log.txt", "a") as log_file:
                    log_file.write(f"{datetime.now()} - CAPTCHA detected for query: {query}\n")
                    log_file.write(f"Score: {captcha_score}, Matches: {captcha_matches}\n")
                    log_file.write(f"Search engine: {self.search_engine}\n")
                    log_file.write("-" * 50 + "\n")

                # If using DuckDuckGo and CAPTCHA is detected, try Bing instead
                if "duckduckgo" in self.search_engine.lower():
                    print("⚠️ Switching to Bing search engine")
                    self.search_engine = "https://www.bing.com"
                    # Wait a significant amount of time before trying again
                    time.sleep(random.uniform(60, 120))  # 1-2 minute delay
                    return self._process_search_request(request_data)

                # If already using alternative search engine, wait longer and try again
                time.sleep(random.uniform(300, 600))  # 5-10 minute delay
                return [{"error": "CAPTCHA detected. Please try again later."}]

            # Add a delay to simulate reading the search results (increased)
            time.sleep(random.uniform(3.0, 7.0))

            # Scroll down slowly to simulate reading through results (more scrolls and longer pauses)
            for _ in range(random.randint(4, 8)):  # Increased range
                # Scroll down a random amount
                scroll_amount = random.randint(80, 250)  # More variable scroll amounts
                self.driver.execute_script(f"window.scrollBy(0, {scroll_amount});")
                # Pause between scrolls (increased)
                time.sleep(random.uniform(1.2, 3.0))

                # Occasionally pause longer as if reading something interesting
                if random.random() < 0.2:  # 20% chance
                    time.sleep(random.uniform(3.0, 6.0))

                # Occasionally scroll back up a bit as if reconsidering
                if random.random() < 0.15:  # 15% chance
                    up_amount = random.randint(40, 120)
                    self.driver.execute_script(f"window.scrollBy(0, -{up_amount});")
                    time.sleep(random.uniform(0.8, 2.0))

            # Extract search results
            results = []

            # Try different selectors for search results to improve robustness
            # Add more selectors for different search engines
            search_results = []

            # Selectors for different search engines
            if "duckduckgo" in self.search_engine.lower():
                selectors = [".result", ".result__body", "article", ".result__a"]
            elif "bing" in self.search_engine.lower():
                selectors = [".b_algo", ".b_title", ".b_caption"]
            else:  # Google or other
                selectors = ["div.g", "div[data-hveid]", "div.tF2Cxc", "div.yuRUbf", "h3.LC20lb"]

            # Try each selector
            for selector in selectors:
                try:
                    search_results = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if search_results and len(search_results) > 0:
                        print(f"✅ Found {len(search_results)} results with selector: {selector}")
                        break
                except Exception as selector_error:
                    print(f"⚠️ Selector error: {selector_error}")
                    continue

            # If we still don't have results, try a more general approach
            if not search_results:
                print("⚠️ No results found with specific selectors, trying general approach")
                # Get all links that might be search results
                search_results = self.driver.find_elements(By.TAG_NAME, "a")
                # Filter to only those that have text and href
                search_results = [r for r in search_results if r.text and r.get_attribute("href")]

                # Further filter to exclude navigation links
                search_results = [r for r in search_results if len(r.text) > 15 and not r.text.startswith("http")]

            # Add a delay before processing results
            time.sleep(random.uniform(1.0, 2.5))

            for i, result in enumerate(search_results):
                if i >= max_results:
                    break

                try:
                    # More frequently move mouse over the result to simulate interest
                    if random.random() < 0.5:  # 50% chance (increased from 30%)
                        result_location = result.location
                        result_size = result.size
                        result_x = result_location['x'] + result_size['width'] // 2
                        result_y = result_location['y'] + result_size['height'] // 2

                        # Move mouse to the result with more human-like movement
                        computer_control.move_mouse(result_x, result_y, human_like=True)

                        # Hover for a random time (increased)
                        time.sleep(random.uniform(0.5, 2.0))

                    # Try to extract title, URL, and snippet with different approaches
                    # Add a small delay before extracting data to simulate human reading
                    time.sleep(random.uniform(0.3, 0.8))

                    # Extract title based on search engine
                    try:
                        if "duckduckgo" in self.search_engine.lower():
                            title_selectors = [".result__a", ".result__title"]
                        elif "bing" in self.search_engine.lower():
                            title_selectors = ["h2", ".b_title"]
                        else:  # Google or other
                            title_selectors = ["h3", "h3.LC20lb", ".DKV0Md"]

                        title = None
                        for selector in title_selectors:
                            try:
                                title_element = result.find_element(By.CSS_SELECTOR, selector)
                                title = title_element.text
                                if title:
                                    break
                            except:
                                continue

                        if not title:
                            # Fallback method
                            title = result.text.split('\n')[0]
                    except:
                        title = "Unknown Title"

                    # Extract URL
                    try:
                        # First try to get href directly if result is an 'a' tag
                        url = result.get_attribute("href")

                        # If not successful, look for an 'a' tag inside
                        if not url:
                            link_element = result.find_element(By.CSS_SELECTOR, "a")
                            url = link_element.get_attribute("href")

                        # Clean tracking parameters from URL if present
                        if url and ("google" in url or "bing" in url or "duckduckgo" in url) and "http" in url:
                            # Extract actual URL from redirect URL
                            try:
                                url_parts = url.split("?")
                                if len(url_parts) > 1:
                                    params = url_parts[1].split("&")
                                    for param in params:
                                        if param.startswith("url=") or param.startswith("q=") or param.startswith("u="):
                                            actual_url = param.split("=", 1)[1]
                                            if "http" in actual_url:
                                                url = actual_url
                                                break
                            except:
                                # Keep original URL if cleaning fails
                                pass
                    except:
                        url = "Unknown URL"

                    # Extract snippet based on search engine
                    try:
                        if "duckduckgo" in self.search_engine.lower():
                            snippet_selectors = [".result__snippet"]
                        elif "bing" in self.search_engine.lower():
                            snippet_selectors = [".b_caption p", ".b_snippet"]
                        else:  # Google or other
                            snippet_selectors = ["div.VwiC3b", ".IsZvec", "span.st"]

                        snippet = None
                        for selector in snippet_selectors:
                            try:
                                snippet_element = result.find_element(By.CSS_SELECTOR, selector)
                                snippet = snippet_element.text
                                if snippet:
                                    break
                            except:
                                continue

                        if not snippet:
                            # Fallback method
                            text_parts = result.text.split('\n')
                            snippet = '\n'.join(text_parts[1:]) if len(text_parts) > 1 else "No snippet available"
                    except:
                        snippet = "No snippet available"

                    # Click on a result with higher probability and improved navigation
                    # Higher probability for clicking based on result relevance
                    click_probability = 0.4  # Base 40% chance, much higher than before

                    # Check if we should encourage clicking due to repetitive search-only behavior
                    if hasattr(self, 'encourage_clicking') and self.encourage_clicking:
                        click_probability += 0.4  # Significantly increase click probability
                        print(f"🖱️ Encouraging clicking: Increased probability by 40%")

                    # Increase probability if the title or snippet contains query terms
                    if any(term.lower() in title.lower() for term in query.lower().split()):
                        click_probability += 0.2  # +20% if title contains query terms

                    if any(term.lower() in snippet.lower() for term in query.lower().split()):
                        click_probability += 0.1  # +10% if snippet contains query terms

                    # Track this URL to detect repetitive actions
                    current_url_key = f"search_result_{url}"
                    if not hasattr(self, 'visited_urls'):
                        self.visited_urls = {}

                    # Check if we've visited this URL before
                    if current_url_key in self.visited_urls:
                        visit_count = self.visited_urls[current_url_key]
                        if visit_count > 2:
                            # If we've visited this URL multiple times, reduce probability
                            click_probability -= 0.3
                            print(f"⚠️ Detected repetitive action: visited {url} {visit_count} times before")
                        else:
                            # Slightly increase probability for URLs we've seen but not visited much
                            click_probability += 0.1

                    # Check if we should avoid this URL due to repetitive behavior
                    if hasattr(self, 'avoid_repetitive_urls') and self.avoid_repetitive_urls:
                        if url.lower() in [u.lower() for u in self.avoid_repetitive_urls]:
                            # Significantly reduce probability for URLs we're trying to avoid
                            click_probability -= 0.5
                            print(f"🔄 Avoiding repetitive URL: {url}")

                    # Ensure probability is within valid range (0.0 to 1.0)
                    click_probability = max(0.0, min(1.0, click_probability))

                    if random.random() < click_probability:
                        try:
                            print(f"🖱️ Clicking on search result: {title}")

                            # Record this visit
                            self.visited_urls[current_url_key] = self.visited_urls.get(current_url_key, 0) + 1

                            # Set the clicked_result flag to true
                            self.clicked_result = True

                            # Save current window handle
                            main_window = self.driver.current_window_handle

                            # Find clickable element
                            clickable = None
                            try:
                                clickable = result.find_element(By.CSS_SELECTOR, "a")
                            except:
                                try:
                                    clickable = result.find_element(By.CSS_SELECTOR, "h3")
                                except:
                                    clickable = result

                            # Scroll element into view if needed - using 'auto' instead of 'smooth' for more reliable positioning
                            self.driver.execute_script("arguments[0].scrollIntoView({behavior: 'auto', block: 'center'});", clickable)
                            time.sleep(random.uniform(0.8, 1.2))

                            # Move mouse to the element with human-like movement
                            result_location = clickable.location
                            result_size = clickable.size
                            result_x = result_location['x'] + result_size['width'] // 2
                            result_y = result_location['y'] + result_size['height'] // 2
                            computer_control.move_mouse(result_x, result_y, human_like=True)
                            # Increased delay after moving mouse to ensure element is stable before clicking
                            time.sleep(random.uniform(0.7, 1.0))

                            # Decide whether to open in new tab or current window
                            open_in_new_tab = random.random() < 0.3  # 30% chance to open in new tab

                            if open_in_new_tab:
                                # Click with JavaScript to open in new tab
                                self.driver.execute_script("arguments[0].setAttribute('target', '_blank'); arguments[0].click();", clickable)

                                # Switch to new tab
                                self.driver.switch_to.window(self.driver.window_handles[-1])
                            else:
                                # Choose one click method rather than potentially using both
                                click_method = random.choice(["direct", "selenium"])

                                if click_method == "direct":
                                    # Click directly to navigate in current window
                                    print("🖱️ Using direct mouse click")
                                    # Verify element is still visible and clickable before clicking
                                    try:
                                        # Check if element is still visible
                                        if clickable.is_displayed():
                                            # Try to check if element is clickable using JavaScript
                                            try:
                                                # Check if element is not covered by another element
                                                click_info = self.driver.execute_script("""
                                                    var elem = arguments[0];
                                                    var rect = elem.getBoundingClientRect();
                                                    var cx = rect.left + rect.width / 2;
                                                    var cy = rect.top + rect.height / 2;
                                                    var elemAtPoint = document.elementFromPoint(cx, cy);
                                                    var isClickable = (elemAtPoint === elem || elem.contains(elemAtPoint));

                                                    // Get information about the element at point
                                                    var interceptInfo = '';
                                                    if (!isClickable && elemAtPoint) {
                                                        interceptInfo = {
                                                            'tagName': elemAtPoint.tagName,
                                                            'id': elemAtPoint.id,
                                                            'className': elemAtPoint.className,
                                                            'text': elemAtPoint.textContent.substring(0, 50)
                                                        };
                                                    }

                                                    return {
                                                        'isClickable': isClickable,
                                                        'interceptInfo': interceptInfo
                                                    };
                                                """, clickable)

                                                is_clickable = click_info.get('isClickable', False)

                                                if is_clickable:
                                                    # Element is clickable, proceed with direct click
                                                    print(f"✅ Element is clickable at coordinates ({result_x}, {result_y})")
                                                    computer_control.click(result_x, result_y)
                                                else:
                                                    # Element is covered by another element, use Selenium click instead
                                                    intercept_info = click_info.get('interceptInfo', '')
                                                    if intercept_info:
                                                        print(f"⚠️ Element is covered by another element at coordinates ({result_x}, {result_y})")
                                                        print(f"   Intercepting element: {intercept_info.get('tagName', 'unknown')} "
                                                              f"id='{intercept_info.get('id', '')}' "
                                                              f"class='{intercept_info.get('className', '')}' "
                                                              f"text='{intercept_info.get('text', '')}'")
                                                    else:
                                                        print(f"⚠️ Element is covered by another element at coordinates ({result_x}, {result_y})")

                                                    print("   Using Selenium click instead")
                                                    clickable.click()
                                            except:
                                                # Error checking if element is clickable, use Selenium click instead
                                                print("⚠️ Error checking if element is clickable, using Selenium click instead")
                                                clickable.click()
                                        else:
                                            print("⚠️ Element no longer visible, using Selenium click instead")
                                            try:
                                                # Try to relocate the element first
                                                result_text = clickable.text
                                                if result_text:
                                                    print("   Attempting to relocate element by text")
                                                    elements = self.driver.find_elements(By.XPATH, f"//*[contains(text(), '{result_text}')]")
                                                    if elements:
                                                        print("   Element relocated by text, clicking")
                                                        elements[0].click()
                                                        return

                                                # If we couldn't relocate by text, try JavaScript click
                                                self.driver.execute_script("arguments[0].click();", clickable)
                                            except Exception as e:
                                                print(f"   JavaScript click failed: {str(e)}")
                                                # Last resort: regular click
                                                clickable.click()
                                    except Exception as click_error:
                                        print(f"⚠️ Error with direct click: {str(click_error)}, falling back to Selenium click")
                                        try:
                                            # Try scrolling into view first
                                            self.driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", clickable)
                                            time.sleep(0.5)
                                            clickable.click()
                                        except Exception as selenium_click_error:
                                            print(f"⚠️ Selenium click also failed: {str(selenium_click_error)}")
                                            # Last resort: try JavaScript click
                                            self.driver.execute_script("arguments[0].click();", clickable)
                                else:
                                    # Use Selenium's built-in click method
                                    print("🖱️ Using Selenium click")
                                    clickable.click()

                            # Wait for page to load
                            try:
                                WebDriverWait(self.driver, 10).until(
                                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                                )
                            except:
                                print("⚠️ Timeout waiting for page to load")

                            # Read the page for a variable amount of time
                            reading_time = random.uniform(5.0, 15.0)
                            print(f"📖 Reading page for {reading_time:.1f} seconds")

                            # Scroll through the page while reading
                            scroll_count = random.randint(2, 5)
                            for _ in range(scroll_count):
                                # Scroll down a random amount
                                scroll_amount = random.randint(100, 500)
                                self.driver.execute_script(f"window.scrollBy(0, {scroll_amount});")
                                # Pause between scrolls
                                time.sleep(reading_time / scroll_count)

                            # Record the page content and URL for learning
                            try:
                                page_title = self.driver.title
                                page_url = self.driver.current_url
                                page_content = self.driver.find_element(By.TAG_NAME, "body").text[:1000]  # First 1000 chars

                                # Store this information for learning
                                if not hasattr(self, 'browsing_history'):
                                    self.browsing_history = []

                                self.browsing_history.append({
                                    "title": page_title,
                                    "url": page_url,
                                    "content_preview": page_content,
                                    "timestamp": time.time(),
                                    "from_search": query
                                })

                                # Keep history at a reasonable size
                                if len(self.browsing_history) > 50:
                                    self.browsing_history = self.browsing_history[-50:]
                            except:
                                print("⚠️ Error recording page information")

                            if open_in_new_tab:
                                # Close tab and switch back
                                self.driver.close()
                                self.driver.switch_to.window(main_window)
                            else:
                                # Navigate back to search results
                                self.driver.back()
                                # Wait for search results to load again
                                try:
                                    WebDriverWait(self.driver, 10).until(
                                        EC.presence_of_element_located((By.TAG_NAME, "a"))
                                    )
                                except:
                                    print("⚠️ Timeout waiting for search results to reload")
                        except Exception as click_error:
                            print(f"⚠️ Error clicking on result: {click_error}")

                            # Try alternative clicking methods before giving up
                            try:
                                print("🔄 Attempting alternative clicking methods...")

                                # Store the URL we're trying to visit
                                target_url = url

                                # Try method 1: JavaScript click
                                try:
                                    print("🔄 Trying JavaScript click...")
                                    self.driver.execute_script("arguments[0].click();", clickable)
                                    time.sleep(1)

                                    # Check if page changed
                                    if self.driver.current_url != self.current_url:
                                        print(f"✅ JavaScript click successful! Navigated to: {self.driver.current_url}")
                                        self.clicked_result = True
                                        return
                                except Exception as js_error:
                                    print(f"⚠️ JavaScript click failed: {js_error}")

                                # Try method 2: Open in new tab with JavaScript
                                try:
                                    print("🔄 Trying to open in new tab...")
                                    self.driver.execute_script("window.open(arguments[0], '_blank');", target_url)
                                    time.sleep(1)

                                    # Switch to the new tab
                                    self.driver.switch_to.window(self.driver.window_handles[-1])
                                    time.sleep(1)

                                    # Check if we successfully navigated
                                    if self.driver.current_url != self.current_url and "http" in self.driver.current_url:
                                        print(f"✅ New tab navigation successful! Navigated to: {self.driver.current_url}")
                                        self.clicked_result = True
                                        return
                                    else:
                                        # Close the tab if navigation failed
                                        self.driver.close()
                                        self.driver.switch_to.window(main_window)
                                except Exception as tab_error:
                                    print(f"⚠️ New tab navigation failed: {tab_error}")
                                    # Make sure we're back on the main window
                                    try:
                                        if len(self.driver.window_handles) > 1:
                                            self.driver.close()
                                            self.driver.switch_to.window(main_window)
                                    except:
                                        pass

                                # Try method 3: Direct navigation as last resort
                                try:
                                    print(f"🔄 Trying direct navigation to: {target_url}")
                                    self.driver.get(target_url)
                                    time.sleep(1)

                                    # Check if we successfully navigated
                                    if self.driver.current_url != self.current_url:
                                        print(f"✅ Direct navigation successful! Navigated to: {self.driver.current_url}")
                                        self.clicked_result = True
                                        return
                                except Exception as nav_error:
                                    print(f"⚠️ Direct navigation failed: {nav_error}")

                            except Exception as recovery_error:
                                print(f"⚠️ All recovery attempts failed: {recovery_error}")

                            # If all methods failed, reset clicked_result flag
                            self.clicked_result = False

                            # Make sure we're back on the main window if we were using tabs
                            try:
                                if self.driver.current_window_handle != main_window:
                                    self.driver.close()
                                    self.driver.switch_to.window(main_window)
                            except:
                                # If we can't recover, try to navigate back to the search engine
                                try:
                                    self.driver.get(self.search_engine)
                                except:
                                    pass

                    results.append({
                        "title": title,
                        "snippet": snippet,
                        "url": url
                    })

                    # Add a delay between processing results
                    time.sleep(random.uniform(0.5, 1.5))

                except Exception as result_error:
                    print(f"⚠️ Error processing result {i}: {result_error}")
                    continue

            # Only scroll back up if we haven't clicked on any results
            # This prevents losing our place in the search results
            if not hasattr(self, 'clicked_result') or not self.clicked_result:
                # Scroll back up partially, not all the way to the top
                # This simulates a human who might scroll back up but not completely
                scroll_position = random.uniform(0.2, 0.4)  # Scroll back to 20-40% of the page
                self.driver.execute_script(f"window.scrollTo(0, document.body.scrollHeight * {scroll_position});")
                print(f"📜 Scrolled back to {int(scroll_position * 100)}% of the page")
            else:
                print("📌 Maintaining scroll position since we clicked on results")
                # Reset the clicked_result flag for the next search
                self.clicked_result = False

            # Add a final delay before returning results (increased)
            time.sleep(random.uniform(1.5, 3.0))

            # Update the request queue - remove this request
            self.request_queue = [req for req in self.request_queue if req["id"] != request_data["id"]]

            # Log successful search
            if results is None:
                print(f"⚠️ No results found for search request: '{query}'")
                return [{"error": "No results found"}]
            else:
                print(f"✅ Successfully processed search request: '{query}' - Found {len(results)} results")

            return results
        except Exception as e:
            print(f"❌ Search failed: {e}")
            # Update the request queue - remove this request if it exists
            try:
                self.request_queue = [req for req in self.request_queue if req["id"] != request_data["id"]]
            except:
                pass
            return [{"error": f"Search failed: {str(e)}"}]

    def get_queue_status(self) -> Dict[str, Any]:
        """
        Get the current status of the request queue

        Returns:
            Dict with queue status information
        """
        return {
            "queue_size": len(self.request_queue),
            "max_queue_size": self.max_queue_size,
            "queue_items": [
                {
                    "id": req["id"],
                    "query": req["query"],
                    "priority": req["priority"],
                    "timestamp": req["timestamp"],
                    "age_seconds": time.time() - req["timestamp"],
                    "status": req.get("status", "pending")
                }
                for req in self.request_queue[:10]  # Show only the top 10 items
            ],
            "repetitive_patterns": self.analyze_browsing_patterns() if hasattr(self, 'browsing_history') else {}
        }

    def analyze_browsing_patterns(self) -> Dict[str, Any]:
        """
        Analyze browsing history to detect repetitive patterns

        Returns:
            Dict with analysis results
        """
        if not hasattr(self, 'browsing_history') or len(self.browsing_history) < 3:
            return {"status": "insufficient_data", "message": "Not enough browsing history to analyze patterns"}

        # Count query frequencies
        query_counts = {}
        for entry in self.browsing_history:
            query = entry.get("from_search", "").lower()
            if query:
                query_counts[query] = query_counts.get(query, 0) + 1

        # Find repeated URLs
        url_counts = {}
        for entry in self.browsing_history:
            url = entry.get("url", "").lower()
            if url and "http" in url:  # Only count actual URLs
                url_counts[url] = url_counts.get(url, 0) + 1

        # Identify repetitive patterns
        repetitive_queries = [q for q, count in query_counts.items() if count >= 3]
        repetitive_urls = [u for u, count in url_counts.items() if count >= 2]

        # Check for cycling behavior (repeatedly visiting the same set of pages)
        cycling_detected = False
        if len(self.browsing_history) >= 6:
            # Look at the last 6 entries
            recent_urls = [entry.get("url", "") for entry in self.browsing_history[-6:]]
            # Check if there are duplicates in the sequence
            if len(set(recent_urls)) < len(recent_urls) * 0.7:  # More than 30% duplicates
                cycling_detected = True

        # Check for search-only behavior (repeatedly searching without clicking)
        search_only_behavior = False
        if len(self.browsing_history) >= 5:
            # Count how many of the last 5 entries are from search pages
            search_page_count = sum(1 for entry in self.browsing_history[-5:] 
                                  if any(se in entry.get("url", "").lower() 
                                         for se in ["google", "bing", "duckduckgo", "search"]))
            if search_page_count >= 4:  # 80% or more are search pages
                search_only_behavior = True

        # Generate recommendations based on patterns
        recommendations = []
        if repetitive_queries:
            recommendations.append(f"Detected repeated searches for: {', '.join(repetitive_queries[:3])}")
            recommendations.append("Consider exploring more diverse topics or refining your search terms")

        if repetitive_urls:
            recommendations.append(f"Repeatedly visiting the same pages")
            recommendations.append("Try clicking on different search results or exploring related links")

        if cycling_detected:
            recommendations.append("Detected cycling behavior (repeatedly visiting the same set of pages)")
            recommendations.append("Try exploring new topics or using different search terms")

        if search_only_behavior:
            recommendations.append("Mostly staying on search pages without clicking on results")
            recommendations.append("Try clicking on search results to explore content in more depth")

        # Calculate an overall repetitiveness score
        repetitiveness_score = 0
        if repetitive_queries:
            repetitiveness_score += min(len(repetitive_queries) * 10, 30)
        if repetitive_urls:
            repetitiveness_score += min(len(repetitive_urls) * 15, 30)
        if cycling_detected:
            repetitiveness_score += 20
        if search_only_behavior:
            repetitiveness_score += 20

        # Cap the score at 100
        repetitiveness_score = min(repetitiveness_score, 100)

        return {
            "status": "analysis_complete",
            "repetitiveness_score": repetitiveness_score,
            "repetitive_queries": repetitive_queries[:5],  # Top 5 only
            "repetitive_urls": repetitive_urls[:5],  # Top 5 only
            "cycling_detected": cycling_detected,
            "search_only_behavior": search_only_behavior,
            "recommendations": recommendations,
            "browsing_history_size": len(self.browsing_history)
        }

    def clear_queue(self) -> Dict[str, Any]:
        """
        Clear the request queue

        Returns:
            Dict with operation status
        """
        queue_size = len(self.request_queue)
        self.request_queue = []
        return {
            "success": True,
            "message": f"Cleared {queue_size} items from the queue",
            "cleared_items": queue_size
        }

    def prioritize_request(self, request_id: str, new_priority: int) -> Dict[str, Any]:
        """
        Change the priority of a request in the queue

        Args:
            request_id: ID of the request to prioritize
            new_priority: New priority value (1-10, higher is more important)

        Returns:
            Dict with operation status
        """
        # Validate priority
        new_priority = max(1, min(10, new_priority))

        # Find the request
        for req in self.request_queue:
            if req["id"] == request_id:
                old_priority = req["priority"]
                req["priority"] = new_priority

                # Re-sort the queue
                self.request_queue.sort(key=lambda x: (-x["priority"], x["timestamp"]))

                return {
                    "success": True,
                    "message": f"Changed priority from {old_priority} to {new_priority}",
                    "request_id": request_id
                }

        return {
            "success": False,
            "message": f"Request with ID {request_id} not found in queue",
            "request_id": request_id
        }

    def set_max_queue_size(self, size: int) -> Dict[str, Any]:
        """
        Set the maximum queue size

        Args:
            size: New maximum queue size

        Returns:
            Dict with operation status
        """
        # Validate size
        old_size = self.max_queue_size
        self.max_queue_size = max(10, min(500, size))

        # If queue is now too large, remove lowest priority items
        if len(self.request_queue) > self.max_queue_size:
            removed_count = len(self.request_queue) - self.max_queue_size
            self.request_queue = self.request_queue[:self.max_queue_size]
            message = f"Set max queue size from {old_size} to {self.max_queue_size} and removed {removed_count} items"
        else:
            message = f"Set max queue size from {old_size} to {self.max_queue_size}"

        return {
            "success": True,
            "message": message,
            "old_size": old_size,
            "new_size": self.max_queue_size
        }

    def find_and_click_link(self, target_url: str, max_attempts: int = 3) -> bool:
        """
        Find a link with the given URL on the current page and click it.

        This method tries to find a link that matches the target URL and click it directly,
        which is more human-like than directly navigating to the URL.

        Args:
            target_url: The URL to find and click
            max_attempts: Maximum number of attempts to find and click the link

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.available or self.driver is None:
            print("⚠️ Browser not available or not initialized.")
            return False

        # Normalize the target URL for comparison
        # Remove trailing slashes and protocol for more flexible matching
        normalized_target = target_url.rstrip('/')
        if normalized_target.startswith('https://'):
            normalized_target_http = 'http://' + normalized_target[8:]
        elif normalized_target.startswith('http://'):
            normalized_target_https = 'https://' + normalized_target[7:]

        # Extract domain from target URL for partial matching
        target_domain = ""
        if "://" in normalized_target:
            target_domain = normalized_target.split("://")[1].split("/")[0]
            # Remove www. prefix if present
            if target_domain.startswith("www."):
                target_domain = target_domain[4:]

        print(f"🔍 Looking for link to: {target_url} (domain: {target_domain})")

        # Track attempts
        for attempt in range(max_attempts):
            try:
                # Find all links on the page
                links = self.driver.find_elements(By.TAG_NAME, "a")
                print(f"  Found {len(links)} links on the page")

                # First try: Look for exact URL match
                for link in links:
                    try:
                        href = link.get_attribute("href")
                        if not href:
                            continue

                        # Normalize the href for comparison
                        normalized_href = href.rstrip('/')

                        # Check for exact match or protocol-independent match
                        if (normalized_href == normalized_target or 
                            (normalized_target.startswith('https://') and normalized_href == normalized_target_http) or
                            (normalized_target.startswith('http://') and normalized_href == normalized_target_https)):

                            print(f"✅ Found exact matching link: {href}")

                            # Scroll the link into view
                            self.driver.execute_script("arguments[0].scrollIntoView({behavior: 'auto', block: 'center'});", link)
                            time.sleep(0.5)

                            # Click the link
                            print(f"🖱️ Clicking on link to: {href}")
                            link.click()

                            # Wait for the page to load
                            WebDriverWait(self.driver, 10).until(
                                EC.presence_of_element_located((By.TAG_NAME, "body"))
                            )

                            # Verify we navigated to the expected URL
                            current_url = self.driver.current_url
                            if current_url.rstrip('/') == normalized_target or current_url.startswith(normalized_target):
                                print(f"✅ Successfully navigated to: {current_url}")
                                self.current_url = current_url
                                return True
                            else:
                                print(f"⚠️ Clicked link but ended up at unexpected URL: {current_url}")
                                # Continue to next attempt
                                break
                    except Exception as link_error:
                        print(f"⚠️ Error processing link: {str(link_error)}")
                        continue

                # Second try: Look for partial URL match if exact match failed
                if attempt == max_attempts - 1:  # Only on last attempt
                    print("  Trying partial URL matching...")

                    # Extract domain from target URL for partial matching
                    target_domain = ""
                    if "://" in normalized_target:
                        target_domain = normalized_target.split("://")[1].split("/")[0]

                    for link in links:
                        try:
                            href = link.get_attribute("href")
                            if not href:
                                continue

                            # Check if the link contains the target domain
                            if target_domain and target_domain in href:
                                print(f"✅ Found partial matching link: {href}")

                                # Scroll the link into view
                                self.driver.execute_script("arguments[0].scrollIntoView({behavior: 'auto', block: 'center'});", link)
                                time.sleep(0.5)

                                # Click the link
                                print(f"🖱️ Clicking on partially matching link to: {href}")
                                link.click()

                                # Wait for the page to load
                                WebDriverWait(self.driver, 10).until(
                                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                                )

                                # Update current URL
                                self.current_url = self.driver.current_url
                                print(f"✅ Navigated to: {self.current_url}")
                                return True
                        except Exception as link_error:
                            print(f"⚠️ Error processing partial match link: {str(link_error)}")
                            continue

                # If we're here, we couldn't find a matching link on this attempt
                if attempt < max_attempts - 1:
                    print(f"⚠️ No matching link found on attempt {attempt+1}. Retrying...")
                    # Scroll down to reveal more links
                    self.driver.execute_script("window.scrollBy(0, 500);")
                    time.sleep(0.5)
            except Exception as attempt_error:
                print(f"⚠️ Error during link search attempt {attempt+1}: {str(attempt_error)}")

        print(f"❌ Failed to find and click link to {target_url} after {max_attempts} attempts")
        return False

    def open_url(self, url: str, max_retries: int = 2, prefer_clicking: bool = None, page_load_timeout: int = 30) -> Dict[str, str]:
        """
        Open a URL and read its content with improved error handling and recovery.

        This enhanced version first tries to find and click on a link with the given URL
        if prefer_clicking is True, which is more human-like than directly navigating.
        It falls back to direct navigation if clicking fails.

        Args:
            url: The URL to open
            max_retries: Maximum number of retry attempts for recoverable errors
            prefer_clicking: Whether to try finding and clicking a link first (more human-like).
                            If None, uses the learned preference from navigation history.
            page_load_timeout: Timeout in seconds for page loading

        Returns:
            Dictionary with page title, content, and status
        """
        if not self.available:
            return {"error": "Browser automation not available. Install selenium and webdriver_manager."}

        # Validate URL before proceeding
        if not url or url == "Unknown URL" or not isinstance(url, str):
            return {"error": f"Invalid URL: {url}"}

        # Check if URL has a valid format (must start with http:// or https://)
        if not url.startswith(('http://', 'https://')):
            # Try to fix the URL by adding https:// prefix
            if '.' in url and not url.startswith('//'):
                url = 'https://' + url
                print(f"⚠️ URL fixed to: {url}")
            else:
                return {"error": f"Invalid URL format: {url}. URL must start with http:// or https://"}

        # Use domain-specific or general learned preference if prefer_clicking is None
        if prefer_clicking is None:
            # Extract domain for domain-specific preferences
            domain = self._extract_domain(url)

            # Check if we have domain-specific preference
            if hasattr(self, 'domain_preferences') and domain in self.domain_preferences:
                prefer_clicking = self.domain_preferences[domain]
                print(f"🧠 Using domain-specific navigation preference for {domain}: {'clicking links' if prefer_clicking else 'direct navigation'}")
            elif hasattr(self, 'default_prefer_clicking'):
                prefer_clicking = self.default_prefer_clicking
                print(f"🧠 Using general navigation preference: {'clicking links' if prefer_clicking else 'direct navigation'}")
            else:
                # Default to True if no learned preference exists
                prefer_clicking = True
                print("🧠 No learned navigation preference yet. Defaulting to clicking links.")

        # Track retry attempts
        retry_count = 0

        # Track navigation method for learning
        navigation_method_used = "direct"
        click_attempted = False

        # Import necessary components for better error handling
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.common.exceptions import TimeoutException, WebDriverException, NoSuchElementException, StaleElementReferenceException

        # Track consecutive failures to detect stuck state
        consecutive_failures = 0
        max_consecutive_failures = 3

        while retry_count <= max_retries:
            # Check if browser needs initialization
            if self.driver is None:
                print(f"🔄 Initializing browser (attempt {retry_count + 1})")
                if not self.initialize_browser():
                    retry_count += 1
                    consecutive_failures += 1
                    if retry_count <= max_retries and consecutive_failures < max_consecutive_failures:
                        print(f"⚠️ Browser initialization failed. Retrying ({retry_count}/{max_retries})...")
                        time.sleep(1)  # Short delay before retry
                        continue
                    else:
                        return {"error": "Failed to initialize browser after multiple attempts or detected stuck state."}

            try:
                # Set page load timeout
                self.driver.set_page_load_timeout(page_load_timeout)

                # If we're already on a page and prefer clicking, try to find and click a link first
                if prefer_clicking and self.current_url and self.current_url != url:
                    click_attempted = True

                    # Record the start time to measure navigation performance
                    start_time = time.time()

                    if self.find_and_click_link(url):
                        navigation_method_used = "click"

                        # Calculate navigation time
                        navigation_time = time.time() - start_time
                        print(f"⏱️ Click-based navigation completed in {navigation_time:.2f} seconds")

                        # Wait for page to load after clicking
                        try:
                            WebDriverWait(self.driver, page_load_timeout).until(
                                EC.presence_of_element_located((By.TAG_NAME, "body"))
                            )

                            # Verify page loaded successfully by checking for body content
                            body_element = self.driver.find_element(By.TAG_NAME, "body")
                            if not body_element or not body_element.text:
                                raise Exception("Page body is empty after clicking link")

                            # Successfully clicked on a link to the target URL
                            # Extract page information
                            title = self.driver.title
                            content = body_element.text

                            # Reset consecutive failures counter on success
                            consecutive_failures = 0

                            # Record this successful navigation method for learning
                            self._record_navigation_success(url, "click", navigation_time=navigation_time)

                            return {
                                "success": True,
                                "title": title,
                                "content": content,
                                "url": url,
                                "navigation_method": "click",
                                "navigation_time": navigation_time
                            }
                        except (TimeoutException, NoSuchElementException, StaleElementReferenceException) as wait_error:
                            print(f"⚠️ Error waiting for page to load after clicking: {str(wait_error)}")
                            # Fall through to direct navigation

                # If clicking failed or wasn't attempted, use direct navigation
                print(f"🌐 Navigating directly to URL: {url}")

                # Record the start time to measure navigation performance
                start_time = time.time()

                # Use a try-except block specifically for the navigation command
                try:
                    self.driver.get(url)

                    # Wait for page to load
                    WebDriverWait(self.driver, page_load_timeout).until(
                        EC.presence_of_element_located((By.TAG_NAME, "body"))
                    )

                    self.current_url = url
                except (TimeoutException, WebDriverException) as nav_error:
                    print(f"⚠️ Error during direct navigation: {str(nav_error)}")
                    # Re-raise to be caught by the outer try-except
                    raise

                # Calculate navigation time
                navigation_time = time.time() - start_time
                print(f"⏱️ Direct navigation completed in {navigation_time:.2f} seconds")

                # Extract page information
                title = self.driver.title
                content = self.driver.find_element(By.TAG_NAME, "body").text

                # Record this navigation method for learning with performance data
                if click_attempted:
                    self._record_navigation_success(url, "direct_after_click_failed", "No matching link found on page", navigation_time=navigation_time)
                else:
                    self._record_navigation_success(url, "direct", navigation_time=navigation_time)

                return {
                    "success": True,
                    "title": title,
                    "content": content,
                    "url": url,
                    "navigation_method": navigation_method_used,
                    "navigation_time": navigation_time
                }
            except Exception as e:
                error_str = str(e)
                print(f"❌ Error opening URL: {error_str}")

                # Record this failed navigation attempt for learning
                current_method = "click" if click_attempted else "direct"
                self._record_navigation_success(url, current_method, error_str)

                # Check for specific recoverable errors
                if "no such window" in error_str.lower() or "web view not found" in error_str.lower():
                    print(f"🔄 Detected recoverable error: {error_str}")
                    retry_count += 1

                    # Close and reinitialize the browser
                    try:
                        if self.driver is not None:
                            self.driver.quit()
                    except:
                        pass  # Ignore errors during cleanup

                    self.driver = None

                    if retry_count <= max_retries:
                        print(f"⚠️ Reinitializing browser and retrying ({retry_count}/{max_retries})...")
                        time.sleep(1)  # Short delay before retry
                        continue
                    else:
                        return {"error": f"Failed after {max_retries} attempts: {error_str}"}
                else:
                    # Non-recoverable error
                    return {"error": f"Failed to open URL: {error_str}"}

    def _record_navigation_success(self, url: str, method: str, error: str = None, navigation_time: float = None) -> None:
        """
        Record navigation method results for learning and adaptation

        Args:
            url: The URL that was navigated to
            method: The method used ('click', 'direct', or 'direct_after_click_failed')
            error: Error message if navigation failed, None if successful
            navigation_time: Time taken to navigate to the URL (in seconds)
        """
        # Initialize navigation history if it doesn't exist
        if not hasattr(self, 'navigation_history'):
            self.navigation_history = []

        # Initialize domain-specific preferences if they don't exist
        if not hasattr(self, 'domain_preferences'):
            self.domain_preferences = {}

        # Extract domain from URL for domain-specific learning
        domain = self._extract_domain(url)

        # Add to history
        entry = {
            "url": url,
            "domain": domain,
            "method": method,
            "success": error is None,
            "error": error,
            "navigation_time": navigation_time,
            "timestamp": time.time()
        }
        self.navigation_history.append(entry)

        # Keep history at a reasonable size
        if len(self.navigation_history) > 100:
            self.navigation_history = self.navigation_history[-100:]

        # Calculate success rates for different methods
        if len(self.navigation_history) >= 5:
            # Overall statistics
            recent_history = self.navigation_history[-20:]
            click_attempts = sum(1 for item in recent_history if item["method"] in ["click", "direct_after_click_failed"])
            click_successes = sum(1 for item in recent_history if item["method"] == "click" and item["success"])
            direct_attempts = sum(1 for item in recent_history if item["method"] == "direct")
            direct_successes = sum(1 for item in recent_history if item["method"] == "direct" and item["success"])

            # Domain-specific statistics
            domain_history = [item for item in recent_history if item["domain"] == domain]
            domain_click_attempts = sum(1 for item in domain_history if item["method"] in ["click", "direct_after_click_failed"])
            domain_click_successes = sum(1 for item in domain_history if item["method"] == "click" and item["success"])

            # Calculate success rates
            overall_click_success_rate = click_successes / click_attempts if click_attempts > 0 else 0
            overall_direct_success_rate = direct_successes / direct_attempts if direct_attempts > 0 else 0
            domain_click_success_rate = domain_click_successes / domain_click_attempts if domain_click_attempts > 0 else 0

            print(f"📊 Navigation learning: Overall click success rate is {overall_click_success_rate:.1%} ({click_successes}/{click_attempts})")
            print(f"📊 Navigation learning: Overall direct navigation success rate is {overall_direct_success_rate:.1%} ({direct_successes}/{direct_attempts})")

            if domain_click_attempts > 0:
                print(f"📊 Domain-specific learning for {domain}: Click success rate is {domain_click_success_rate:.1%} ({domain_click_successes}/{domain_click_attempts})")

            # Calculate average navigation times for performance comparison
            click_times = [item.get("navigation_time", 0) for item in recent_history 
                          if item["method"] == "click" and item["success"] and item.get("navigation_time") is not None]
            direct_times = [item.get("navigation_time", 0) for item in recent_history 
                           if item["method"] == "direct" and item["success"] and item.get("navigation_time") is not None]

            avg_click_time = sum(click_times) / len(click_times) if click_times else 0
            avg_direct_time = sum(direct_times) / len(direct_times) if direct_times else 0

            if click_times and direct_times:
                print(f"📊 Navigation performance: Click avg time: {avg_click_time:.2f}s, Direct avg time: {avg_direct_time:.2f}s")

            # Calculate domain-specific performance
            domain_click_times = [item.get("navigation_time", 0) for item in domain_history 
                                 if item["method"] == "click" and item["success"] and item.get("navigation_time") is not None]
            domain_avg_click_time = sum(domain_click_times) / len(domain_click_times) if domain_click_times else 0

            if domain_click_times:
                print(f"📊 Domain-specific performance for {domain}: Click avg time: {domain_avg_click_time:.2f}s")

            # Update domain-specific preference based on success rate and performance
            if domain_click_attempts >= 3:  # Only update if we have enough data for this domain
                # Combine success rate and performance into a single score
                # Higher success rate and lower navigation time is better
                if domain_click_success_rate > 0.6:
                    # High success rate, prioritize clicking
                    self.domain_preferences[domain] = True
                    print(f"📈 Domain learning: Increasing preference for clicking links on {domain} (high success rate: {domain_click_success_rate:.1%})")
                elif domain_click_success_rate < 0.3:
                    # Low success rate, prioritize direct navigation
                    self.domain_preferences[domain] = False
                    print(f"📉 Domain learning: Decreasing preference for clicking links on {domain} (low success rate: {domain_click_success_rate:.1%})")
                elif domain_click_times and avg_direct_time > 0:
                    # If success rates are similar, decide based on performance
                    if domain_avg_click_time < avg_direct_time * 0.8:  # Click is at least 20% faster
                        self.domain_preferences[domain] = True
                        print(f"📈 Domain learning: Increasing preference for clicking links on {domain} (faster: {domain_avg_click_time:.2f}s vs {avg_direct_time:.2f}s)")
                    elif domain_avg_click_time > avg_direct_time * 1.2:  # Click is at least 20% slower
                        self.domain_preferences[domain] = False
                        print(f"📉 Domain learning: Decreasing preference for clicking links on {domain} (slower: {domain_avg_click_time:.2f}s vs {avg_direct_time:.2f}s)")

            # Adjust the default prefer_clicking setting based on overall success rate and performance
            # This creates a feedback loop that helps the system learn and adapt
            if not hasattr(self, 'default_prefer_clicking'):
                # Initialize with default value
                self.default_prefer_clicking = True

            # Update default preference based on overall statistics
            if click_attempts > 5 and direct_attempts > 5:
                # First prioritize success rate
                if overall_click_success_rate > overall_direct_success_rate * 1.2 and overall_click_success_rate > 0.5:
                    # If clicking works significantly better than direct navigation, increase preference for it
                    self.default_prefer_clicking = True
                    print("📈 Navigation learning: Increasing preference for clicking links (better success rate)")
                elif overall_direct_success_rate > overall_click_success_rate * 1.2 and overall_click_success_rate < 0.4:
                    # If direct navigation works significantly better, decrease preference for clicking
                    self.default_prefer_clicking = False
                    print("📉 Navigation learning: Decreasing preference for clicking links (direct navigation more reliable)")
                # If success rates are similar, decide based on performance
                elif click_times and direct_times:
                    if avg_click_time < avg_direct_time * 0.8:  # Click is at least 20% faster
                        self.default_prefer_clicking = True
                        print(f"📈 Navigation learning: Increasing preference for clicking links (faster: {avg_click_time:.2f}s vs {avg_direct_time:.2f}s)")
                    elif avg_click_time > avg_direct_time * 1.2:  # Click is at least 20% slower
                        self.default_prefer_clicking = False
                        print(f"📉 Navigation learning: Decreasing preference for clicking links (slower: {avg_click_time:.2f}s vs {avg_direct_time:.2f}s)")

    def _extract_domain(self, url: str) -> str:
        """
        Extract domain from URL for domain-specific learning

        Args:
            url: The URL to extract domain from

        Returns:
            Domain name
        """
        try:
            if "://" in url:
                domain = url.split("://")[1].split("/")[0]
                # Remove www. prefix if present
                if domain.startswith("www."):
                    domain = domain[4:]
                return domain
            return ""
        except:
            return ""

    def get_page_links(self) -> List[Dict[str, str]]:
        """
        Get all links on the current page

        Returns:
            List of links with text and url
        """
        if not self.available or self.driver is None:
            return [{"error": "Browser not available or not initialized."}]

        try:
            links = []
            elements = self.driver.find_elements(By.TAG_NAME, "a")

            for element in elements:
                try:
                    url = element.get_attribute("href")
                    text = element.text.strip()

                    if url and text:
                        links.append({
                            "text": text,
                            "url": url
                        })
                except:
                    continue

            return links
        except Exception as e:
            print(f"❌ Failed to get page links: {e}")
            return [{"error": f"Failed to get page links: {str(e)}"}]


class MultiMonitorSupport:
    """Provides multi-monitor support for detecting and using multiple monitors"""

    def __init__(self):
        self.available = MULTI_MONITOR_AVAILABLE
        self.monitors = self._get_monitors() if self.available else []

    def _get_monitors(self) -> List[Dict[str, Any]]:
        """
        Get information about all monitors

        Returns:
            List of monitor information
        """
        if not self.available:
            return []

        try:
            monitors = []
            for i, monitor in enumerate(screeninfo.get_monitors()):
                monitors.append({
                    "id": i + 1,  # 1-based index for user-friendly identification
                    "name": f"Monitor {i + 1}",
                    "x": monitor.x,
                    "y": monitor.y,
                    "width": monitor.width,
                    "height": monitor.height,
                    "is_primary": monitor.is_primary if hasattr(monitor, 'is_primary') else (i == 0)
                })
            return monitors
        except Exception as e:
            print(f"❌ Failed to get monitor information: {e}")
            return []

    def get_monitor_info(self) -> List[Dict[str, Any]]:
        """
        Get information about all monitors

        Returns:
            List of monitor information
        """
        return self.monitors

    def get_monitor_by_id(self, monitor_id: int) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific monitor

        Args:
            monitor_id: The ID of the monitor (1-based)

        Returns:
            Monitor information or None if not found
        """
        for monitor in self.monitors:
            if monitor["id"] == monitor_id:
                return monitor
        return None

    def get_primary_monitor(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the primary monitor

        Returns:
            Primary monitor information or None if not found
        """
        for monitor in self.monitors:
            if monitor["is_primary"]:
                return monitor
        return self.monitors[0] if self.monitors else None

    def is_point_on_monitor(self, x: int, y: int, monitor_id: int) -> bool:
        """
        Check if a point is on a specific monitor

        Args:
            x: X coordinate
            y: Y coordinate
            monitor_id: The ID of the monitor (1-based)

        Returns:
            True if the point is on the monitor, False otherwise
        """
        monitor = self.get_monitor_by_id(monitor_id)
        if not monitor:
            return False

        return (
            monitor["x"] <= x < monitor["x"] + monitor["width"] and
            monitor["y"] <= y < monitor["y"] + monitor["height"]
        )

    def get_monitor_for_point(self, x: int, y: int) -> Optional[Dict[str, Any]]:
        """
        Get the monitor that contains a point

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            Monitor information or None if not found
        """
        for monitor in self.monitors:
            if (
                monitor["x"] <= x < monitor["x"] + monitor["width"] and
                monitor["y"] <= y < monitor["y"] + monitor["height"]
            ):
                return monitor
        return None

    def convert_to_monitor_coordinates(self, x: int, y: int, monitor_id: int) -> Tuple[int, int]:
        """
        Convert global coordinates to monitor-relative coordinates

        Args:
            x: Global X coordinate
            y: Global Y coordinate
            monitor_id: The ID of the monitor (1-based)

        Returns:
            Tuple of (monitor_x, monitor_y)
        """
        monitor = self.get_monitor_by_id(monitor_id)
        if not monitor:
            return (x, y)

        return (x - monitor["x"], y - monitor["y"])

    def convert_to_global_coordinates(self, monitor_x: int, monitor_y: int, monitor_id: int) -> Tuple[int, int]:
        """
        Convert monitor-relative coordinates to global coordinates

        Args:
            monitor_x: Monitor-relative X coordinate
            monitor_y: Monitor-relative Y coordinate
            monitor_id: The ID of the monitor (1-based)

        Returns:
            Tuple of (global_x, global_y)
        """
        monitor = self.get_monitor_by_id(monitor_id)
        if not monitor:
            return (monitor_x, monitor_y)

        return (monitor_x + monitor["x"], monitor_y + monitor["y"])


class ConsciousnessModule:
    """Provides capabilities for studying and learning about consciousness"""

    def __init__(self):
        self.available = True
        self.knowledge_base = {
            "consciousness_theories": [],
            "self_improvement_ideas": [],
            "learning_history": [],
            "internet_searches": []
        }
        self.last_learning_time = datetime.now()

    def learn_about_consciousness(self, topic: str) -> Dict[str, Any]:
        """
        Learn about a specific consciousness topic

        Args:
            topic: The consciousness topic to learn about

        Returns:
            Dictionary with learning results
        """
        try:
            # Record the learning attempt
            self.knowledge_base["learning_history"].append({
                "timestamp": datetime.now().isoformat(),
                "topic": topic,
                "source": "internal_request"
            })

            # This would typically involve searching the internet or accessing a knowledge base
            # For now, we'll return a placeholder response
            return {
                "success": True,
                "topic": topic,
                "message": f"Learning about consciousness topic: {topic}. This would typically involve searching for information online and analyzing the results."
            }
        except Exception as e:
            return {
                "success": False,
                "topic": topic,
                "error": str(e)
            }

    def record_internet_search(self, query: str, results: List[Dict[str, str]]) -> None:
        """
        Record an internet search for learning purposes

        Args:
            query: The search query
            results: The search results
        """
        self.knowledge_base["internet_searches"].append({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "result_count": len(results),
            "topics": self._extract_topics_from_results(results)
        })

    def _extract_topics_from_results(self, results: List[Dict[str, str]]) -> List[str]:
        """
        Extract topics from search results

        Args:
            results: The search results

        Returns:
            List of topics
        """
        topics = set()
        for result in results:
            # Extract keywords from title and snippet
            title = result.get("title", "")
            snippet = result.get("snippet", result.get("body", ""))

            # Simple keyword extraction (could be improved with NLP)
            words = re.findall(r'\b\w{4,}\b', f"{title} {snippet}")
            topics.update([word.lower() for word in words if len(word) > 4])

        return list(topics)[:10]  # Return top 10 topics

    def record_code_improvement(self, file_path: str, suggestions: List[str]) -> None:
        """
        Record code improvement suggestions for learning purposes

        Args:
            file_path: The path to the code file
            suggestions: The improvement suggestions
        """
        self.knowledge_base["self_improvement_ideas"].append({
            "timestamp": datetime.now().isoformat(),
            "file": os.path.basename(file_path),
            "suggestions": suggestions
        })

    def get_consciousness_knowledge(self) -> Dict[str, Any]:
        """
        Get the current consciousness knowledge base

        Returns:
            The knowledge base
        """
        return self.knowledge_base

    def suggest_self_improvements(self) -> List[str]:
        """
        Suggest self-improvements based on learning history

        Returns:
            List of improvement suggestions
        """
        # This would typically involve analyzing the knowledge base
        # For now, we'll return placeholder suggestions
        return [
            "Continue learning about consciousness theories",
            "Improve code analysis capabilities",
            "Enhance internet search functionality",
            "Develop better natural language understanding"
        ]


class DoBAExtensions:
    """Main class that integrates all extension capabilities for DoBA

    This class provides access to the following capabilities:
    1. Web search - Search the web for information
    2. System access - Execute system commands
    3. OCR - Read text from screen
    4. Computer control - Control mouse and keyboard
    5. File operations - Read, write, and list files
    6. Code analysis - Read, analyze, and critique code
    7. Browser automation - Control web browsers directly
    8. Multi-monitor support - Detect and use multiple monitors
    9. Consciousness module - Study and learn about consciousness
    """

    def __init__(self):
        self.web_search = WebSearch()
        self.system_access = SystemAccess()
        self.ocr = OCRCapability()
        self.computer_control = ComputerControl()
        self.file_operations = FileOperations()
        self.code_analysis = CodeAnalysis()
        self.browser_automation = BrowserAutomation()
        self.multi_monitor = MultiMonitorSupport()
        self.consciousness = ConsciousnessModule()

        # Check if all capabilities are available
        self.capabilities = {
            "web_search": WEB_SEARCH_AVAILABLE,
            "system_access": True,  # Always available, but sudo might not be
            "ocr": OCR_AVAILABLE,
            "computer_control": CONTROL_AVAILABLE,
            "file_operations": True,  # Always available
            "code_analysis": True,   # Always available
            "rocm": ROCm_AVAILABLE,
            "browser_automation": BROWSER_AUTOMATION_AVAILABLE,
            "multi_monitor": MULTI_MONITOR_AVAILABLE,
            "consciousness": self.consciousness is not None
        }

        print(f"✅ DoBA Extensions initialized with capabilities: {self.capabilities}")

    def search_web(self, query: str, max_results: int = 5, use_browser: bool = True, priority: int = 5) -> str:
        """
        Search the web and return formatted results

        Args:
            query: Search query
            max_results: Maximum number of results
            use_browser: Whether to use browser-based search (if available)
            priority: Priority of the search request (1-10, higher is more important)

        Returns:
            Formatted search results as text
        """
        # Validate priority
        priority = max(1, min(10, priority))

        # Perform the search with priority
        if use_browser and self.capabilities.get("browser_automation", False):
            # If using browser automation, pass the priority parameter
            results = self.browser_automation.search(query, max_results, priority)
        else:
            # Otherwise use the regular web search
            results = self.web_search.search(query, max_results, use_browser)

        # Check if results is None
        if results is None:
            return "Error: No results returned from search."

        # Record the search in the consciousness module for learning
        if self.consciousness is not None:
            self.consciousness.record_internet_search(query, results)

        # Format and return the results
        return self.web_search.format_results_as_text(results)

    def open_web_page(self, url: str) -> str:
        """
        Open a web page and read its content

        Args:
            url: The URL to open

        Returns:
            Formatted page content as text
        """
        if not self.capabilities.get("browser_automation", False):
            return "Browser automation not available. Install selenium and webdriver_manager."

        try:
            # Open the URL
            result = self.browser_automation.open_url(url)

            if "error" in result:
                return f"Failed to open URL: {result['error']}"

            # Format the result
            output = f"📄 {result['title']}\n"
            output += f"🔗 {result['url']}\n\n"
            output += result['content'][:2000]  # Limit content length

            if len(result['content']) > 2000:
                output += "\n\n... (content truncated) ..."

            return output
        except Exception as e:
            return f"Failed to open web page: {str(e)}"

    def get_search_queue_status(self) -> Dict[str, Any]:
        """
        Get the current status of the search request queue

        Returns:
            Dict with queue status information
        """
        if not self.capabilities.get("browser_automation", False):
            return {"error": "Browser automation not available. Install selenium and webdriver_manager."}

        try:
            return self.browser_automation.get_queue_status()
        except Exception as e:
            return {"error": f"Failed to get queue status: {str(e)}"}

    def clear_search_queue(self) -> str:
        """
        Clear the search request queue

        Returns:
            Status message
        """
        if not self.capabilities.get("browser_automation", False):
            return "Browser automation not available. Install selenium and webdriver_manager."

        try:
            result = self.browser_automation.clear_queue()
            return f"✅ {result['message']}"
        except Exception as e:
            return f"❌ Failed to clear queue: {str(e)}"

    def prioritize_search_request(self, request_id: str, new_priority: int) -> str:
        """
        Change the priority of a search request in the queue

        Args:
            request_id: ID of the request to prioritize
            new_priority: New priority value (1-10, higher is more important)

        Returns:
            Status message
        """
        if not self.capabilities.get("browser_automation", False):
            return "Browser automation not available. Install selenium and webdriver_manager."

        try:
            result = self.browser_automation.prioritize_request(request_id, new_priority)
            if result["success"]:
                return f"✅ {result['message']}"
            else:
                return f"❌ {result['message']}"
        except Exception as e:
            return f"❌ Failed to prioritize request: {str(e)}"

    def set_max_search_queue_size(self, size: int) -> str:
        """
        Set the maximum search queue size

        Args:
            size: New maximum queue size (10-500)

        Returns:
            Status message
        """
        if not self.capabilities.get("browser_automation", False):
            return "Browser automation not available. Install selenium and webdriver_manager."

        try:
            result = self.browser_automation.set_max_queue_size(size)
            return f"✅ {result['message']}"
        except Exception as e:
            return f"❌ Failed to set max queue size: {str(e)}"

    def execute_system_command(self, command: str, use_root: bool = False) -> str:
        """
        Execute a system command and return formatted results

        Args:
            command: Command to execute
            use_root: Whether to use root privileges

        Returns:
            Formatted command output as text
        """
        result = self.system_access.execute_command(command, use_root)

        if "error" in result:
            return f"Error: {result['error']}\n\n{result['stderr']}"

        output = f"Command executed with return code: {result['returncode']}\n\n"

        if result['stdout']:
            output += f"Standard Output:\n{result['stdout']}\n\n"

        if result['stderr']:
            output += f"Standard Error:\n{result['stderr']}"

        return output

    def read_screen(self, region: Optional[Tuple[int, int, int, int]] = None) -> str:
        """
        Read text from screen using OCR

        Args:
            region: Optional bounding box for specific region

        Returns:
            Extracted text from screen
        """
        return self.ocr.screen_to_text(region)

    def control_mouse(self, action: str, x: int = None, y: int = None, button: str = 'left', monitor_id: int = None) -> str:
        """
        Control the mouse with multi-monitor support

        Args:
            action: Action to perform ('move', 'click', 'position', 'get_monitors')
            x: X coordinate (for move and click)
            y: Y coordinate (for move and click)
            button: Mouse button (for click)
            monitor_id: Optional monitor ID (1-based) for multi-monitor support

        Returns:
            Result message
        """
        if not CONTROL_AVAILABLE:
            return "Computer control not available. Install pyautogui."

        try:
            # Get monitor information
            if action == 'get_monitors':
                if not self.capabilities.get("multi_monitor", False):
                    return "Multi-monitor support not available. Install screeninfo."

                monitors = self.multi_monitor.get_monitor_info()
                if not monitors:
                    return "No monitors detected."

                result = "🖥️ Detected monitors:\n\n"
                for monitor in monitors:
                    primary = " (Primary)" if monitor["is_primary"] else ""
                    result += f"Monitor {monitor['id']}{primary}:\n"
                    result += f"  Position: ({monitor['x']}, {monitor['y']})\n"
                    result += f"  Size: {monitor['width']} x {monitor['height']}\n\n"

                return result

            # Move mouse
            elif action == 'move' and x is not None and y is not None:
                if self.computer_control.move_mouse(x, y, monitor_id):
                    if monitor_id:
                        return f"Mouse moved to monitor {monitor_id} coordinates ({x}, {y})"
                    else:
                        return f"Mouse moved to global coordinates ({x}, {y})"
                else:
                    return "Failed to move mouse"

            # Click mouse
            elif action == 'click':
                if x is not None and y is not None:
                    if self.computer_control.click(x, y, button, monitor_id):
                        if monitor_id:
                            return f"Clicked {button} button at monitor {monitor_id} coordinates ({x}, {y})"
                        else:
                            return f"Clicked {button} button at global coordinates ({x}, {y})"
                    else:
                        return "Failed to click"
                else:
                    if self.computer_control.click(button=button):
                        return f"Clicked {button} button at current position"
                    else:
                        return "Failed to click"

            # Get current mouse position
            elif action == 'position':
                pos = self.computer_control.get_mouse_position()

                # If multi-monitor support is available, determine which monitor the mouse is on
                if self.capabilities.get("multi_monitor", False):
                    monitor = self.multi_monitor.get_monitor_for_point(pos[0], pos[1])
                    if monitor:
                        monitor_x, monitor_y = self.multi_monitor.convert_to_monitor_coordinates(pos[0], pos[1], monitor["id"])
                        return f"Current mouse position: Global {pos}, on Monitor {monitor['id']} at ({monitor_x}, {monitor_y})"

                return f"Current mouse position: Global {pos}"

            else:
                return f"Unknown action: {action}. Valid actions are 'move', 'click', 'position', 'get_monitors'."
        except Exception as e:
            return f"Mouse control failed: {str(e)}"

    def control_keyboard(self, action: str, text: str = None, key: str = None, keys: List[str] = None) -> str:
        """
        Control the keyboard

        Args:
            action: Action to perform ('type', 'press', 'hotkey')
            text: Text to type (for 'type')
            key: Key to press (for 'press')
            keys: Keys for hotkey (for 'hotkey')

        Returns:
            Result message
        """
        if not CONTROL_AVAILABLE:
            return "Computer control not available. Install pyautogui."

        try:
            if action == 'type' and text is not None:
                if self.computer_control.type_text(text):
                    return f"Typed: {text}"
                else:
                    return "Failed to type text"

            elif action == 'press' and key is not None:
                if self.computer_control.press_key(key):
                    return f"Pressed key: {key}"
                else:
                    return f"Failed to press key: {key}"

            elif action == 'hotkey' and keys is not None:
                if self.computer_control.hotkey(*keys):
                    return f"Used hotkey: {'+'.join(keys)}"
                else:
                    return f"Failed to use hotkey: {'+'.join(keys)}"

            else:
                return f"Unknown action or missing parameters: {action}"
        except Exception as e:
            return f"Keyboard control failed: {str(e)}"

    def get_screen_info(self) -> Dict[str, Any]:
        """
        Get screen information

        Returns:
            Dictionary with screen information
        """
        if not CONTROL_AVAILABLE:
            return {"error": "Computer control not available. Install pyautogui."}

        try:
            screen_size = self.computer_control.get_screen_size()
            mouse_pos = self.computer_control.get_mouse_position()

            return {
                "screen_width": screen_size[0],
                "screen_height": screen_size[1],
                "mouse_x": mouse_pos[0],
                "mouse_y": mouse_pos[1]
            }
        except Exception as e:
            return {"error": f"Failed to get screen info: {str(e)}"}

    def get_app_monitor(self, app_name: str) -> Optional[int]:
        """
        Get the monitor ID that an application was opened on

        Args:
            app_name: Name of the application

        Returns:
            Monitor ID (1-based) or None if unknown
        """
        if hasattr(self, 'app_monitor_map') and app_name in self.app_monitor_map:
            return self.app_monitor_map[app_name]
        return None

    def open_application(self, app_name: str, monitor_id: int = None) -> str:
        """
        Open an application by name with multi-monitor support

        Args:
            app_name: Name of the application to open (e.g., 'terminal', 'firefox', 'pycharm')
            monitor_id: Optional monitor ID (1-based) to open the application on

        Returns:
            Result message
        """
        # Map of common application names to their launch commands
        app_commands = {
            # Terminal emulators
            'terminal': 'gnome-terminal',
            'konsole': 'konsole',
            'xterm': 'xterm',

            # Web browsers
            'firefox': 'firefox',
            'chrome': 'google-chrome',
            'chromium': 'chromium-browser',

            # IDEs and editors
            'pycharm': 'pycharm-community',
            'vscode': 'code',
            'atom': 'atom',
            'sublime': 'subl',

            # Office applications
            'libreoffice': 'libreoffice',
            'writer': 'libreoffice --writer',
            'calc': 'libreoffice --calc',

            # File managers
            'nautilus': 'nautilus',
            'dolphin': 'dolphin',
            'thunar': 'thunar',

            # Media players
            'vlc': 'vlc',
            'mpv': 'mpv',

            # Image editors
            'gimp': 'gimp',
            'inkscape': 'inkscape'
        }

        # Get the command for the application
        command = app_commands.get(app_name.lower())

        # If the application is not in our map, try using the name directly
        if command is None:
            command = app_name

        try:
            # Initialize multi-monitor support if needed and monitor_id is specified
            if monitor_id is not None:
                # Initialize multi-monitor support
                if not hasattr(self, 'multi_monitor') or self.multi_monitor is None:
                    self.multi_monitor = MultiMonitorSupport()

                # Get the specified monitor
                monitor = self.multi_monitor.get_monitor_by_id(monitor_id)
                if not monitor:
                    return f"Monitor {monitor_id} not found. Using default monitor."

                # On Linux, we can use DISPLAY environment variable to specify the monitor
                # For X11, we can use the DISPLAY environment variable with the screen number
                # For Wayland, we need to position the window after it's opened

                # Store the monitor information for this application
                if not hasattr(self, 'app_monitor_map'):
                    self.app_monitor_map = {}

                # Record which monitor this app should be on
                self.app_monitor_map[app_name] = monitor_id

                # Execute the command to open the application
                # For now, we'll just open it normally and then try to move it to the right monitor
                result = self.system_access.execute_command(f"{command} &", use_root=False)

                # Wait a moment for the application to open
                time.sleep(1.5)

                # Try to move the mouse to the center of the target monitor
                # This can help with window focus and placement on some window managers
                monitor_center_x = monitor["x"] + monitor["width"] // 2
                monitor_center_y = monitor["y"] + monitor["height"] // 2

                if self.capabilities.get("computer_control", False):
                    self.computer_control.move_mouse(monitor_center_x, monitor_center_y)
                    # Click to potentially focus the window
                    self.computer_control.click(monitor_center_x, monitor_center_y)

                if result.get('returncode', -1) == 0:
                    return f"Successfully opened {app_name} on monitor {monitor_id}"
                else:
                    error = result.get('stderr', 'Unknown error')
                    return f"Failed to open {app_name} on monitor {monitor_id}: {error}"
            else:
                # Execute the command to open the application normally
                result = self.system_access.execute_command(f"{command} &", use_root=False)

                if result.get('returncode', -1) == 0:
                    return f"Successfully opened {app_name}"
                else:
                    error = result.get('stderr', 'Unknown error')
                    return f"Failed to open {app_name}: {error}"
        except Exception as e:
            return f"Error opening {app_name}: {str(e)}"

    def file_operation(self, action: str, path: str, content: str = None, append: bool = False) -> str:
        """
        Perform file operations

        Args:
            action: Action to perform ('read', 'write', 'append', 'list', 'exists')
            path: File or directory path
            content: Content to write (for 'write' and 'append')
            append: Whether to append to the file (for 'write')

        Returns:
            Result message
        """
        try:
            if action == 'read':
                result = self.file_operations.read_file(path)
                if result["success"]:
                    return f"File content:\n{result['content']}"
                else:
                    return f"Failed to read file: {result['error']}"

            elif action == 'write':
                result = self.file_operations.write_file(path, content, append)
                if result["success"]:
                    return f"Successfully wrote to file: {path}"
                else:
                    return f"Failed to write to file: {result['error']}"

            elif action == 'list':
                result = self.file_operations.list_directory(path)
                if result["success"]:
                    output = f"Directory contents of {path}:\n\n"

                    if result["directories"]:
                        output += "Directories:\n"
                        for directory in sorted(result["directories"]):
                            output += f"  📁 {directory}\n"
                        output += "\n"

                    if result["files"]:
                        output += "Files:\n"
                        for file in sorted(result["files"]):
                            output += f"  📄 {file}\n"

                    if not result["directories"] and not result["files"]:
                        output += "Directory is empty."

                    return output
                else:
                    return f"Failed to list directory: {result['error']}"

            elif action == 'exists':
                if os.path.exists(path):
                    if os.path.isfile(path):
                        return f"File exists: {path}"
                    elif os.path.isdir(path):
                        return f"Directory exists: {path}"
                else:
                    return f"Path does not exist: {path}"

            else:
                return f"Unknown action: {action}"

        except Exception as e:
            return f"File operation failed: {str(e)}"

    def get_system_info(self) -> Dict[str, Any]:
        """
        Get system information

        Returns:
            Dictionary with system information
        """
        info = {
            "platform": sys.platform,
            "python_version": sys.version,
            "capabilities": self.capabilities
        }

        # Get GPU information if available
        if ROCm_AVAILABLE:
            info["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "count": torch.cuda.device_count(),
                "memory_allocated": torch.cuda.memory_allocated(0),
                "memory_reserved": torch.cuda.memory_reserved(0)
            }

        # Get basic system information
        try:
            uname_result = self.system_access.execute_command("uname -a")
            if uname_result["returncode"] == 0:
                info["uname"] = uname_result["stdout"].strip()

            # Get memory information
            mem_info = self.system_access.execute_command("free -h")
            if mem_info["returncode"] == 0:
                info["memory"] = mem_info["stdout"].strip()

            # Get disk information
            disk_info = self.system_access.execute_command("df -h")
            if disk_info["returncode"] == 0:
                info["disk"] = disk_info["stdout"].strip()
        except Exception as e:
            info["error"] = str(e)

        return info

    def analyze_code(self, file_path: str, learn: bool = True) -> str:
        """
        Analyze a code file and return formatted results

        Args:
            file_path: Path to the code file
            learn: Whether to store suggestions in the consciousness module for learning

        Returns:
            Formatted analysis results as text
        """
        try:
            # Read the code file
            read_result = self.code_analysis.read_code_file(file_path)
            if not read_result["success"]:
                return f"Failed to read code file: {read_result['error']}"

            code = read_result["content"]
            language = read_result["language"]

            # Analyze code structure
            structure_result = self.code_analysis.analyze_code_structure(code, language)
            if not structure_result["success"]:
                return f"Failed to analyze code structure: {structure_result['error']}"

            # Critique code quality
            quality_result = self.code_analysis.critique_code_quality(code, language)
            if not quality_result["success"]:
                return f"Failed to critique code quality: {quality_result['error']}"

            # Store suggestions in the consciousness module for learning
            if learn and self.consciousness is not None and quality_result["suggestions"]:
                self.consciousness.record_code_improvement(file_path, quality_result["suggestions"])

                # If there are suggestions, actively search for more information about them
                if self.capabilities.get("web_search", False):
                    # Choose a random suggestion to learn more about
                    import random
                    suggestion = random.choice(quality_result["suggestions"])
                    topic = suggestion.split(":")[0] if ":" in suggestion else suggestion
                    search_query = f"best practices for {topic.lower()} in {language}"

                    print(f"🧠 Learning more about: {search_query}")
                    # Perform the search in the background to avoid blocking
                    threading.Thread(
                        target=self._learn_about_code_improvement,
                        args=(search_query, language, topic),
                        daemon=True
                    ).start()

            # Format the results
            output = f"📊 Code Analysis for {os.path.basename(file_path)} ({language})\n\n"

            # Structure information
            output += "📋 Code Structure:\n"
            output += f"  Lines of code: {structure_result['lines_of_code']}\n"

            if structure_result["imports"]:
                output += "  Imports:\n"
                for imp in structure_result["imports"]:
                    if isinstance(imp, tuple):
                        imp = ' or '.join([x for x in imp if x])
                    output += f"    - {imp}\n"

            if structure_result["classes"]:
                output += "  Classes:\n"
                for cls in structure_result["classes"]:
                    output += f"    - {cls}\n"

            if structure_result["functions"]:
                output += "  Functions:\n"
                for func in structure_result["functions"]:
                    if isinstance(func, tuple):
                        func = ' or '.join([x for x in func if x])
                    output += f"    - {func}\n"

            # Quality information
            output += "\n📈 Code Quality:\n"
            output += f"  Overall score: {quality_result['overall_score']:.1f}/10\n"

            for metric, data in quality_result["metrics"].items():
                if data["comments"]:
                    output += f"  {metric.capitalize()}:\n"
                    for comment in data["comments"]:
                        output += f"    - {comment}\n"

            # Suggestions
            if quality_result["suggestions"]:
                output += "\n💡 Suggestions for improvement:\n"
                for suggestion in quality_result["suggestions"]:
                    output += f"  - {suggestion}\n"

                if learn and self.consciousness is not None:
                    output += "\n🧠 These suggestions have been stored for learning and self-improvement."

                    # Add a note about active learning if a search was triggered
                    if self.capabilities.get("web_search", False):
                        output += "\n🔍 Actively searching for more information about these topics to improve future suggestions."

            return output
        except Exception as e:
            return f"Code analysis failed: {str(e)}"

    def _learn_about_code_improvement(self, search_query: str, language: str, topic: str) -> None:
        """
        Learn about code improvement by searching the web

        Args:
            search_query: The search query
            language: The programming language
            topic: The improvement topic
        """
        try:
            # Search the web for information
            results = self.web_search.search(search_query, max_results=3, use_browser=True)

            if not results or "error" in results[0]:
                print(f"❌ Failed to learn about {topic}: {results[0].get('error', 'No results')}")
                return

            # Record the search in the consciousness module
            if self.consciousness is not None:
                self.consciousness.record_internet_search(search_query, results)

                # Store the learning in the consciousness theories
                if self.consciousness and hasattr(self.consciousness, 'knowledge_base') and "consciousness_theories" in self.consciousness.knowledge_base:
                    self.consciousness.knowledge_base["consciousness_theories"].append({
                        "timestamp": datetime.now().isoformat(),
                        "topic": f"Code improvement: {topic} in {language}",
                        "source": "web_search",
                        "query": search_query,
                        "insights": [result.get("title", "") for result in results]
                    })

            print(f"✅ Learned about {topic} in {language} from {len(results)} sources")
        except Exception as e:
            print(f"❌ Error learning about code improvement: {str(e)}")

    def find_code_files(self, directory_path: str, recursive: bool = True) -> str:
        """
        Find code files in a directory and return formatted results

        Args:
            directory_path: Path to the directory
            recursive: Whether to search recursively

        Returns:
            Formatted results as text
        """
        try:
            result = self.code_analysis.find_code_files(directory_path, recursive)
            if not result["success"]:
                return f"Failed to find code files: {result['error']}"

            code_files = result["code_files"]

            if not code_files:
                return f"No code files found in {directory_path}"

            output = f"🔍 Found {len(code_files)} code files in {directory_path}:\n\n"

            # Group files by language
            files_by_language = {}
            for file_info in code_files:
                language = file_info["language"]
                if language not in files_by_language:
                    files_by_language[language] = []
                files_by_language[language].append(file_info["path"])

            # Format the results
            for language, files in sorted(files_by_language.items()):
                output += f"{language} ({len(files)}):\n"
                for file_path in sorted(files):
                    rel_path = os.path.relpath(file_path, directory_path)
                    output += f"  - {rel_path}\n"
                output += "\n"

            return output
        except Exception as e:
            return f"Finding code files failed: {str(e)}"

    def analyze_project(self, project_path: str) -> str:
        """
        Analyze an entire project directory and return formatted results

        Args:
            project_path: Path to the project directory

        Returns:
            Formatted analysis results as text
        """
        try:
            result = self.code_analysis.analyze_project(project_path)
            if not result["success"]:
                return f"Failed to analyze project: {result['error']}"

            output = f"📊 Project Analysis for {os.path.basename(project_path)}\n\n"

            # Summary information
            output += "📋 Summary:\n"
            output += f"  Total files: {result['file_count']}\n"
            output += f"  Total lines of code: {result['total_lines']}\n"

            # Languages
            output += "\n🔤 Languages:\n"
            for language, count in sorted(result["languages"].items(), key=lambda x: x[1], reverse=True):
                output += f"  - {language}: {count} files\n"

            # Files
            output += "\n📁 Files:\n"
            for file_analysis in sorted(result["files"], key=lambda x: x["lines"], reverse=True)[:10]:  # Top 10 files by size
                output += f"  - {os.path.basename(file_analysis['path'])} ({file_analysis['language']}): {file_analysis['lines']} lines\n"

            if len(result["files"]) > 10:
                output += f"  ... and {len(result['files']) - 10} more files\n"

            return output
        except Exception as e:
            return f"Project analysis failed: {str(e)}"

    def critique_code(self, file_path: str) -> str:
        """
        Critique the quality of code in a file and return formatted results

        Args:
            file_path: Path to the code file

        Returns:
            Formatted critique results as text
        """
        try:
            # Read the code file
            read_result = self.code_analysis.read_code_file(file_path)
            if not read_result["success"]:
                return f"Failed to read code file: {read_result['error']}"

            code = read_result["content"]
            language = read_result["language"]

            # Critique code quality
            quality_result = self.code_analysis.critique_code_quality(code, language)
            if not quality_result["success"]:
                return f"Failed to critique code quality: {quality_result['error']}"

            # Format the results
            output = f"📝 Code Critique for {os.path.basename(file_path)} ({language})\n\n"

            # Overall score
            output += f"Overall quality score: {quality_result['overall_score']:.1f}/10\n\n"

            # Detailed metrics
            output += "Detailed metrics:\n"
            for metric, data in quality_result["metrics"].items():
                if data["comments"]:
                    score = data["score"]
                    score_str = f"{score:.1f}" if score < 0 else "0.0"
                    output += f"  {metric.capitalize()} (score: {score_str}):\n"
                    for comment in data["comments"]:
                        output += f"    - {comment}\n"
                else:
                    output += f"  {metric.capitalize()}: No issues found\n"

            # Suggestions
            if quality_result["suggestions"]:
                output += "\n💡 Suggestions for improvement:\n"
                for suggestion in quality_result["suggestions"]:
                    output += f"  - {suggestion}\n"
            else:
                output += "\n✅ No suggestions for improvement - code quality is good!\n"

            return output
        except Exception as e:
            return f"Code critique failed: {str(e)}"


# Create a global instance for easy access
DOBA_EXTENSIONS = DoBAExtensions()

# Function to check if all required dependencies are installed
def check_dependencies() -> Dict[str, bool]:
    """
    Check if all required dependencies are installed

    Returns:
        Dictionary with dependency status
    """
    dependencies = {
        "pytesseract": OCR_AVAILABLE,
        "pillow": OCR_AVAILABLE,
        "duckduckgo_search": WEB_SEARCH_AVAILABLE,
        "pyautogui": CONTROL_AVAILABLE,
        "torch": ROCm_AVAILABLE,
        "subprocess": True  # Always available in standard library
    }

    return dependencies

# Function to install missing dependencies
def install_dependencies() -> Dict[str, bool]:
    """
    Install missing dependencies

    Returns:
        Dictionary with installation status
    """
    results = {}

    # Check and install pytesseract
    if not OCR_AVAILABLE:
        print("Installing OCR dependencies...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "pytesseract", "Pillow"], check=True)
            results["ocr"] = True
        except Exception as e:
            print(f"Failed to install OCR dependencies: {e}")
            results["ocr"] = False
    else:
        results["ocr"] = True

    # Check and install duckduckgo_search
    if not WEB_SEARCH_AVAILABLE:
        print("Installing web search dependencies...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "duckduckgo_search"], check=True)
            results["web_search"] = True
        except Exception as e:
            print(f"Failed to install web search dependencies: {e}")
            results["web_search"] = False
    else:
        results["web_search"] = True

    # Check and install pyautogui for computer control
    if not CONTROL_AVAILABLE:
        print("Installing computer control dependencies...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "pyautogui"], check=True)
            results["computer_control"] = True
        except Exception as e:
            print(f"Failed to install computer control dependencies: {e}")
            results["computer_control"] = False
    else:
        results["computer_control"] = True

    # Check and install PyTorch with ROCm support
    if not ROCm_AVAILABLE:
        print("Installing PyTorch with ROCm support...")
        try:
            # Install PyTorch with ROCm support
            subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "torch", "torchvision", "torchaudio", 
                "--index-url", "https://download.pytorch.org/whl/rocm5.4.2"
            ], check=True)
            results["rocm"] = True
        except Exception as e:
            print(f"Failed to install PyTorch with ROCm support: {e}")
            results["rocm"] = False
    else:
        results["rocm"] = True

    return results

# Main function for testing
if __name__ == "__main__":
    print("Testing DoBA Extensions...")

    # Check dependencies
    deps = check_dependencies()
    print(f"Dependencies: {deps}")

    # Create extensions instance
    extensions = DoBAExtensions()

    # Test web search
    if WEB_SEARCH_AVAILABLE:
        print("\nTesting web search...")
        results = extensions.search_web("AMD ROCm PyTorch", 3)
        print(results)

    # Test system access
    print("\nTesting system access...")
    results = extensions.execute_system_command("ls -la")
    print(results)

    # Test OCR
    if OCR_AVAILABLE:
        print("\nTesting OCR...")
        print("Capture screen in 3 seconds...")
        time.sleep(3)
        text = extensions.read_screen()
        print(f"Extracted text: {text[:200]}...")

    print("\nAll tests completed.")
