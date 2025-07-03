import sqlite3
from datetime import datetime
import os
import re
import time
import math
import threading
import random

class SQLiteNuclearMemory:
    def __init__(self):
        self.db_file = "nuclear_memory.db"
        # Initialize caches
        self._facts_cache = {}  # Cache for recall_facts
        self._search_cache = {}  # Cache for search_facts_by_value
        self._conversation_cache = {}  # Cache for get_diverse_conversations
        self._value_cache = {}  # Cache for frequently accessed values
        self._last_logged_patterns = {}  # Track last logged patterns
        self._facts_extraction_cache = {}  # Cache for extracted facts

        # Cache statistics for monitoring
        self.cache_hits = 0
        self.cache_misses = 0
        self.last_optimization_time = time.time()

        # Initialize the database
        self.init_db()

        # Start background thread for autonomous optimization
        self._start_autonomous_optimization()

    def init_db(self):
        conn = sqlite3.connect(self.db_file)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS nuclear_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(category, key) ON CONFLICT REPLACE
            )
        ''')

        # Add the missing memory_interactions table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS memory_interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_message TEXT NOT NULL,
                ai_response TEXT,
                session_id TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                extracted_facts TEXT
            )
        ''')

        # Check if extracted_facts column exists and add it if it doesn't
        try:
            conn.execute("SELECT extracted_facts FROM memory_interactions LIMIT 1")
        except sqlite3.OperationalError:
            print("⚠️ Adding missing 'extracted_facts' column to memory_interactions table")
            try:
                conn.execute("ALTER TABLE memory_interactions ADD COLUMN extracted_facts TEXT")
                print("✅ Added 'extracted_facts' column to memory_interactions table")
            except Exception as e:
                print(f"❌ Failed to add 'extracted_facts' column: {e}")

        conn.commit()
        conn.close()

    def store_fact(self, category, key, value):
        """
        Store a fact in nuclear memory with content-based deduplication.

        Args:
            category: The category of the fact
            key: The key for the fact
            value: The value to store
        """
        # Convert value to string for consistent handling
        value_str = str(value)

        # Check for duplicate content before storing
        duplicate = self._check_duplicate_content(value_str)
        if duplicate:
            # If we found a duplicate, log it but don't store again
            print(f"🔄 NUCLEAR DEDUPLICATION: Content already exists as {duplicate['category']}.{duplicate['key']} = {duplicate['value']}")
            return

        # Maximum number of retries for database operations
        max_retries = 3
        retry_delay = 0.5  # Initial delay in seconds

        # No duplicate found, proceed with storage with retry logic
        for attempt in range(max_retries):
            try:
                # Add timeout parameter to prevent "database is locked" errors
                conn = sqlite3.connect(self.db_file, timeout=5.0)
                conn.execute('''
                    INSERT OR REPLACE INTO nuclear_facts (category, key, value)
                    VALUES (?, ?, ?)
                ''', (category, key, value_str))
                conn.commit()
                conn.close()
                print(f"🎯 NUCLEAR STORED: {category}.{key} = {value}")
                return
            except sqlite3.OperationalError as e:
                # Handle "database is locked" error specifically
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    print(f"⚠️ Database is locked during store_fact, retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    # Exponential backoff for retry delay
                    retry_delay *= 2
                else:
                    error_msg = str(e)
                    print(f"❌ Error storing fact: {error_msg}")
                    # Log more detailed error information
                    print(f"  - Error type: {type(e).__name__}")
                    print(f"  - Full error message: {error_msg}")
                    # Rethrow the exception to be handled by the caller
                    raise
            except Exception as e:
                error_msg = str(e)
                print(f"❌ Error storing fact: {error_msg}")
                # Log more detailed error information
                print(f"  - Error type: {type(e).__name__}")
                print(f"  - Full error message: {error_msg}")
                # Rethrow the exception to be handled by the caller
                raise

    def _check_duplicate_content(self, value):
        """
        Check if the same or very similar content already exists in the database.

        Args:
            value: The value to check for duplicates

        Returns:
            dict: Information about the duplicate if found, None otherwise
        """
        # Skip very short values as they're likely to have coincidental matches
        if len(value) < 10:
            return None

        # Maximum number of retries for database operations
        max_retries = 3
        retry_delay = 0.5  # Initial delay in seconds

        for attempt in range(max_retries):
            try:
                # Add timeout parameter to prevent "database is locked" errors
                conn = sqlite3.connect(self.db_file, timeout=5.0)

                # First try exact match
                cursor = conn.execute('''
                    SELECT category, key, value FROM nuclear_facts
                    WHERE value = ?
                    LIMIT 1
                ''', (value,))

                result = cursor.fetchone()
                if result:
                    conn.close()
                    return {
                        'category': result[0],
                        'key': result[1],
                        'value': result[2]
                    }

                # If no exact match, try fuzzy matching for longer content
                if len(value) > 50:
                    # Extract significant words (longer than 4 chars) for matching
                    significant_words = [word for word in value.split() if len(word) > 4]

                    # Limit the number of significant words to avoid "Expression tree is too large" error
                    # SQLite has a limit on expression tree depth (default 1000)
                    # According to issue reports, we need to limit words to avoid hitting 1240
                    if len(significant_words) > 20:
                        # Keep only the first 20 significant words to avoid exceeding SQLite's limit
                        print(f"⚠️ Limiting significant words from {len(significant_words)} to 20 to avoid SQLite expression tree depth limit")
                        significant_words = significant_words[:20]

                    # If we have enough significant words, try matching on those
                    if len(significant_words) >= 3:
                        # Build a query that matches if at least 80% of significant words are present
                        min_matches = max(3, int(len(significant_words) * 0.8))
                        query_parts = []
                        params = []

                        for word in significant_words:
                            query_parts.append("value LIKE ?")
                            params.append(f'%{word}%')

                        # Only return results if we have at least min_matches matches
                        # Use a different approach to avoid duplicating parameters
                        # Instead of using CASE expressions with parameters, use a fixed list of column indices
                        # This reduces the number of parameters needed by half
                        match_conditions = []
                        for i, word in enumerate(significant_words):
                            match_conditions.append(f"(CASE WHEN value LIKE ? THEN 1 ELSE 0 END)")

                        query = f'''
                            SELECT category, key, value,
                            {" + ".join(match_conditions)} as match_count
                            FROM nuclear_facts
                            GROUP BY category, key
                            HAVING match_count >= ?
                            ORDER BY match_count DESC
                            LIMIT 1
                        '''

                        # Add only the parameters for the LIKE conditions and the min_matches
                        # No need to duplicate the parameters
                        all_params = params + [min_matches]

                        cursor = conn.execute(query, all_params)
                        result = cursor.fetchone()

                        if result:
                            conn.close()
                            return {
                                'category': result[0],
                                'key': result[1],
                                'value': result[2]
                            }

                conn.close()
                return None

            except sqlite3.OperationalError as e:
                # Handle "database is locked" error specifically
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    print(f"⚠️ Database is locked, retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    # Exponential backoff for retry delay
                    retry_delay *= 2
                else:
                    error_msg = str(e)
                    print(f"⚠️ Error checking for duplicate content: {error_msg}")
                    # Log more detailed error information
                    print(f"  - Error type: {type(e).__name__}")
                    print(f"  - Full error message: {error_msg}")
                    return None
            except Exception as e:
                error_msg = str(e)
                print(f"⚠️ Error checking for duplicate content: {error_msg}")
                # Log more detailed error information
                print(f"  - Error type: {type(e).__name__}")
                print(f"  - Full error message: {error_msg}")
                return None

    def remove_fact(self, category, key):
        """Remove a fact from nuclear memory by category and key"""
        conn = sqlite3.connect(self.db_file)
        conn.execute('''
            DELETE FROM nuclear_facts
            WHERE category = ? AND key = ?
        ''', (category, key))
        rows_affected = conn.total_changes
        conn.commit()
        conn.close()
        if rows_affected > 0:
            print(f"🗑️ NUCLEAR REMOVED: {category}.{key}")
            return True
        else:
            print(f"⚠️ NUCLEAR REMOVE FAILED: No fact found with {category}.{key}")
            return False

    def search_facts_by_value(self, value_pattern):
        """Search for facts by value pattern with enhanced caching"""
        # Use cache if available and recent (increased from 5 seconds to 2 minutes)
        cache_key = f"value_{value_pattern}"
        current_time = time.time()

        if cache_key in self._search_cache:
            cache_entry = self._search_cache[cache_key]
            # Only use cache if it's recent (120 seconds instead of 5)
            if current_time - cache_entry['time'] < 120:
                # Update access statistics
                if 'access_count' not in cache_entry:
                    cache_entry['access_count'] = 1
                    cache_entry['last_access'] = current_time
                else:
                    cache_entry['access_count'] += 1
                    cache_entry['last_access'] = current_time

                # Update cache entry without changing the facts
                self._search_cache[cache_key] = cache_entry

                # Track cache hits
                self.cache_hits += 1

                # Only log if we haven't logged this pattern recently (within 30 seconds)
                # or if it's been accessed multiple times (every 10th access)
                should_log = False
                if cache_key not in self._last_logged_patterns or current_time - self._last_logged_patterns[cache_key] > 30:
                    should_log = True
                elif 'access_count' in cache_entry and cache_entry['access_count'] % 10 == 0:
                    should_log = True

                if should_log:
                    access_count = cache_entry.get('access_count', 1)
                    print(f"🔍 SEARCH CACHE HIT: Found {len(cache_entry['facts'])} facts matching '{value_pattern}' (accessed {access_count} times)")
                    self._last_logged_patterns[cache_key] = current_time

                return cache_entry['facts']

        # Cache miss - track it
        self.cache_misses += 1

        # Tokenize the value pattern for more efficient search if it contains multiple words
        tokens = [token.strip() for token in value_pattern.split() if len(token.strip()) > 2]

        conn = sqlite3.connect(self.db_file)

        if len(tokens) > 1:
            # Use tokenized search for multi-word patterns
            query_parts = []
            params = []

            for token in tokens:
                query_parts.append("value LIKE ?")
                params.append(f'%{token}%')

            query = f'''
                SELECT category, key, value FROM nuclear_facts
                WHERE {" AND ".join(query_parts)}
                ORDER BY timestamp DESC
            '''
            cursor = conn.execute(query, params)
        else:
            # Use simple search for single word or original pattern
            cursor = conn.execute('''
                SELECT category, key, value FROM nuclear_facts
                WHERE value LIKE ?
                ORDER BY timestamp DESC
            ''', (f'%{value_pattern}%',))

        results = cursor.fetchall()
        conn.close()

        facts = []
        for row in results:
            facts.append({
                'category': row[0],
                'key': row[1],
                'value': row[2]
            })

        # Store in cache with enhanced metadata
        self._search_cache[cache_key] = {
            'facts': facts, 
            'time': current_time,
            'access_count': 1,
            'last_access': current_time,
            'value_pattern': value_pattern
        }

        # Implement a more sophisticated cache eviction policy
        # based on both recency and frequency (LFU/LRU hybrid)
        if len(self._search_cache) > 100:  # Increased from 50 to 100
            # Calculate a score for each cache entry based on:
            # 1. How recently it was accessed (higher is better)
            # 2. How frequently it's accessed (higher is better)
            # 3. How many facts it contains (higher is better)
            scored_entries = []
            for k, entry in self._search_cache.items():
                # Recency score: 0-1 based on how recently it was accessed
                last_access = entry.get('last_access', entry['time'])
                recency = max(0, min(1, (current_time - last_access) / 300))

                # Frequency score: log scale to prevent very frequent items from dominating
                access_count = entry.get('access_count', 1)
                frequency = min(10, math.log2(access_count + 1))

                # Value score: more facts = more valuable
                value = min(5, len(entry['facts']))

                # Combined score (lower is better for eviction)
                score = (1 - recency) * 0.5 + frequency * 0.3 + value * 0.2

                scored_entries.append((k, score))

            # Sort by score (ascending) and remove the 20 lowest scoring entries
            scored_entries.sort(key=lambda x: x[1])
            for k, _ in scored_entries[:20]:  # Increased from 10 to 20
                del self._search_cache[k]

            print(f"🧠 SEARCH CACHE MANAGEMENT: Evicted 20 least valuable entries, keeping {len(self._search_cache)}")

        # Only log if we haven't logged this pattern recently (within 30 seconds)
        if cache_key not in self._last_logged_patterns or current_time - self._last_logged_patterns[cache_key] > 30:
            print(f"🔍 NUCLEAR SEARCH: Found {len(facts)} facts matching '{value_pattern}'")
            self._last_logged_patterns[cache_key] = current_time

        # Pre-cache related searches if this is an important query with results
        if len(facts) > 2 and len(value_pattern.split()) > 1:
            # This is an important query with results, pre-cache related queries
            threading.Thread(target=self._precache_related_searches, args=(value_pattern, facts), daemon=True).start()

        return facts

    def _get_filtered_facts_from_cache(self, cache_entry, keywords, is_ai_identity_query):
        """
        Get filtered facts from cache entry, handling AI identity queries specially.

        Args:
            cache_entry: The cache entry containing facts
            keywords: The keywords used for the query
            is_ai_identity_query: Whether this is an AI identity query

        Returns:
            List of facts or None if no facts are left after filtering
        """
        # For AI identity queries, filter out any web_knowledge facts from cached results
        if is_ai_identity_query:
            original_count = len(cache_entry['facts'])
            # Filter out web_knowledge facts and learning_attempts.failed_search
            filtered_facts = [fact for fact in cache_entry['facts'] if not fact.startswith('web_knowledge.') and not fact.startswith('learning_attempts.failed_search')]
            filtered_count = len(filtered_facts)

            if original_count != filtered_count:
                print(f"🎯 NUCLEAR AI IDENTITY: Filtered out {original_count - filtered_count} web_knowledge facts from cached results")

            # If we filtered out all facts from cache, don't use cache and get fresh results
            if len(filtered_facts) == 0:
                print(f"🎯 NUCLEAR AI IDENTITY: No facts left in cache after filtering, getting fresh results")
                return None

            # Prioritize AI identity facts in the filtered results
            ai_identity_facts = [fact for fact in filtered_facts if 
                                fact.startswith('ai_identity.') or 
                                'doba' in fact.lower() or 
                                ('name' in fact.lower() and ('ai' in fact.lower() or 'assistant' in fact.lower()))]

            if ai_identity_facts:
                print(f"🎯 NUCLEAR AI IDENTITY: Found {len(ai_identity_facts)} AI identity facts in cache")
                # Combine the facts, putting AI identity facts first
                # Remove any duplicates that might exist in both lists
                facts_set = set(ai_identity_facts)
                prioritized_facts = ai_identity_facts + [f for f in filtered_facts if f not in facts_set]
                return prioritized_facts
            else:
                print(f"🎯 NUCLEAR AI IDENTITY: No AI identity facts found in cache")
                # For AI identity queries, if no AI identity facts are found in cache,
                # force a fresh query to try to get AI identity facts directly
                return None
        else:
            return cache_entry['facts']

    def recall_facts(self, keywords):
        """Recall facts from nuclear memory with improved efficiency and precision"""
        # If no keywords, return empty list to avoid unnecessary processing
        if not keywords:
            print("🎯 NUCLEAR RECALL: No keywords provided")
            return []

        # Check if this is an AI identity query
        is_ai_identity_query = any(kw in ["type", "name", "identity", "ai", "DoBA"] for kw in keywords)

        if is_ai_identity_query:
            print(f"🎯 NUCLEAR AI IDENTITY: Filtering out web_knowledge facts for AI identity query with keywords {keywords}")

        # Create a cache key by sorting and joining the keywords
        # Include the is_ai_identity_query flag in the cache key to avoid incorrect cached results
        cache_key = "_".join(sorted(keywords)) + f"_ai_identity_{is_ai_identity_query}"

        # Check if we have a cached result for these keywords
        cached_facts = None
        if cache_key in self._facts_cache:
            cache_entry = self._facts_cache[cache_key]
            current_time = time.time()

            # Use cache if it's recent (less than 5 minutes old)
            if current_time - cache_entry['time'] < 300:  # Increased from 30 to 300 seconds
                # Update access statistics
                cache_entry['access_count'] += 1
                cache_entry['last_access'] = current_time

                # Update cache entry without changing the facts
                self._facts_cache[cache_key] = cache_entry

                # Track cache hits
                self.cache_hits += 1

                # Get cached facts (may be None if filtered out)
                cached_facts = self._get_filtered_facts_from_cache(cache_entry, keywords, is_ai_identity_query)

                # Only log every 10th access to reduce console spam
                if cache_entry['access_count'] % 10 == 1:
                    print(f"🎯 FACTS CACHE HIT: Using cached facts for {keywords} (accessed {cache_entry['access_count']} times)")

                # If we have valid cached facts, return them
                if cached_facts is not None:
                    return cached_facts

                # If cached_facts is None, we need to remove this entry from cache to force a fresh query
                if is_ai_identity_query and cache_key in self._facts_cache:
                    del self._facts_cache[cache_key]

        # Cache miss - track it
        self.cache_misses += 1

        # Use a shorter connection timeout to prevent hanging
        conn = sqlite3.connect(self.db_file, timeout=2.0)

        # Create index on category and key if it doesn't exist
        try:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_nuclear_facts_category_key ON nuclear_facts(category, key)")
            conn.commit()
        except Exception as e:
            print(f"Warning: Could not create index: {e}")

        # Optimize query by prioritizing category and key fields only
        # This is much faster than searching in all fields including values

        # For better performance with multiple keywords, use separate queries and UNION
        # This avoids the combinatorial explosion of OR conditions
        if len(keywords) > 1:
            queries = []
            all_params = []

            for keyword in keywords:
                keyword_pattern = f'%{keyword}%'
                if is_ai_identity_query:
                    # For AI identity queries, exclude web_knowledge facts
                    queries.append("SELECT category, key, value, timestamp FROM nuclear_facts WHERE (category LIKE ? OR key LIKE ?) AND category != 'web_knowledge'")
                else:
                    queries.append("SELECT category, key, value, timestamp FROM nuclear_facts WHERE category LIKE ? OR key LIKE ?")
                all_params.extend([keyword_pattern, keyword_pattern])

            # Combine with UNION and add limit and order
            query = " UNION ".join(queries) + " ORDER BY timestamp DESC LIMIT 3"  # Further reduced from 5 to 3
            params = all_params
        else:
            # For single keyword, use simpler query
            conditions = []
            params = []

            # Process the single keyword
            if keywords:
                keyword = keywords[0]
                if is_ai_identity_query:
                    # For AI identity queries, exclude web_knowledge facts
                    conditions.append("((category LIKE ? OR key LIKE ?) AND category != 'web_knowledge')")
                else:
                    conditions.append("(category LIKE ? OR key LIKE ?)")
                keyword_pattern = f'%{keyword}%'
                params.extend([keyword_pattern, keyword_pattern])

                where_clause = " OR ".join(conditions)
                query = f'''
                    SELECT category, key, value, timestamp FROM nuclear_facts 
                    WHERE {where_clause}
                    ORDER BY timestamp DESC
                    LIMIT 3  -- Further reduced from 5 to 3
                '''
            else:
                query = "SELECT category, key, value, timestamp FROM nuclear_facts ORDER BY timestamp DESC LIMIT 3"
                params = []

        # Execute with a timeout to prevent long-running queries
        try:
            cursor = conn.execute(query, params)
            results = cursor.fetchall()
        except sqlite3.Error as e:
            print(f"SQLite error in recall_facts: {e}")
            results = []
        finally:
            conn.close()

        # Format facts with more aggressive truncation
        facts = []
        for row in results:
            # Further reduce max length from 150 to 100 characters
            value_text = row[2]
            if len(value_text) > 100:
                value_text = value_text[:97] + "..."
            facts.append(f"{row[0]}.{row[1]}: {value_text}")

        # For AI identity queries, filter out any web_knowledge facts and learning_attempts.failed_search
        if is_ai_identity_query:
            original_count = len(facts)
            facts = [fact for fact in facts if not fact.startswith('web_knowledge.') and not fact.startswith('learning_attempts.failed_search')]
            filtered_count = len(facts)

            if original_count != filtered_count:
                print(f"🎯 NUCLEAR AI IDENTITY: Filtered out {original_count - filtered_count} irrelevant facts from database results")

            # For AI identity queries, always try to get AI identity facts directly
            # This ensures we prioritize facts specifically about the AI's identity
            print(f"🎯 NUCLEAR AI IDENTITY: Retrieving AI identity facts directly")
            ai_identity_facts = []

            # Use a direct query to get AI identity facts
            try:
                conn = sqlite3.connect(self.db_file, timeout=2.0)
                # Improved query to better match AI identity facts
                query = """
                SELECT category, key, value, timestamp 
                FROM nuclear_facts 
                WHERE category = 'ai_identity' 
                   OR (key = 'name' AND (value LIKE '%DoBA%' OR value LIKE '%AI%' OR value LIKE '%assistant%'))
                   OR key LIKE '%identity%' 
                   OR key LIKE '%DoBA%' 
                ORDER BY 
                   CASE 
                     WHEN category = 'ai_identity' THEN 1
                     WHEN key = 'name' AND value LIKE '%DoBA%' THEN 2
                     ELSE 3
                   END,
                   timestamp DESC 
                LIMIT 5
                """
                cursor = conn.execute(query)
                results = cursor.fetchall()

                for row in results:
                    value_text = row[2]
                    if len(value_text) > 100:
                        value_text = value_text[:97] + "..."
                    ai_identity_facts.append(f"{row[0]}.{row[1]}: {value_text}")

                print(f"🎯 NUCLEAR AI IDENTITY: Retrieved {len(ai_identity_facts)} AI identity facts directly")

                # If we found AI identity facts, prioritize them
                if ai_identity_facts:
                    # Combine the facts, putting AI identity facts first
                    # Remove any duplicates that might exist in both lists
                    facts_set = set(ai_identity_facts)
                    facts = ai_identity_facts + [f for f in facts if f not in facts_set]
                    print(f"🎯 NUCLEAR AI IDENTITY: Combined {len(ai_identity_facts)} AI identity facts with {len(facts) - len(ai_identity_facts)} other facts")
                elif len(facts) == 0:
                    # If no AI identity facts were found and no other facts exist,
                    # create a default fact about the AI's identity
                    default_fact = "ai_identity.name: DoBA"
                    facts.append(default_fact)
                    print(f"🎯 NUCLEAR AI IDENTITY: No AI identity facts found, using default: {default_fact}")
            except Exception as e:
                print(f"❌ Error retrieving AI identity facts: {e}")
                # If there was an error and no facts exist, create a default fact
                if len(facts) == 0:
                    default_fact = "ai_identity.name: DoBA"
                    facts.append(default_fact)
                    print(f"🎯 NUCLEAR AI IDENTITY: Error retrieving facts, using default: {default_fact}")
            finally:
                conn.close()

        # Cache the results with enhanced metadata
        current_time = time.time()
        self._facts_cache[cache_key] = {
            'facts': facts, 
            'time': current_time,
            'access_count': 1,
            'last_access': current_time,
            'keywords': keywords
        }

        # Implement a more sophisticated cache eviction policy
        # based on both recency and frequency (LFU/LRU hybrid)
        if len(self._facts_cache) > 50:  # Increased from 30 to 50
            # Calculate a score for each cache entry based on:
            # 1. How recently it was accessed (higher is better)
            # 2. How frequently it's accessed (higher is better)
            # 3. How many facts it contains (higher is better)
            scored_entries = []
            for k, entry in self._facts_cache.items():
                # Recency score: 0-1 based on how recently it was accessed
                recency = max(0, min(1, (current_time - entry['last_access']) / 300))

                # Frequency score: log scale to prevent very frequent items from dominating
                frequency = min(10, math.log2(entry['access_count'] + 1))

                # Value score: more facts = more valuable
                value = min(5, len(entry['facts']))

                # Combined score (lower is better for eviction)
                score = (1 - recency) * 0.5 + frequency * 0.3 + value * 0.2

                scored_entries.append((k, score))

            # Sort by score (ascending) and remove the 10 lowest scoring entries
            scored_entries.sort(key=lambda x: x[1])
            for k, _ in scored_entries[:10]:  # Increased from 5 to 10
                del self._facts_cache[k]

            print(f"🧠 CACHE MANAGEMENT: Evicted 10 least valuable entries, keeping {len(self._facts_cache)}")

        # Pre-cache related facts if this is an important query
        if len(facts) > 0 and len(keywords) > 1:
            # This is an important query with results, pre-cache related queries
            threading.Thread(target=self._precache_related_facts, args=(keywords, facts), daemon=True).start()

        # Log cache statistics periodically
        if (self.cache_hits + self.cache_misses) % 100 == 0:
            hit_rate = self.cache_hits / max(1, (self.cache_hits + self.cache_misses)) * 100
            print(f"🧠 CACHE STATS: Hit rate: {hit_rate:.1f}% ({self.cache_hits} hits, {self.cache_misses} misses)")

        print(f"🎯 NUCLEAR RECALLED: {len(facts)} facts for {keywords}")
        return facts

    def _precache_related_searches(self, value_pattern, original_facts):
        """
        Pre-cache related searches in the background to improve response times for future queries.
        This method is called in a separate thread to avoid blocking the main thread.

        Args:
            value_pattern: The original value pattern used for the search
            original_facts: The facts returned by the original search
        """
        try:
            # Don't log this operation to reduce console spam
            # print(f"🧠 PRECACHING SEARCHES: Generating related patterns for '{value_pattern}'")

            # Generate related search patterns
            related_patterns = []

            # 1. Split the value pattern into words
            words = [w.strip() for w in value_pattern.split() if len(w.strip()) > 2]

            # 2. Generate single-word patterns
            if len(words) > 1:
                for word in words:
                    related_patterns.append(word)

            # 3. Generate combinations of words
            if len(words) > 2:
                for i in range(len(words) - 1):
                    related_patterns.append(f"{words[i]} {words[i+1]}")

            # 4. Extract categories and keys from the original facts
            categories = set()
            keys = set()
            for fact in original_facts:
                if 'category' in fact and 'key' in fact:
                    categories.add(fact['category'])
                    keys.add(fact['key'])

            # 5. Generate patterns with categories and keys
            for word in words:
                for category in categories:
                    related_patterns.append(f"{word} {category}")
                for key in keys:
                    related_patterns.append(f"{word} {key}")

            # Limit the number of related patterns to avoid excessive database load
            if len(related_patterns) > 5:
                # Randomly select 5 patterns
                related_patterns = random.sample(related_patterns, 5)

            # Process each related pattern
            for pattern in related_patterns:
                # Skip if identical to the original pattern
                if pattern == value_pattern:
                    continue

                # Create cache key
                cache_key = f"value_{pattern}"

                # Skip if already in cache
                if cache_key in self._search_cache:
                    continue

                # Fetch facts for this related pattern
                try:
                    # Use a shorter timeout for background queries
                    conn = sqlite3.connect(self.db_file, timeout=1.0)

                    # Simplified query for background processing
                    cursor = conn.execute('''
                        SELECT category, key, value FROM nuclear_facts
                        WHERE value LIKE ?
                        LIMIT 3
                    ''', (f'%{pattern}%',))

                    results = cursor.fetchall()
                    conn.close()

                    # Format facts
                    related_facts = []
                    for row in results:
                        related_facts.append({
                            'category': row[0],
                            'key': row[1],
                            'value': row[2]
                        })

                    # Only cache if we found facts
                    if related_facts:
                        current_time = time.time()
                        self._search_cache[cache_key] = {
                            'facts': related_facts,
                            'time': current_time,
                            'access_count': 0,  # Start at 0 since it hasn't been accessed yet
                            'last_access': current_time,
                            'value_pattern': pattern,
                            'precached': True  # Mark as precached
                        }
                        # print(f"🧠 PRECACHED SEARCH: {len(related_facts)} facts for '{pattern}'")

                except Exception as e:
                    # Silently ignore errors in background processing
                    pass

        except Exception as e:
            # Silently ignore errors in background processing
            pass

    def _precache_related_facts(self, keywords, original_facts):
        """
        Pre-cache related facts in the background to improve response times for future queries.
        This method is called in a separate thread to avoid blocking the main thread.

        Args:
            keywords: The original keywords used for the query
            original_facts: The facts returned by the original query
        """
        try:
            # Don't log this operation to reduce console spam
            # print(f"🧠 PRECACHING: Generating related queries for {keywords}")

            # Extract categories and keys from the original facts
            categories = set()
            keys = set()
            for fact in original_facts:
                if "." in fact:
                    parts = fact.split(".", 1)
                    if len(parts) == 2:
                        categories.add(parts[0])
                        key_parts = parts[1].split(":", 1)
                        if len(key_parts) > 0:
                            keys.add(key_parts[0])

            # Generate related keyword combinations
            related_queries = []

            # 1. Single keywords from the original query
            if len(keywords) > 1:
                for keyword in keywords:
                    related_queries.append([keyword])

            # 2. Keywords + categories
            for category in categories:
                for keyword in keywords:
                    related_queries.append([keyword, category])

            # 3. Keywords + keys
            for key in keys:
                for keyword in keywords:
                    related_queries.append([keyword, key])

            # Limit the number of related queries to avoid excessive database load
            if len(related_queries) > 5:
                # Randomly select 5 queries
                related_queries = random.sample(related_queries, 5)

            # Process each related query
            for related_keywords in related_queries:
                # Skip if identical to the original query
                if set(related_keywords) == set(keywords):
                    continue

                # Create cache key
                cache_key = "_".join(sorted(related_keywords))

                # Skip if already in cache
                if cache_key in self._facts_cache:
                    continue

                # Fetch facts for this related query
                try:
                    # Use a shorter timeout for background queries
                    conn = sqlite3.connect(self.db_file, timeout=1.0)

                    # Simplified query for background processing
                    query_parts = []
                    params = []

                    for keyword in related_keywords:
                        query_parts.append("(category LIKE ? OR key LIKE ?)")
                        keyword_pattern = f'%{keyword}%'
                        params.extend([keyword_pattern, keyword_pattern])

                    where_clause = " OR ".join(query_parts)
                    query = f'''
                        SELECT category, key, value FROM nuclear_facts 
                        WHERE {where_clause}
                        LIMIT 3
                    '''

                    cursor = conn.execute(query, params)
                    results = cursor.fetchall()
                    conn.close()

                    # Format facts
                    related_facts = []
                    for row in results:
                        value_text = row[2]
                        if len(value_text) > 100:
                            value_text = value_text[:97] + "..."
                        related_facts.append(f"{row[0]}.{row[1]}: {value_text}")

                    # Only cache if we found facts
                    if related_facts:
                        current_time = time.time()
                        self._facts_cache[cache_key] = {
                            'facts': related_facts,
                            'time': current_time,
                            'access_count': 0,  # Start at 0 since it hasn't been accessed yet
                            'last_access': current_time,
                            'keywords': related_keywords,
                            'precached': True  # Mark as precached
                        }
                        # print(f"🧠 PRECACHED: {len(related_facts)} facts for {related_keywords}")

                except Exception as e:
                    # Silently ignore errors in background processing
                    pass

        except Exception as e:
            # Silently ignore errors in background processing
            pass

    def store_conversation(self, user_message, ai_response="", session_id="default", extracted_facts=""):
        """Store conversation in nuclear memory and autonomously extract valuable facts"""
        # Maximum number of retries for database operations
        max_retries = 3
        retry_delay = 0.5  # Initial delay in seconds

        # If no extracted facts were provided, try to autonomously extract them
        if not extracted_facts and user_message and ai_response:
            extracted_facts = self.autonomously_extract_facts(user_message, ai_response)

        # Try to store the conversation with retry logic
        for attempt in range(max_retries):
            try:
                # Add timeout parameter to prevent "database is locked" errors
                conn = sqlite3.connect(self.db_file, timeout=5.0)

                # Check if extracted_facts column exists
                try:
                    conn.execute("SELECT extracted_facts FROM memory_interactions LIMIT 1")
                    has_extracted_facts = True
                except sqlite3.OperationalError:
                    has_extracted_facts = False
                    print("⚠️ extracted_facts column not found, adding it to the table")
                    try:
                        conn.execute("ALTER TABLE memory_interactions ADD COLUMN extracted_facts TEXT")
                        has_extracted_facts = True
                        print("✅ Added extracted_facts column to memory_interactions table")
                    except Exception as e:
                        print(f"❌ Failed to add extracted_facts column: {e}")

                # Insert with or without extracted_facts based on column existence
                if has_extracted_facts:
                    conn.execute("INSERT INTO memory_interactions (user_message, ai_response, session_id, extracted_facts) VALUES (?, ?, ?, ?)",
                                (user_message, ai_response, session_id, extracted_facts))
                else:
                    conn.execute("INSERT INTO memory_interactions (user_message, ai_response, session_id) VALUES (?, ?, ?)",
                                (user_message, ai_response, session_id))

                conn.commit()
                conn.close()

                # If facts were extracted, store them in the cache for quick retrieval
                if extracted_facts:
                    self._cache_extracted_facts(user_message, extracted_facts)
                    print(f"🎯 NUCLEAR CONVERSATION STORED: {user_message[:50]}... with {len(extracted_facts.split('|'))} extracted facts")
                else:
                    print(f"🎯 NUCLEAR CONVERSATION STORED: {user_message[:50]}...")

                # Successfully stored the conversation, so return
                return

            except sqlite3.OperationalError as e:
                # Handle "database is locked" error specifically
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    print(f"⚠️ Database is locked during store_conversation, retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    # Exponential backoff for retry delay
                    retry_delay *= 2
                else:
                    error_msg = str(e)
                    print(f"❌ Error storing conversation: {error_msg}")
                    # Log more detailed error information
                    print(f"  - Error type: {type(e).__name__}")
                    print(f"  - Full error message: {error_msg}")
                    # Don't rethrow the exception, just log it
            except Exception as e:
                error_msg = str(e)
                print(f"❌ Error storing conversation: {error_msg}")
                # Log more detailed error information
                print(f"  - Error type: {type(e).__name__}")
                print(f"  - Full error message: {error_msg}")
                # Don't rethrow the exception, just log it
                break

    def autonomously_extract_facts(self, user_message, ai_response):
        """
        Autonomously extract valuable facts from a conversation without explicit prompting.

        This method analyzes the conversation to identify valuable information and extracts
        facts based on the content. It uses a set of heuristics to determine what information
        is worth remembering.

        Args:
            user_message: The user's message
            ai_response: The AI's response

        Returns:
            str: Pipe-separated list of extracted facts, or empty string if no facts were extracted
        """
        # Skip extraction for very short messages or questions
        if not user_message or not ai_response or len(user_message.split()) < 5:
            return ""

        # Check if we have a cached result for this conversation
        cache_key = f"facts_{hash(user_message + ai_response)}"
        if hasattr(self, '_facts_extraction_cache') and cache_key in self._facts_extraction_cache:
            return self._facts_extraction_cache[cache_key]

        # Initialize facts extraction cache if needed
        if not hasattr(self, '_facts_extraction_cache'):
            self._facts_extraction_cache = {}

        extracted_facts = []

        try:
            # Combine messages for analysis
            combined_text = f"{user_message}\n{ai_response}"

            # 1. Look for personal information patterns
            personal_info_patterns = [
                (r"(?:my|your) name is (\w+)", "name"),
                (r"(?:I am|I'm) (\d+) years old", "age"),
                (r"(?:I|you) live in ([A-Za-z\s,]+)", "location"),
                (r"(?:my|your) email is ([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", "email"),
                (r"(?:my|your) phone (?:number|is) ([0-9\-\(\)\s]+)", "phone"),
                (r"(?:I|you) work(?:ed)? (?:at|for) ([A-Za-z\s&]+)", "employer"),
                (r"(?:my|your) job is (?:a |an )?([A-Za-z\s]+)", "profession"),
                (r"(?:I|you) (?:like|enjoy|love) ([A-Za-z\s,]+)", "preference"),
                (r"(?:my|your) favorite ([A-Za-z\s]+) is ([A-Za-z\s,]+)", "favorite"),
                (r"(?:I|you) (?:have|had) ([A-Za-z\s]+)", "possession")
            ]

            for pattern, category in personal_info_patterns:
                matches = re.findall(pattern, combined_text, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        # If the match is a tuple (multiple capture groups), use the first two groups
                        subcategory, value = match[0], match[1]
                        fact = f"{category}_{subcategory}:{value}"
                    else:
                        # If the match is a string (single capture group), use it directly
                        fact = f"{category}:{match}"

                    # Only add if not already present
                    if fact not in extracted_facts:
                        extracted_facts.append(fact)

            # 2. Look for factual statements in the AI's response
            factual_patterns = [
                r"([A-Z][a-z]+ is [a-zA-Z0-9\s]+)",
                r"([A-Z][a-z]+ was [a-zA-Z0-9\s]+)",
                r"([A-Z][a-z]+ has [a-zA-Z0-9\s]+)",
                r"([A-Z][a-z]+ are [a-zA-Z0-9\s]+)",
                r"([A-Z][a-z]+ were [a-zA-Z0-9\s]+)",
                r"([A-Z][a-z]+ have [a-zA-Z0-9\s]+)"
            ]

            for pattern in factual_patterns:
                matches = re.findall(pattern, ai_response)
                for match in matches:
                    # Categorize factual statements
                    if " is " in match:
                        parts = match.split(" is ", 1)
                        if len(parts) == 2:
                            subject, predicate = parts
                            fact = f"fact_{subject.lower()}:{predicate}"
                            if fact not in extracted_facts:
                                extracted_facts.append(fact)
                    elif " was " in match or " were " in match or " has " in match or " have " in match or " are " in match:
                        # Similar processing for other verb patterns
                        fact = f"fact:{match}"
                        if fact not in extracted_facts:
                            extracted_facts.append(fact)

            # 3. Look for preferences and opinions in the user's message
            preference_patterns = [
                r"I (?:like|love|enjoy|prefer) ([a-zA-Z0-9\s]+)",
                r"I (?:dislike|hate|don't like) ([a-zA-Z0-9\s]+)",
                r"My favorite ([a-zA-Z0-9\s]+) is ([a-zA-Z0-9\s]+)"
            ]

            for pattern in preference_patterns:
                matches = re.findall(pattern, user_message, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        # If the match is a tuple (multiple capture groups), use the first two groups
                        category, value = match
                        fact = f"preference_{category}:{value}"
                    else:
                        # If the match is a string (single capture group), use it directly
                        if "like" in pattern or "love" in pattern or "enjoy" in pattern or "prefer" in pattern:
                            fact = f"preference_like:{match}"
                        else:
                            fact = f"preference_dislike:{match}"

                    # Only add if not already present
                    if fact not in extracted_facts:
                        extracted_facts.append(fact)

            # Limit the number of extracted facts to avoid excessive storage
            if len(extracted_facts) > 10:
                # Keep only the most specific facts (longer ones tend to be more specific)
                extracted_facts.sort(key=len, reverse=True)
                extracted_facts = extracted_facts[:10]

            # Join facts with pipe separator
            result = "|".join(extracted_facts)

            # Cache the result
            self._facts_extraction_cache[cache_key] = result

            # Limit cache size
            if len(self._facts_extraction_cache) > 100:
                # Remove oldest entries (simple FIFO for this cache)
                keys_to_remove = list(self._facts_extraction_cache.keys())[:-50]  # Keep the 50 most recent
                for key in keys_to_remove:
                    del self._facts_extraction_cache[key]

            # Store extracted facts in nuclear memory
            if extracted_facts:
                # Generate a unique key for this conversation
                conversation_key = f"conv_{int(time.time())}_{hash(user_message) % 10000}"

                # Store each fact with the conversation as the category
                for i, fact in enumerate(extracted_facts):
                    if ":" in fact:
                        fact_parts = fact.split(":", 1)
                        if len(fact_parts) == 2:
                            category, value = fact_parts
                            self.store_fact(category, f"{conversation_key}_{i}", value)

            return result

        except Exception as e:
            print(f"Error extracting facts: {e}")
            return ""

    def _cache_extracted_facts(self, query, extracted_facts):
        """
        Cache extracted facts for quick retrieval based on the query.

        Args:
            query: The query that generated the facts
            extracted_facts: The extracted facts as a pipe-separated string
        """
        if not extracted_facts:
            return

        # Split the query into words for keyword extraction
        words = query.lower().split()

        # Extract keywords (non-stopwords)
        stop_words = {"what", "is", "the", "and", "for", "you", "your", "my", "me", "i", "are", "to", "of", "in", "it"}
        keywords = []

        for word in words:
            # Clean the word
            word = word.strip('?.,!:;()"\'')
            # Only consider words that are meaningful
            if len(word) > 3 and word not in stop_words:
                keywords.append(word)
                # Limit to 3 keywords for efficiency
                if len(keywords) >= 3:
                    break

        # If we have keywords and facts, cache them
        if keywords and extracted_facts:
            # Create cache key
            cache_key = "_".join(sorted(keywords))

            # Parse the extracted facts
            facts = []
            for fact in extracted_facts.split("|"):
                if ":" in fact:
                    category, value = fact.split(":", 1)
                    facts.append(f"{category}:{value}")

            # Only cache if we have facts
            if facts:
                current_time = time.time()

                # Check if we already have this cache key
                if cache_key in self._facts_cache:
                    # Update existing cache entry
                    cache_entry = self._facts_cache[cache_key]

                    # Add new facts if they're not already present
                    existing_facts = set(cache_entry['facts'])
                    for fact in facts:
                        if fact not in existing_facts:
                            cache_entry['facts'].append(fact)

                    # Update metadata
                    cache_entry['time'] = current_time
                    cache_entry['last_access'] = current_time
                    cache_entry['access_count'] += 1

                    # Update cache
                    self._facts_cache[cache_key] = cache_entry
                else:
                    # Create new cache entry
                    self._facts_cache[cache_key] = {
                        'facts': facts,
                        'time': current_time,
                        'access_count': 1,
                        'last_access': current_time,
                        'keywords': keywords,
                        'extracted': True  # Mark as extracted from conversation
                    }

    def _start_autonomous_optimization(self):
        """Start a background thread for autonomous optimization of caches and memory usage"""
        try:
            # Start a daemon thread that will run in the background
            optimization_thread = threading.Thread(
                target=self._autonomous_optimization_thread,
                daemon=True
            )
            optimization_thread.start()
            print("🧠 AUTONOMOUS OPTIMIZATION: Started background thread for cache optimization")
        except Exception as e:
            print(f"⚠️ Failed to start autonomous optimization thread: {e}")

    def _autonomous_optimization_thread(self):
        """
        Background thread that periodically optimizes caches and memory usage.
        This runs autonomously without user prompting.
        """
        try:
            # Initial delay to allow system to initialize
            time.sleep(60)

            while True:
                try:
                    current_time = time.time()

                    # Only optimize every 5 minutes (reduced from 10 minutes)
                    if current_time - self.last_optimization_time > 300:
                        print("🧠 AUTONOMOUS OPTIMIZATION: Running cache optimization")

                        # 1. Consolidate similar facts in the facts cache
                        self._consolidate_facts_cache()

                        # 2. Identify and remove redundant entries in the search cache
                        self._optimize_search_cache()

                        # 3. Clean up extraction cache
                        self._cleanup_extraction_cache()

                        # 4. Deduplicate database facts (every 4th run, approximately every 20 minutes)
                        # This is more expensive so we don't run it every time
                        if not hasattr(self, '_db_dedup_counter'):
                            self._db_dedup_counter = 0
                        self._db_dedup_counter += 1

                        if self._db_dedup_counter >= 4:
                            self._deduplicate_database_facts()
                            self._db_dedup_counter = 0

                        # 5. Log optimization statistics
                        total_cache_entries = (
                            len(self._facts_cache) +
                            len(self._search_cache) +
                            len(self._conversation_cache) +
                            len(self._facts_extraction_cache)
                        )

                        hit_rate = self.cache_hits / max(1, (self.cache_hits + self.cache_misses)) * 100
                        print(f"🧠 CACHE OPTIMIZATION COMPLETE: {total_cache_entries} total entries, {hit_rate:.1f}% hit rate")

                        # Update last optimization time
                        self.last_optimization_time = current_time

                    # Sleep for 2 minutes before checking again (reduced from 5 minutes)
                    time.sleep(120)

                except Exception as e:
                    print(f"⚠️ Error in autonomous optimization cycle: {e}")
                    # Sleep for 5 minutes before trying again
                    time.sleep(300)

        except Exception as e:
            print(f"⚠️ Autonomous optimization thread terminated: {e}")

    def _consolidate_facts_cache(self):
        """Consolidate similar facts in the facts cache to reduce redundancy"""
        try:
            # Group cache entries by similar keywords
            keyword_groups = {}
            content_groups = {}  # New: group by content similarity

            # First pass: group by keyword similarity and content
            for cache_key, entry in list(self._facts_cache.items()):
                # Group by keywords if available
                if 'keywords' in entry:
                    # Create a frozenset of keywords for grouping
                    keyword_set = frozenset(entry['keywords'])

                    if keyword_set not in keyword_groups:
                        keyword_groups[keyword_set] = []

                    keyword_groups[keyword_set].append((cache_key, entry))

                # Also group by content similarity for facts
                if 'facts' in entry and entry['facts']:
                    # Create content signatures for each fact
                    for fact in entry['facts']:
                        if ':' in fact:
                            # Extract the value part of the fact
                            fact_parts = fact.split(':', 1)
                            if len(fact_parts) == 2:
                                fact_value = fact_parts[1].strip()

                                # Skip very short values
                                if len(fact_value) < 10:
                                    continue

                                # Create a content signature from significant words
                                significant_words = sorted([word.lower() for word in fact_value.split() if len(word) > 4])

                                if len(significant_words) >= 2:
                                    # Use the first 5 significant words as a signature
                                    content_sig = tuple(significant_words[:5])

                                    if content_sig not in content_groups:
                                        content_groups[content_sig] = []

                                    # Store the entry and the specific fact
                                    content_groups[content_sig].append((cache_key, entry, fact))

            # Second pass: consolidate entries within each keyword group
            consolidated_count = 0

            for keyword_set, entries in keyword_groups.items():
                if len(entries) > 1:
                    # Sort entries by access count (descending) and recency (descending)
                    entries.sort(key=lambda x: (
                        x[1].get('access_count', 0),
                        x[1].get('last_access', 0)
                    ), reverse=True)

                    # Keep the most valuable entry
                    primary_key, primary_entry = entries[0]

                    # Consolidate facts from other entries into the primary entry
                    primary_facts = set(primary_entry['facts'])

                    for secondary_key, secondary_entry in entries[1:]:
                        # Add unique facts from secondary entry to primary entry
                        for fact in secondary_entry['facts']:
                            if fact not in primary_facts:
                                primary_entry['facts'].append(fact)
                                primary_facts.add(fact)

                        # Remove the secondary entry
                        if secondary_key in self._facts_cache:
                            del self._facts_cache[secondary_key]
                            consolidated_count += 1

            # Third pass: consolidate facts with similar content across different entries
            content_consolidated = 0

            for content_sig, fact_entries in content_groups.items():
                if len(fact_entries) > 1:
                    # Group by cache key to avoid processing the same entry multiple times
                    entry_facts = {}

                    for cache_key, entry, fact in fact_entries:
                        if cache_key not in entry_facts:
                            entry_facts[cache_key] = []
                        entry_facts[cache_key].append(fact)

                    # Skip if all facts are from the same entry
                    if len(entry_facts) <= 1:
                        continue

                    # Sort entries by access count and recency
                    sorted_entries = sorted(
                        [(k, self._facts_cache[k]) for k in entry_facts.keys() if k in self._facts_cache],
                        key=lambda x: (
                            x[1].get('access_count', 0),
                            x[1].get('last_access', 0)
                        ),
                        reverse=True
                    )

                    if not sorted_entries:
                        continue

                    # Keep the most valuable entry
                    primary_key, primary_entry = sorted_entries[0]
                    primary_facts = set(primary_entry['facts'])

                    # For each other entry, move unique facts to the primary entry
                    for secondary_key, _ in sorted_entries[1:]:
                        if secondary_key in entry_facts and secondary_key in self._facts_cache:
                            for fact in entry_facts[secondary_key]:
                                if fact not in primary_facts:
                                    primary_entry['facts'].append(fact)
                                    primary_facts.add(fact)
                                    content_consolidated += 1
                                else:
                                    # Remove duplicate facts from secondary entries
                                    secondary_entry = self._facts_cache[secondary_key]
                                    if fact in secondary_entry['facts']:
                                        secondary_entry['facts'].remove(fact)
                                        content_consolidated += 1

            total_consolidated = consolidated_count + content_consolidated
            if total_consolidated > 0:
                print(f"🧠 CACHE CONSOLIDATION: Consolidated {consolidated_count} redundant entries and {content_consolidated} similar facts")

        except Exception as e:
            print(f"⚠️ Error consolidating facts cache: {e}")

    def _optimize_search_cache(self):
        """Identify and remove redundant entries in the search cache"""
        try:
            # Identify low-value entries (low access count, old, few results)
            low_value_entries = []
            current_time = time.time()

            for cache_key, entry in list(self._search_cache.items()):
                # Calculate entry value based on:
                # 1. Access count (higher is better)
                # 2. Recency (newer is better)
                # 3. Number of facts (more is better)

                access_count = entry.get('access_count', 1)
                age = current_time - entry.get('last_access', entry['time'])
                fact_count = len(entry['facts'])

                # Low value if:
                # - Accessed only once AND older than 30 minutes
                # - OR has no facts
                # - OR is older than 2 hours and accessed less than 3 times
                if (access_count == 1 and age > 1800) or \
                   (fact_count == 0) or \
                   (age > 7200 and access_count < 3):
                    low_value_entries.append(cache_key)

            # Remove low-value entries
            for key in low_value_entries:
                if key in self._search_cache:
                    del self._search_cache[key]

            if low_value_entries:
                print(f"🧠 SEARCH CACHE OPTIMIZATION: Removed {len(low_value_entries)} low-value entries")

        except Exception as e:
            print(f"⚠️ Error optimizing search cache: {e}")

    def _cleanup_extraction_cache(self):
        """Clean up the extraction cache to remove old entries"""
        try:
            # Keep only the 50 most recent entries
            if len(self._facts_extraction_cache) > 50:
                # Simple approach: just keep the most recent 50 entries
                # This is efficient for this cache which doesn't need sophisticated management
                keys = list(self._facts_extraction_cache.keys())
                keys_to_remove = keys[:-50]  # Remove all but the last 50

                for key in keys_to_remove:
                    del self._facts_extraction_cache[key]

                print(f"🧠 EXTRACTION CACHE CLEANUP: Removed {len(keys_to_remove)} old entries")

        except Exception as e:
            print(f"⚠️ Error cleaning up extraction cache: {e}")

    def _deduplicate_database_facts(self):
        """
        Find and remove duplicate facts in the database.
        This is a more intensive operation that runs less frequently.
        """
        try:
            conn = sqlite3.connect(self.db_file)

            # First, find facts with identical values
            cursor = conn.execute('''
                SELECT value, COUNT(*) as count, GROUP_CONCAT(category || '.' || key) as entries
                FROM nuclear_facts
                GROUP BY value
                HAVING count > 1
                ORDER BY count DESC
                LIMIT 100
            ''')

            exact_duplicates = cursor.fetchall()
            total_removed = 0

            # Process exact duplicates
            for value, count, entries in exact_duplicates:
                # Skip very short values as they might be coincidental matches
                if len(value) < 10:
                    continue

                # Get all facts with this value
                cursor = conn.execute('''
                    SELECT id, category, key, timestamp
                    FROM nuclear_facts
                    WHERE value = ?
                    ORDER BY timestamp DESC
                ''', (value,))

                duplicate_facts = cursor.fetchall()

                # Keep the most recent fact, delete others
                if len(duplicate_facts) > 1:
                    # Keep the first one (most recent by timestamp)
                    keep_id = duplicate_facts[0][0]
                    keep_category = duplicate_facts[0][1]
                    keep_key = duplicate_facts[0][2]

                    # Delete all others
                    delete_ids = [fact[0] for fact in duplicate_facts[1:]]

                    if delete_ids:
                        placeholders = ','.join(['?'] * len(delete_ids))
                        conn.execute(f'''
                            DELETE FROM nuclear_facts
                            WHERE id IN ({placeholders})
                        ''', delete_ids)

                        total_removed += len(delete_ids)
                        print(f"🔄 DEDUPLICATION: Kept {keep_category}.{keep_key}, removed {len(delete_ids)} duplicates with identical content")

            # Next, find facts with very similar content (for longer text)
            # This is more expensive, so we limit it to longer text values
            cursor = conn.execute('''
                SELECT id, category, key, value
                FROM nuclear_facts
                WHERE length(value) > 50
                ORDER BY timestamp DESC
                LIMIT 1000
            ''')

            long_facts = cursor.fetchall()
            processed_ids = set()  # Track IDs we've already processed

            # Process each long fact
            for fact_id, category, key, value in long_facts:
                # Skip if we've already processed this fact
                if fact_id in processed_ids:
                    continue

                # Extract significant words for fuzzy matching
                significant_words = [word for word in value.split() if len(word) > 4]

                # Skip if not enough significant words
                if len(significant_words) < 3:
                    continue

                # Build a query to find similar facts
                min_matches = max(3, int(len(significant_words) * 0.8))
                query_parts = []
                params = []

                for word in significant_words:
                    query_parts.append("value LIKE ?")
                    params.append(f'%{word}%')

                # Exclude the current fact
                query = f'''
                    SELECT id, category, key, value,
                    {" + ".join([f"(CASE WHEN value LIKE ? THEN 1 ELSE 0 END)" for _ in significant_words])} as match_count
                    FROM nuclear_facts
                    WHERE id != ?
                    HAVING match_count >= ?
                    ORDER BY match_count DESC
                '''

                # Add parameters for LIKE conditions, the fact_id, and min_matches
                all_params = params + params + [fact_id, min_matches]

                cursor = conn.execute(query, all_params)
                similar_facts = cursor.fetchall()

                # If we found similar facts, keep the current one and remove others
                if similar_facts:
                    similar_ids = [f[0] for f in similar_facts]
                    processed_ids.update(similar_ids)  # Mark these as processed

                    # Delete similar facts
                    placeholders = ','.join(['?'] * len(similar_ids))
                    conn.execute(f'''
                        DELETE FROM nuclear_facts
                        WHERE id IN ({placeholders})
                    ''', similar_ids)

                    total_removed += len(similar_ids)
                    print(f"🔄 DEDUPLICATION: Kept {category}.{key}, removed {len(similar_ids)} facts with similar content")

            conn.commit()
            conn.close()

            if total_removed > 0:
                print(f"🧠 DATABASE DEDUPLICATION: Removed {total_removed} duplicate facts")

        except Exception as e:
            print(f"⚠️ Error deduplicating database facts: {e}")
            try:
                conn.close()
            except:
                pass

    def get_all_facts(self):
        """Retrieve all facts from nuclear memory"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.execute('''
                SELECT category, key, value, timestamp FROM nuclear_facts
                ORDER BY timestamp DESC
            ''')
            results = cursor.fetchall()
            conn.close()

            facts = []
            for row in results:
                facts.append({
                    'collection': row[0],
                    'key': row[1],
                    'data': row[2],
                    'timestamp': row[3]
                })

            print(f"🎯 NUCLEAR RETRIEVED: {len(facts)} total facts")
            return facts
        except Exception as e:
            print(f"Error retrieving all facts: {e}")
            return []

    def get_diverse_conversations(self, limit=5):
        """Get diverse conversation examples from memory with optimized performance"""
        try:
            # Reduced default limit from 10 to 5 for better performance

            # Create a connection with a shorter timeout
            conn = sqlite3.connect(self.db_file, timeout=2.0)  # Reduced from 5.0

            # Create index on timestamp if it doesn't exist
            try:
                conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_interactions_timestamp ON memory_interactions(timestamp)")
                conn.commit()
            except Exception as e:
                print(f"Warning: Could not create index: {e}")

            # Check if we have a cached result
            cache_key = f"diverse_conversations_{limit}"
            current_time = time.time()

            if cache_key in self._conversation_cache:
                cached_result = self._conversation_cache[cache_key]
                # Only use cache if it's recent (increased from 60 to 300 seconds)
                if current_time - cached_result['time'] < 300:
                    # Update access statistics
                    if 'access_count' not in cached_result:
                        cached_result['access_count'] = 1
                        cached_result['last_access'] = current_time
                    else:
                        cached_result['access_count'] += 1
                        cached_result['last_access'] = current_time

                    # Update cache entry without changing the conversations
                    self._conversation_cache[cache_key] = cached_result

                    # Track cache hits
                    self.cache_hits += 1

                    # Only log every 5th access to reduce console spam
                    access_count = cached_result.get('access_count', 1)
                    if access_count % 5 == 1:
                        print(f"🎯 CONVERSATION CACHE HIT: Using cached conversations (accessed {access_count} times)")

                    return cached_result['conversations']

            # Cache miss - track it
            self.cache_misses += 1

            # Simplified query that just gets the most recent conversations
            # This is much faster than the previous query that selected distinct session_ids
            query = """
                SELECT user_message, ai_response, session_id, timestamp
                FROM memory_interactions 
                ORDER BY timestamp DESC
                LIMIT ?
            """

            cursor = conn.execute(query, (limit*2,))  # Get twice as many as needed for diversity
            results = cursor.fetchall()
            conn.close()

            # Process results with more aggressive truncation
            conversations = []
            seen_sessions = set()  # Track seen session_ids for diversity

            for row in results:
                session_id = row[2]

                # Skip if we already have a conversation from this session and we have enough conversations
                if session_id in seen_sessions and len(conversations) >= limit:
                    continue

                # More aggressive truncation - reduced from 100 to 80 chars
                user_msg = row[0][:80] if row[0] else ""
                ai_resp = row[1][:80] if row[1] else ""

                conversations.append({
                    "user_message": user_msg,
                    "ai_response": ai_resp,
                    "session_id": session_id,
                    "timestamp": row[3],
                    "extracted_facts": ""  # Simplified - don't bother with extracted_facts
                })

                seen_sessions.add(session_id)

                # Stop once we have enough conversations
                if len(conversations) >= limit:
                    break

            # Cache the result with enhanced metadata
            current_time = time.time()
            self._conversation_cache[cache_key] = {
                'conversations': conversations,
                'time': current_time,
                'access_count': 1,
                'last_access': current_time,
                'limit': limit
            }

            # Implement a more sophisticated cache eviction policy
            # based on both recency and frequency (LFU/LRU hybrid)
            if len(self._conversation_cache) > 20:  # Increased from 10 to 20
                # Calculate a score for each cache entry based on:
                # 1. How recently it was accessed (higher is better)
                # 2. How frequently it's accessed (higher is better)
                # 3. How many conversations it contains (higher is better)
                scored_entries = []
                for k, entry in self._conversation_cache.items():
                    # Recency score: 0-1 based on how recently it was accessed
                    last_access = entry.get('last_access', entry['time'])
                    recency = max(0, min(1, (current_time - last_access) / 300))

                    # Frequency score: log scale to prevent very frequent items from dominating
                    access_count = entry.get('access_count', 1)
                    frequency = min(10, math.log2(access_count + 1))

                    # Value score: more conversations = more valuable
                    value = min(5, len(entry['conversations']))

                    # Combined score (lower is better for eviction)
                    score = (1 - recency) * 0.5 + frequency * 0.3 + value * 0.2

                    scored_entries.append((k, score))

                # Sort by score (ascending) and remove the 5 lowest scoring entries
                scored_entries.sort(key=lambda x: x[1])
                for k, _ in scored_entries[:5]:  # Increased from 3 to 5
                    del self._conversation_cache[k]

                print(f"🧠 CONVERSATION CACHE MANAGEMENT: Evicted 5 least valuable entries, keeping {len(self._conversation_cache)}")

            # Log cache statistics periodically
            if (self.cache_hits + self.cache_misses) % 100 == 0:
                hit_rate = self.cache_hits / max(1, (self.cache_hits + self.cache_misses)) * 100
                print(f"🧠 CACHE STATS: Hit rate: {hit_rate:.1f}% ({self.cache_hits} hits, {self.cache_misses} misses)")

            print(f"🎯 NUCLEAR RETRIEVED: {len(conversations)} diverse conversations")
            return conversations
        except Exception as e:
            print(f"Error getting diverse conversations: {e}")
            # Return an empty list in case of error to prevent crashes
            return []

# Create global instance
NUCLEAR_MEMORY = SQLiteNuclearMemory()
