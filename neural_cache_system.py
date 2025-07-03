import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import json
import hashlib
import random
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional, Union

class NeuralEmbeddingLayer(nn.Module):
    """Neural embedding layer for transforming inputs into vector representations."""

    def __init__(self, input_dim: int = 768, output_dim: int = 256):
        """
        Initialize the neural embedding layer.

        Args:
            input_dim: Dimension of input vectors
            output_dim: Dimension of output embeddings
        """
        super(NeuralEmbeddingLayer, self).__init__()

        # Create a multi-layer neural network for embedding
        self.embedding_network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 384),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(384, output_dim),
            nn.Tanh()  # Normalize embeddings to range [-1, 1]
        )

    def forward(self, x):
        """Transform input vectors into embeddings."""
        return self.embedding_network(x)

class RelevancePredictor(nn.Module):
    """Neural network for predicting relevance of cached items."""

    def __init__(self, embedding_dim: int = 256):
        """
        Initialize the relevance predictor.

        Args:
            embedding_dim: Dimension of input embeddings
        """
        super(RelevancePredictor, self).__init__()

        # Create a neural network for relevance prediction
        self.relevance_network = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 1, 128),  # *2 for query and item embeddings, +1 for time feature
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )

    def forward(self, query_embedding, item_embedding, time_feature):
        """
        Predict relevance score for a query-item pair.

        Args:
            query_embedding: Embedding of the query
            item_embedding: Embedding of the cached item
            time_feature: Time-based feature (e.g., age of cached item)

        Returns:
            Relevance score between 0 and 1
        """
        # Check time_feature shape and unsqueeze appropriately
        if time_feature.dim() == 0:  # Scalar tensor
            time_feature = time_feature.unsqueeze(0).unsqueeze(0)
        elif time_feature.dim() == 1:  # 1D tensor
            time_feature = time_feature.unsqueeze(1)
        # For 2D tensor, we assume it's already in the right shape

        # Concatenate query embedding, item embedding, and time feature
        combined = torch.cat([query_embedding, item_embedding, time_feature], dim=1)
        return self.relevance_network(combined)

class NeuralCacheSystem:
    """
    A highly complex, neural network-based caching system that replaces traditional cache storage.

    This system uses neural networks to:
    1. Generate embeddings for queries and cached items
    2. Predict relevance of cached items to current queries
    3. Make human-like decisions about when to use cached data
    4. Adapt over time to improve efficiency and accuracy
    """

    def __init__(self, device: str = None):
        """
        Initialize the neural cache system.

        Args:
            device: Device to run neural networks on ('cpu' or 'cuda')
        """
        # Determine device (use GPU if available)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🧠 Neural Cache System initializing on {self.device}")

        # Initialize neural network components
        self.embedding_layer = NeuralEmbeddingLayer().to(self.device)
        self.relevance_predictor = RelevancePredictor().to(self.device)

        # Initialize optimizers
        self.embedding_optimizer = optim.Adam(self.embedding_layer.parameters(), lr=0.001)
        self.relevance_optimizer = optim.Adam(self.relevance_predictor.parameters(), lr=0.001)

        # Initialize storage for cached items
        self.response_store = []
        self.fact_store = []
        self.search_store = []

        # Initialize experience buffer for training
        self.experience_buffer = []
        self.max_buffer_size = 1000

        # Track performance metrics
        self.hit_count = 0
        self.miss_count = 0
        self.last_training_time = time.time()

        # Adaptive parameters
        self.relevance_threshold = 0.75  # Minimum relevance score to use cached item
        self.time_decay_factor = 0.1  # How quickly relevance decays with time
        self.max_cache_size = 200  # Maximum number of items in each cache

        # Create random initial embeddings for text
        self.random_embedding = torch.rand(768).to(self.device)

        # Try to import sentence transformers for better embeddings
        try:
            from sentence_transformers import SentenceTransformer
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L12-v2')
            self.has_transformer = True
            print("✅ Using SentenceTransformer (all-MiniLM-L12-v2) for 768D embeddings")
        except ImportError:
            self.has_transformer = False
            print("⚠️ SentenceTransformer not available, using simpler embedding method")

    def get_text_embedding(self, text: str) -> torch.Tensor:
        """
        Generate an embedding for text input.

        Args:
            text: Input text

        Returns:
            Embedding tensor
        """
        if not text:
            return self.random_embedding

        # Use sentence transformer if available
        if self.has_transformer:
            # Generate embedding using sentence transformer
            with torch.no_grad():
                embedding = torch.tensor(self.sentence_transformer.encode(text[:1000])).to(self.device)
                return embedding

        # Fallback method if sentence transformer is not available
        # Create a simple embedding based on character frequencies
        char_freq = {}
        for char in text.lower():
            if char.isalnum():
                char_freq[char] = char_freq.get(char, 0) + 1

        # Create a vector from character frequencies
        embedding = torch.zeros(768).to(self.device)
        for i, char in enumerate("abcdefghijklmnopqrstuvwxyz0123456789"):
            if i < 768:
                embedding[i] = char_freq.get(char, 0) / max(1, len(text))

        return embedding

    def get_cache_key(self, query: str, context: str = "") -> str:
        """
        Generate a unique cache key for a query.

        Args:
            query: The query string
            context: Additional context (e.g., model name)

        Returns:
            Cache key string
        """
        # Create a hash of the query and context
        combined = (query + context).encode('utf-8')
        return hashlib.md5(combined).hexdigest()

    def get_time_feature(self, timestamp: float) -> torch.Tensor:
        """
        Convert a timestamp into a time feature for the neural network.

        Args:
            timestamp: Unix timestamp

        Returns:
            Time feature tensor
        """
        # Calculate age in seconds
        age = time.time() - timestamp

        # Normalize age (0 = fresh, 1 = old)
        # Use a logarithmic scale to better represent time differences
        normalized_age = min(1.0, max(0.0, np.log(1 + age / 60) / np.log(1 + 3600)))

        return torch.tensor(normalized_age, dtype=torch.float32).to(self.device)

    def predict_relevance(self, query_embedding: torch.Tensor, item_embedding: torch.Tensor, 
                          time_feature: torch.Tensor) -> float:
        """
        Predict the relevance of a cached item to the current query.

        Args:
            query_embedding: Embedding of the query
            item_embedding: Embedding of the cached item
            time_feature: Time-based feature

        Returns:
            Relevance score between 0 and 1
        """
        with torch.no_grad():
            # Check if embeddings need to be reshaped
            if query_embedding.dim() == 1:
                query_embedding = query_embedding.unsqueeze(0)
            if item_embedding.dim() == 1:
                item_embedding = item_embedding.unsqueeze(0)

            # Check embedding dimensions and process if needed
            if query_embedding.shape[1] == 768:  # Raw embedding from get_text_embedding
                query_embedding = self.embedding_layer(query_embedding)
            elif query_embedding.shape[1] != 256:  # Not already processed to output dim
                print(f"⚠️ NEURAL CACHE WARNING: Unexpected query embedding dimension: {query_embedding.shape} vs expected 768")
                print(f"  - Model: {self.sentence_transformer._model_name if hasattr(self, 'sentence_transformer') and hasattr(self.sentence_transformer, '_model_name') else 'unknown'}")
                print(f"  - Source: query embedding")
                # Try to adapt the embedding to the expected dimension
                query_embedding = torch.nn.functional.pad(query_embedding, (0, 256 - query_embedding.shape[1])) if query_embedding.shape[1] < 256 else query_embedding[:, :256]

            if item_embedding.shape[1] == 768:  # Raw embedding from get_text_embedding
                item_embedding = self.embedding_layer(item_embedding)
            elif item_embedding.shape[1] != 256:  # Not already processed to output dim
                print(f"⚠️ NEURAL CACHE WARNING: Unexpected item embedding dimension: {item_embedding.shape} vs expected 768")
                print(f"  - Model: {self.sentence_transformer._model_name if hasattr(self, 'sentence_transformer') and hasattr(self.sentence_transformer, '_model_name') else 'unknown'}")
                print(f"  - Source: item embedding")
                # Try to adapt the embedding to the expected dimension
                item_embedding = torch.nn.functional.pad(item_embedding, (0, 256 - item_embedding.shape[1])) if item_embedding.shape[1] < 256 else item_embedding[:, :256]

            # Predict relevance
            relevance = self.relevance_predictor(query_embedding, item_embedding, time_feature)

            return relevance.item()

    def get_cached_response(self, query: str, model: str = "") -> Optional[Dict[str, Any]]:
        """
        Get a cached response for a query if available and relevant.

        Args:
            query: The query string
            model: Model name for context

        Returns:
            Cached response or None if no relevant cache exists
        """
        if not self.response_store:
            return None

        # Generate query embedding
        query_embedding = self.get_text_embedding(query)

        # Find the most relevant cached response
        best_relevance = 0
        best_response = None

        for item in self.response_store:
            # Skip if model doesn't match
            if model and item['model'] != model:
                continue

            # Get item embedding and time feature
            item_embedding = item['embedding']
            time_feature = self.get_time_feature(item['time'])

            # Predict relevance
            relevance = self.predict_relevance(query_embedding, item_embedding, time_feature)

            # Update best response if this one is more relevant
            if relevance > best_relevance:
                best_relevance = relevance
                best_response = item

        # Use cached response if relevance exceeds threshold
        if best_relevance > self.relevance_threshold:
            self.hit_count += 1
            print(f"🧠 NEURAL CACHE HIT: Response relevance {best_relevance:.2f}")

            # Add to experience buffer for training
            self.experience_buffer.append({
                'query_embedding': query_embedding,
                'item_embedding': best_response['embedding'],
                'time_feature': self.get_time_feature(best_response['time']),
                'label': 1.0  # Positive example
            })

            return best_response

        self.miss_count += 1
        return None

    def store_response(self, query: str, response: str, model: str = "") -> None:
        """
        Store a response in the neural cache.

        Args:
            query: The query string
            response: The response string
            model: Model name for context
        """
        # Generate embeddings
        query_embedding = self.get_text_embedding(query)
        response_embedding = self.get_text_embedding(response)

        # Check for embedding dimension mismatches
        expected_dim = 768
        if query_embedding.shape[1] != expected_dim or response_embedding.shape[1] != expected_dim:
            print(f"⚠️ NEURAL CACHE WARNING: Embedding dimension mismatch detected")
            print(f"  - Query embedding shape: {query_embedding.shape}")
            print(f"  - Response embedding shape: {response_embedding.shape}")
            print(f"  - Expected dimension: {expected_dim}")
            print(f"  - Model: {self.sentence_transformer._model_name if hasattr(self, 'sentence_transformer') and hasattr(self.sentence_transformer, '_model_name') else 'unknown'}")
            print(f"  - Action: Skipping cache insert due to persistent shape mismatch")
            return  # Skip cache insert

        # Create cache item
        cache_item = {
            'query': query,
            'response': response,
            'model': model,
            'time': time.time(),
            'embedding': response_embedding,
            'query_embedding': query_embedding
        }

        # Add to cache
        self.response_store.append(cache_item)

        # Limit cache size
        if len(self.response_store) > self.max_cache_size:
            # Remove least recently used items
            self.response_store.sort(key=lambda x: x['time'])
            self.response_store = self.response_store[-self.max_cache_size:]

        # Periodically train the neural network
        self._maybe_train()

    def get_cached_facts(self, keywords: List[str]) -> Optional[List[str]]:
        """
        Get cached facts for keywords if available and relevant.

        Args:
            keywords: List of keywords

        Returns:
            List of facts or None if no relevant cache exists
        """
        if not self.fact_store:
            return None

        # Generate query embedding from keywords
        query_text = " ".join(keywords)
        query_embedding = self.get_text_embedding(query_text)

        # Find the most relevant cached facts
        best_relevance = 0
        best_facts = None

        for item in self.fact_store:
            # Get item embedding and time feature
            item_embedding = item['embedding']
            time_feature = self.get_time_feature(item['time'])

            # Predict relevance
            relevance = self.predict_relevance(query_embedding, item_embedding, time_feature)

            # Update best facts if these are more relevant
            if relevance > best_relevance:
                best_relevance = relevance
                best_facts = item

        # Use cached facts if relevance exceeds threshold
        if best_relevance > self.relevance_threshold:
            self.hit_count += 1
            print(f"🧠 NEURAL CACHE HIT: Facts relevance {best_relevance:.2f}")

            # Add to experience buffer for training
            self.experience_buffer.append({
                'query_embedding': query_embedding,
                'item_embedding': best_facts['embedding'],
                'time_feature': self.get_time_feature(best_facts['time']),
                'label': 1.0  # Positive example
            })

            return best_facts['facts']

        self.miss_count += 1
        return None

    def store_facts(self, keywords: List[str], facts: List[str]) -> None:
        """
        Store facts in the neural cache.

        Args:
            keywords: List of keywords
            facts: List of facts
        """
        # Generate embeddings
        query_text = " ".join(keywords)
        query_embedding = self.get_text_embedding(query_text)
        facts_text = " ".join(facts)
        facts_embedding = self.get_text_embedding(facts_text)

        # Check for embedding dimension mismatches
        expected_dim = 768
        if query_embedding.shape[1] != expected_dim or facts_embedding.shape[1] != expected_dim:
            print(f"⚠️ NEURAL CACHE WARNING: Embedding dimension mismatch detected in fact storage")
            print(f"  - Query embedding shape: {query_embedding.shape}")
            print(f"  - Facts embedding shape: {facts_embedding.shape}")
            print(f"  - Expected dimension: {expected_dim}")
            print(f"  - Model: {self.sentence_transformer._model_name if hasattr(self, 'sentence_transformer') and hasattr(self.sentence_transformer, '_model_name') else 'unknown'}")
            print(f"  - Action: Skipping cache insert due to persistent shape mismatch")
            return  # Skip cache insert

        # Create cache item
        cache_item = {
            'keywords': keywords,
            'facts': facts,
            'time': time.time(),
            'embedding': facts_embedding,
            'query_embedding': query_embedding
        }

        # Add to cache
        self.fact_store.append(cache_item)

        # Limit cache size
        if len(self.fact_store) > self.max_cache_size:
            # Remove least recently used items
            self.fact_store.sort(key=lambda x: x['time'])
            self.fact_store = self.fact_store[-self.max_cache_size:]

        # Periodically train the neural network
        self._maybe_train()

    def get_cached_search(self, query: str, search_type: str = "web") -> Optional[Dict[str, Any]]:
        """
        Get cached search results for a query if available and relevant.

        Args:
            query: The search query
            search_type: Type of search ("web", "news", "images")

        Returns:
            Cached search results or None if no relevant cache exists
        """
        if not self.search_store:
            return None

        # Generate query embedding
        query_embedding = self.get_text_embedding(query)

        # Find the most relevant cached search results
        best_relevance = 0
        best_search = None

        for item in self.search_store:
            # Skip if search type doesn't match
            if item['search_type'] != search_type:
                continue

            # Get item embedding and time feature
            item_embedding = item['embedding']
            time_feature = self.get_time_feature(item['time'])

            # Predict relevance
            relevance = self.predict_relevance(query_embedding, item_embedding, time_feature)

            # Update best search if this one is more relevant
            if relevance > best_relevance:
                best_relevance = relevance
                best_search = item

        # Use cached search if relevance exceeds threshold
        if best_relevance > self.relevance_threshold:
            self.hit_count += 1
            print(f"🧠 NEURAL CACHE HIT: Search relevance {best_relevance:.2f}")

            # Add to experience buffer for training
            self.experience_buffer.append({
                'query_embedding': query_embedding,
                'item_embedding': best_search['embedding'],
                'time_feature': self.get_time_feature(best_search['time']),
                'label': 1.0  # Positive example
            })

            return best_search

        self.miss_count += 1
        return None

    def store_search(self, query: str, results: str, search_type: str = "web") -> None:
        """
        Store search results in the neural cache.

        Args:
            query: The search query
            results: The search results
            search_type: Type of search ("web", "news", "images")
        """
        # Generate embeddings
        query_embedding = self.get_text_embedding(query)
        results_embedding = self.get_text_embedding(results)

        # Check for embedding dimension mismatches
        expected_dim = 768
        if query_embedding.shape[1] != expected_dim or results_embedding.shape[1] != expected_dim:
            print(f"⚠️ NEURAL CACHE WARNING: Embedding dimension mismatch detected in search storage")
            print(f"  - Query embedding shape: {query_embedding.shape}")
            print(f"  - Results embedding shape: {results_embedding.shape}")
            print(f"  - Expected dimension: {expected_dim}")
            print(f"  - Model: {self.sentence_transformer._model_name if hasattr(self, 'sentence_transformer') and hasattr(self.sentence_transformer, '_model_name') else 'unknown'}")
            print(f"  - Action: Skipping cache insert due to persistent shape mismatch")
            return  # Skip cache insert

        # Create cache item
        cache_item = {
            'query': query,
            'results': results,
            'search_type': search_type,
            'time': time.time(),
            'embedding': results_embedding,
            'query_embedding': query_embedding
        }

        # Add to cache
        self.search_store.append(cache_item)

        # Limit cache size
        if len(self.search_store) > self.max_cache_size:
            # Remove least recently used items
            self.search_store.sort(key=lambda x: x['time'])
            self.search_store = self.search_store[-self.max_cache_size:]

        # Periodically train the neural network
        self._maybe_train()

    def _maybe_train(self) -> None:
        """Periodically train the neural network on the experience buffer."""
        # Only train every 5 minutes
        current_time = time.time()
        if current_time - self.last_training_time < 300:  # 5 minutes
            return

        # Only train if we have enough experiences
        if len(self.experience_buffer) < 10:
            return

        print(f"🧠 NEURAL CACHE: Training on {len(self.experience_buffer)} experiences")

        # Generate negative examples (non-matching query-item pairs)
        negative_examples = []
        for _ in range(min(len(self.experience_buffer), 20)):
            # Get two random experiences
            exp1 = random.choice(self.experience_buffer)
            exp2 = random.choice(self.experience_buffer)

            # Create a negative example by mixing query from exp1 with item from exp2
            negative_examples.append({
                'query_embedding': exp1['query_embedding'],
                'item_embedding': exp2['item_embedding'],
                'time_feature': exp1['time_feature'],
                'label': 0.0  # Negative example
            })

        # Combine positive and negative examples
        training_data = self.experience_buffer + negative_examples
        random.shuffle(training_data)

        # Train the neural network
        self._train(training_data)

        # Update last training time
        self.last_training_time = current_time

        # Limit experience buffer size
        if len(self.experience_buffer) > self.max_buffer_size:
            # Keep more recent experiences
            self.experience_buffer = self.experience_buffer[-self.max_buffer_size:]

    def _train(self, training_data: List[Dict[str, Any]]) -> None:
        """
        Train the neural network on a batch of experiences.

        Args:
            training_data: List of training examples
        """
        # Set networks to training mode
        self.embedding_layer.train()
        self.relevance_predictor.train()

        # Training parameters
        num_epochs = 3
        batch_size = 8

        # Create loss function
        criterion = nn.BCELoss()

        # Train for multiple epochs
        for epoch in range(num_epochs):
            total_loss = 0.0

            # Process in batches
            for i in range(0, len(training_data), batch_size):
                batch = training_data[i:i+batch_size]

                # Zero gradients
                self.embedding_optimizer.zero_grad()
                self.relevance_optimizer.zero_grad()

                batch_loss = 0.0

                # Process each example in the batch
                for example in batch:
                    # Get data
                    query_embedding = example['query_embedding']
                    item_embedding = example['item_embedding']
                    time_feature = example['time_feature']
                    label = torch.tensor(example['label'], dtype=torch.float32).to(self.device)

                    # Forward pass
                    # Check if embeddings need to be reshaped
                    if query_embedding.dim() == 1:
                        query_embedding = query_embedding.unsqueeze(0)
                    if item_embedding.dim() == 1:
                        item_embedding = item_embedding.unsqueeze(0)

                    # Check time_feature shape and unsqueeze appropriately
                    if time_feature.dim() == 0:  # Scalar tensor
                        time_feature = time_feature.unsqueeze(0).unsqueeze(0)
                    elif time_feature.dim() == 1:  # 1D tensor
                        time_feature = time_feature.unsqueeze(1)
                    # For 2D tensor, we assume it's already in the right shape

                    # Check embedding dimensions and process if needed
                    if query_embedding.shape[1] == 768:  # Raw embedding from get_text_embedding
                        query_embedding = self.embedding_layer(query_embedding)
                    elif query_embedding.shape[1] != 256:  # Not already processed to output dim
                        # Try to adapt the embedding to the expected dimension
                        query_embedding = torch.nn.functional.pad(query_embedding, (0, 256 - query_embedding.shape[1])) if query_embedding.shape[1] < 256 else query_embedding[:, :256]

                    if item_embedding.shape[1] == 768:  # Raw embedding from get_text_embedding
                        item_embedding = self.embedding_layer(item_embedding)
                    elif item_embedding.shape[1] != 256:  # Not already processed to output dim
                        # Try to adapt the embedding to the expected dimension
                        item_embedding = torch.nn.functional.pad(item_embedding, (0, 256 - item_embedding.shape[1])) if item_embedding.shape[1] < 256 else item_embedding[:, :256]

                    prediction = self.relevance_predictor(query_embedding, item_embedding, time_feature)

                    # Calculate loss
                    loss = criterion(prediction, label.unsqueeze(0).unsqueeze(0))
                    batch_loss += loss.item()

                    # Backward pass
                    loss.backward()

                # Average loss over batch
                batch_loss /= len(batch)
                total_loss += batch_loss

                # Update weights
                self.embedding_optimizer.step()
                self.relevance_optimizer.step()

            # Print progress
            avg_loss = total_loss / (len(training_data) / batch_size)
            print(f"🧠 NEURAL CACHE: Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # Set networks back to evaluation mode
        self.embedding_layer.eval()
        self.relevance_predictor.eval()

        # Adjust relevance threshold based on hit/miss ratio
        if self.hit_count + self.miss_count > 0:
            hit_ratio = self.hit_count / (self.hit_count + self.miss_count)

            # If hit ratio is too low, decrease threshold to be more lenient
            if hit_ratio < 0.2:
                self.relevance_threshold = max(0.5, self.relevance_threshold - 0.05)
                print(f"🧠 NEURAL CACHE: Decreasing relevance threshold to {self.relevance_threshold:.2f}")

            # If hit ratio is too high, increase threshold to be more strict
            elif hit_ratio > 0.8:
                self.relevance_threshold = min(0.95, self.relevance_threshold + 0.05)
                print(f"🧠 NEURAL CACHE: Increasing relevance threshold to {self.relevance_threshold:.2f}")

        # Reset hit/miss counters
        self.hit_count = 0
        self.miss_count = 0

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the neural cache system.

        Returns:
            Dictionary of statistics
        """
        return {
            'response_cache_size': len(self.response_store),
            'fact_cache_size': len(self.fact_store),
            'search_cache_size': len(self.search_store),
            'experience_buffer_size': len(self.experience_buffer),
            'relevance_threshold': self.relevance_threshold,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_ratio': self.hit_count / (self.hit_count + self.miss_count) if (self.hit_count + self.miss_count) > 0 else 0,
            'device': self.device,
            'has_transformer': self.has_transformer
        }
