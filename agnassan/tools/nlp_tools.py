"""Natural language processing tools for Agnassan.

This module provides tools for text analysis, sentiment analysis, entity recognition,
and other NLP tasks that enhance the capabilities of language models.
"""

import logging
import re
import string
from typing import Dict, List, Any, Optional, Union, Tuple
import json

try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer, PorterStemmer
    from nltk.tag import pos_tag
    from nltk.chunk import ne_chunk
    
    # Download necessary NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger', quiet=True)
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    
    try:
        nltk.data.find('chunkers/maxent_ne_chunker')
    except LookupError:
        nltk.download('maxent_ne_chunker', quiet=True)
    
    try:
        nltk.data.find('corpora/words')
    except LookupError:
        nltk.download('words', quiet=True)
    
    HAS_NLP_DEPS = True
except ImportError:
    HAS_NLP_DEPS = False

from .index import register_tool

# Set up logging
logger = logging.getLogger("agnassan.tools.nlp")

@register_tool(
    name="tokenize_text",
    description="Tokenize text into words or sentences."
)
def tokenize_text(text: str, tokenize_type: str = "word") -> List[str]:
    """Tokenize text into words or sentences.
    
    Args:
        text: The text to tokenize
        tokenize_type: The type of tokenization ('word' or 'sentence')
        
    Returns:
        A list of tokens (words or sentences)
    """
    try:
        logger.info(f"Tokenizing text into {tokenize_type}s")
        
        # Check if NLTK is available
        if not HAS_NLP_DEPS:
            return [f"Error: NLTK is not available. Install nltk for text tokenization."]
        
        if tokenize_type.lower() == "word":
            return word_tokenize(text)
        elif tokenize_type.lower() == "sentence":
            return sent_tokenize(text)
        else:
            return [f"Error: Unknown tokenization type: {tokenize_type}. Use 'word' or 'sentence'."]
    except Exception as e:
        logger.error(f"Error tokenizing text: {str(e)}")
        return [f"Error: {str(e)}"]

@register_tool(
    name="remove_stopwords",
    description="Remove stopwords from text."
)
def remove_stopwords(text: str, language: str = "english") -> str:
    """Remove stopwords from text.
    
    Args:
        text: The text to process
        language: The language of the text (default: english)
        
    Returns:
        Text with stopwords removed
    """
    try:
        logger.info(f"Removing stopwords from {language} text")
        
        # Check if NLTK is available
        if not HAS_NLP_DEPS:
            return f"Error: NLTK is not available. Install nltk for stopword removal."
        
        # Tokenize the text
        words = word_tokenize(text)
        
        # Get stopwords for the specified language
        try:
            stop_words = set(stopwords.words(language))
        except Exception:
            return f"Error: Stopwords not available for language: {language}"
        
        # Remove stopwords
        filtered_words = [word for word in words if word.lower() not in stop_words]
        
        # Reconstruct the text
        return ' '.join(filtered_words)
    except Exception as e:
        logger.error(f"Error removing stopwords: {str(e)}")
        return f"Error: {str(e)}"

@register_tool(
    name="lemmatize_text",
    description="Lemmatize words in text to their base form."
)
def lemmatize_text(text: str) -> str:
    """Lemmatize words in text to their base form.
    
    Args:
        text: The text to lemmatize
        
    Returns:
        Text with words lemmatized to their base form
    """
    try:
        logger.info("Lemmatizing text")
        
        # Check if NLTK is available
        if not HAS_NLP_DEPS:
            return f"Error: NLTK is not available. Install nltk for lemmatization."
        
        # Tokenize the text
        words = word_tokenize(text)
        
        # Initialize lemmatizer
        lemmatizer = WordNetLemmatizer()
        
        # Lemmatize each word
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
        
        # Reconstruct the text
        return ' '.join(lemmatized_words)
    except Exception as e:
        logger.error(f"Error lemmatizing text: {str(e)}")
        return f"Error: {str(e)}"

@register_tool(
    name="extract_entities",
    description="Extract named entities from text."
)
def extract_entities(text: str) -> Dict[str, List[str]]:
    """Extract named entities from text.
    
    Args:
        text: The text to analyze
        
    Returns:
        A dictionary of entity types and their occurrences
    """
    try:
        logger.info("Extracting named entities from text")
        
        # Check if NLTK is available
        if not HAS_NLP_DEPS:
            return {"error": "NLTK is not available. Install nltk for entity extraction."}
        
        # Tokenize and tag the text
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        
        # Extract named entities
        entities = ne_chunk(tagged)
        
        # Organize entities by type
        entity_dict = {}
        for entity in entities:
            if hasattr(entity, 'label'):
                entity_type = entity.label()
                entity_text = ' '.join([child[0] for child in entity])
                
                if entity_type not in entity_dict:
                    entity_dict[entity_type] = []
                
                entity_dict[entity_type].append(entity_text)
        
        return entity_dict
    except Exception as e:
        logger.error(f"Error extracting entities: {str(e)}")
        return {"error": str(e)}

@register_tool(
    name="analyze_sentiment",
    description="Analyze the sentiment of text."
)
def analyze_sentiment(text: str) -> Dict[str, Any]:
    """Analyze the sentiment of text.
    
    This is a simple rule-based sentiment analyzer. In a real implementation,
    you would use a more sophisticated approach or a dedicated library.
    
    Args:
        text: The text to analyze
        
    Returns:
        A dictionary containing sentiment analysis results
    """
    try:
        logger.info("Analyzing sentiment of text")
        
        # Check if NLTK is available
        if not HAS_NLP_DEPS:
            return {"error": "NLTK is not available. Install nltk for sentiment analysis."}
        
        # Simple lists of positive and negative words
        # In a real implementation, you would use a more comprehensive lexicon
        positive_words = [
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'terrific', 'outstanding', 'superb', 'nice', 'happy', 'love', 'best',
            'better', 'awesome', 'brilliant', 'positive', 'perfect', 'pleasant'
        ]
        
        negative_words = [
            'bad', 'terrible', 'awful', 'horrible', 'poor', 'disappointing',
            'worst', 'hate', 'dislike', 'negative', 'sad', 'angry', 'upset',
            'unfortunate', 'unpleasant', 'mediocre', 'inferior', 'useless'
        ]
        
        # Tokenize and clean the text
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalpha()]
        
        # Count positive and negative words
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        # Calculate sentiment score (-1 to 1)
        total_sentiment_words = positive_count + negative_count
        if total_sentiment_words > 0:
            sentiment_score = (positive_count - negative_count) / total_sentiment_words
        else:
            sentiment_score = 0
        
        # Determine sentiment label
        if sentiment_score > 0.2:
            sentiment = "positive"
        elif sentiment_score < -0.2:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        return {
            "sentiment": sentiment,
            "score": sentiment_score,
            "positive_words": positive_count,
            "negative_words": negative_count,
            "total_words": len(words)
        }
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {str(e)}")
        return {"error": str(e)}

@register_tool(
    name="summarize_text",
    description="Generate a summary of text using extractive summarization."
)
def summarize_text(text: str, sentences: int = 3) -> str:
    """Generate a summary of text using extractive summarization.
    
    This is a simple extractive summarization approach. In a real implementation,
    you would use a more sophisticated approach or a dedicated library.
    
    Args:
        text: The text to summarize
        sentences: Number of sentences to include in the summary
        
    Returns:
        A summary of the text
    """
    try:
        logger.info(f"Summarizing text to {sentences} sentences")
        
        # Check if NLTK is available
        if not HAS_NLP_DEPS:
            return f"Error: NLTK is not available. Install nltk for text summarization."
        
        # Split text into sentences
        sentence_list = sent_tokenize(text)
        
        # Return the full text if it's already shorter than requested summary
        if len(sentence_list) <= sentences:
            return text
        
        # Calculate word frequency
        word_frequencies = {}
        for word in word_tokenize(text.lower()):
            if word not in stopwords.words('english') and word not in string.punctuation:
                if word not in word_frequencies:
                    word_frequencies[word] = 1
                else:
                    word_frequencies[word] += 1
        
        # Normalize word frequencies
        max_frequency = max(word_frequencies.values()) if word_frequencies else 1
        for word in word_frequencies:
            word_frequencies[word] = word_frequencies[word] / max_frequency
        
        # Calculate sentence scores based on word frequencies
        sentence_scores = {}
        for i, sentence in enumerate(sentence_list):
            for word in word_tokenize(sentence.lower()):
                if word in word_frequencies:
                    if i not in sentence_scores:
                        sentence_scores[i] = word_frequencies[word]
                    else:
                        sentence_scores[i] += word_frequencies[word]
        
        # Get the top sentences
        top_sentence_indices = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:sentences]
        top_sentence_indices.sort()  # Sort by original order
        
        # Construct the summary
        summary = ' '.join([sentence_list[i] for i in top_sentence_indices])
        
        return summary
    except Exception as e:
        logger.error(f"Error summarizing text: {str(e)}")
        return f"Error: {str(e)}"

@register_tool(
    name="extract_keywords",
    description="Extract keywords from text."
)
def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    """Extract keywords from text.
    
    Args:
        text: The text to analyze
        top_n: Number of top keywords to return
        
    Returns:
        A list of keywords
    """
    try:
        logger.info(f"Extracting top {top_n} keywords from text")
        
        # Check if NLTK is available
        if not HAS_NLP_DEPS:
            return [f"Error: NLTK is not available. Install nltk for keyword extraction."]
        
        # Tokenize and clean the text
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalpha()]
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]
        
        # Count word frequencies
        word_freq = {}
        for word in words:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1
        
        # Sort by frequency and get top N
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        keywords = [word for word, freq in sorted_words[:top_n]]
        
        return keywords
    except Exception as e:
        logger.error(f"Error extracting keywords: {str(e)}")
        return [f"Error: {str(e)}"]