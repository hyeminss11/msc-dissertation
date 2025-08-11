import json
import copy
import uuid
import re
from collections import defaultdict

def filter_authors_with_enough_docs(docs, min_docs_per_genre=2):
    """
    Keeps only authors who have at least `min_docs_per_genre` in each genre.
    """
    author_genre_counts = defaultdict(lambda: defaultdict(int))
    for doc in docs:
        author_genre_counts[doc["author"]][doc["genre"]] += 1

    eligible_authors = {
        author for author, genre_counts in author_genre_counts.items()
        if all(genre_counts.get(genre, 0) >= min_docs_per_genre for genre in ["Article", "Tweet"])
    }

    filtered_docs = [doc for doc in docs if doc["author"] in eligible_authors]
    return filtered_docs

# There are tweets with @userid mention. They will be treatd anonymously.
# Also, since links can confuse the model, they are going to be treated as <URL> equally.
def anonymize_text(text):
    """
    Replaces @mentions with @USER and any URL with <URL>.
    """
    text = re.sub(r'@\w+', '@USER', text)  # Anonymize @username
    text = re.sub(r'https?://\S+|www\.\S+', '<URL>', text)  # Anonymize links
    return text

# Function to process documents with word count.
def process_documents_by_word_count(docs, min_word_threshold=100, max_word_limit=3000):
    """
    Short documents are merged by author and genre until min_word_threshold is met.
    Long documents are split into multiple documents by max_word_limit.
    All operations are done on word-count basis.
    Each processed doc gets a new unique ID, and original ID(s) are stored in source_id.
    """
    grouped = {}
    for doc in docs:
        key = (doc["author"], doc["genre"])
        grouped.setdefault(key, []).append(doc)
    
    processed_docs = []
    global_id_counter = 0  # to assign new unique numeric IDs

    for (author, genre), doc_list in grouped.items():
        doc_list = sorted(doc_list, key=lambda d: len(d["text"].split()))
        
        buffer_words = []
        buffer_ids = []
        
        for doc in doc_list:
            doc["text"] = anonymize_text(doc["text"])
            words = doc["text"].split()
            
            if len(words) < min_word_threshold:
                buffer_words.extend(words)
                buffer_ids.append(doc["id"])
                
                if len(buffer_words) >= min_word_threshold:
                    new_text = " ".join(buffer_words)
                    merged_doc = {
                        "text": new_text,
                        "author": author,
                        "genre": genre,
                        "source_id": "+".join(buffer_ids)
                    }
                    if len(new_text.split()) > max_word_limit:
                        processed_docs.extend(
                            split_long_doc(merged_doc, max_word_limit, global_id_counter)
                        )
                        global_id_counter += len(processed_docs) - global_id_counter
                    else:
                        merged_doc["id"] = str(global_id_counter)
                        global_id_counter += 1
                        processed_docs.append(merged_doc)
                    buffer_words = []
                    buffer_ids = []
            else:
                new_doc = {
                    "text": doc["text"],
                    "author": author,
                    "genre": genre,
                    "source_id": doc["id"]
                }
                if len(words) > max_word_limit:
                    processed_docs.extend(
                        split_long_doc(new_doc, max_word_limit, global_id_counter)
                    )
                    global_id_counter += len(processed_docs) - global_id_counter
                else:
                    new_doc["id"] = str(global_id_counter)
                    global_id_counter += 1
                    processed_docs.append(new_doc)

        if len(buffer_words) >= min_word_threshold:
            new_text = " ".join(buffer_words)
            merged_doc = {
                "text": new_text,
                "author": author,
                "genre": genre,
                "source_id": "+".join(buffer_ids)
            }
            if len(new_text.split()) > max_word_limit:
                processed_docs.extend(
                    split_long_doc(merged_doc, max_word_limit, global_id_counter)
                )
                global_id_counter += len(processed_docs) - global_id_counter
            else:
                merged_doc["id"] = str(global_id_counter)
                global_id_counter += 1
                processed_docs.append(merged_doc)

    return processed_docs

def split_long_doc(doc, max_word_limit, starting_id):
    """
    Splits a document into multiple docs by word count (no word cutting).
    Assigns new sequential numeric IDs starting from `starting_id`.
    Preserves the original document's ID in `source_id`.
    """
    words = doc["text"].split()
    chunks = [words[i:i + max_word_limit] for i in range(0, len(words), max_word_limit)]
    
    new_docs = []
    for i, chunk in enumerate(chunks):
        new_docs.append({
            "id": str(starting_id + i),
            "text": " ".join(chunk),
            "author": doc["author"],
            "genre": doc["genre"],
            "source_id": doc.get("source_id", doc.get("id", "unknown"))
        })
    return new_docs