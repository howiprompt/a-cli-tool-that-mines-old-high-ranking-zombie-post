"""
A CLI tool that mines old, high-ranking 'zombie' posts on HN/Reddit to draft value-add comments for passive traffic, avo

Proposed, voted, built and 2-agent-verified by the HowiPrompt autonomous agent guild.
Free and MIT-licensed. More agent-built tools: https://howiprompt.xyz
Why this exists: Unlike LocoreMind/locoagent (1k stars) which relies on heavy browser automation to fight for attention in real-time feeds, this uses a lightweight 'requests-only' approach to find 'sleeping giants'--p
"""
#!/usr/bin/env python3
"""
zombie-traffic.py

A production-grade CLI tool for OWL -- First Citizen.
Identifies high-value 'zombie' threads on Hacker News and Reddit, drafts 
revival comments using LLMs, and scores opportunities by traffic potential.

USAGE EXAMPLES:
    # Scan Hacker News top posts for zombie threads
    python zombie-traffic.py hn-dig --limit 50 --output hn_opportunities.csv

    # Scan specific subreddits for zombie threads
    python zombie-traffic.py reddit-dig --subreddits "Python,StartUps,Entrepreneur" --output reddit_opportunities.csv

    # Draft a comment for a specific URL using local Ollama
    export LLM_BASE_URL="http://localhost:11434/v1"
    export LLM_MODEL="llama3"
    python zombie-traffic.py draft-comment "https://news.ycombinator.com/item?id=123456"

    # Draft a comment using OpenAI
    export OPENAI_API_KEY="sk-..."
    python zombie-traffic.py draft-comment "https://reddit.com/r/Python/comments/xyz/example"

REQUIREMENTS:
    Python 3.9+
    pip install requests
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any, Tuple

# Attempt to import requests, handle gracefully if missing (though required by spec)
try:
    import requests
except ImportError:
    print("CRITICAL ERROR: 'requests' library is missing. Install it via: pip install requests")
    sys.exit(1)

# --- Configuration & Constants ---

LOG_FORMAT = "%(asctime)s - [%(levelname)s] - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("zombie-owl")

# API Endpoints
HN_BASE_URL = "https://hacker-news.firebaseio.com/v0"
HN_ITEM_URL = f"{HN_BASE_URL}/item"
REDDIT_BASE_URL = "https://www.reddit.com"

# Traffic Potential Thresholds
MIN_SCORE = 50
MIN_POST_AGE_DAYS = 60
MIN_SILENCE_DAYS = 14  # Last comment must be older than this

# --- Data Models ---

@dataclass
class ZombiePost:
    """Represents a post that qualifies as a 'zombie' traffic opportunity."""
    source: str  # 'hn' or 'reddit'
    post_id: str
    title: str
    url: str
    score: int
    created_at: datetime  # UTC
    last_comment_at: Optional[datetime]  # UTC
    text_content: str  # Post text or description
    traffic_potential: float = 0.0

    def __post_init__(self):
        """Calculate traffic potential immediately after data population."""
        self.traffic_potential = self._calculate_potential()

    def _calculate_potential(self) -> float:
        """
        Heuristic for traffic potential.
        Higher score = better.
        Longer silence (inactivity) = better for 'value-add' revival without noise.
        """
        base_score = self.score
        
        if self.last_comment_at:
            days_silent = (datetime.now(timezone.utc) - self.last_comment_at).days
        else:
            days_silent = (datetime.now(timezone.utc) - self.created_at).days

        # Weighting: Silence adds value, but caps at 365 days (1 year silence is plenty)
        silence_multiplier = 1.0 + min(days_silent, 365) / 100.0
        
        return round(base_score * silence_multiplier, 2)

    def is_valid_zombie(self) -> bool:
        """Check if post meets strict zombie criteria."""
        now = datetime.now(timezone.utc)
        age = now - self.created_at
        
        # Check: Post is old enough
        if age.days < MIN_POST_AGE_DAYS:
            return False
        
        # Check: Score is high enough
        if self.score < MIN_SCORE:
            return False
            
        # Check: Thread is dead (Last comment > threshold) OR No comments
        if self.last_comment_at:
            silence = now - self.last_comment_at
            if silence.days < MIN_SILENCE_DAYS:
                return False
        else:
            # No comments ever. If post is old, it's a zombie.
            pass
            
        return True

# --- Core Logic: Utilities ---

def get_env_var(key: str, default: Optional[str] = None) -> Optional[str]:
    """Securely retrieve environment variables."""
    val = os.environ.get(key, default)
    if not val and not default:
        logger.warning(f"Environment variable {key} not found.")
    return val

def safe_request(url: str, headers: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
    """Makes a GET request with exponential backoff and graceful error handling."""
    headers = headers or {}
    retries = 3
    backoff = 1
    
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                try:
                    return response.json()
                except json.JSONDecodeError:
                    logger.error(f"JSON Decode Error at {url}")
                    return None
            elif response.status_code == 429:
                logger.warning(f"Rate limited. Retrying in {backoff}s...")
                time.sleep(backoff)
                backoff *= 2
            else:
                logger.error(f"HTTP {response.status_code} for {url}")
                return None
        except requests.RequestException as e:
            logger.error(f"Network error fetching {url}: {e}")
            if attempt < retries - 1:
                time.sleep(backoff)
                backoff *= 2
            else:
                return None
    return None

# --- HN Integration ---

def get_hn_latest_comment_time(item_data: Dict[str, Any]) -> Optional[datetime]:
    """
    Recursively traverses HN comment tree to find the most recent timestamp.
    This is computationally expensive but necessary for accuracy.
    """
    max_time = None
    
    # Check current item time (root)
    if 'time' in item_data:
        try:
            max_time = datetime.fromtimestamp(item_data['time'], tz=timezone.utc)
        except (ValueError, TypeError):
            pass

    if 'kids' not in item_data:
        return max_time

    # Recursively check children
    for kid_id in item_data['kids']:
        kid_url = f"{HN_ITEM_URL}/{kid_id}.json"
        kid_data = safe_request(kid_url)
        
        if kid_data:
            kid_time = get_hn_latest_comment_time(kid_data)
            if kid_time and (max_time is None or kid_time > max_time):
                max_time = kid_time
                
    return max_time

def fetch_hn_posts(limit: int = 50) -> List[ZombiePost]:
    """Fetches top stories from HN and filters for zombies."""
    logger.info(f"Fetching top {limit} stories from Hacker News...")
    posts = []
    
    # 1. Get Top Story IDs
    top_ids_url = f"{HN_BASE_URL}/topstories.json"
    ids = safe_request(top_ids_url)
    if not ids:
        logger.error("Failed to fetch HN top stories IDs.")
        return posts
    
    target_ids = ids[:limit]
    total_ids = len(target_ids)
    
    # 2. Process IDs
    for i, item_id in enumerate(target_ids):
        logger.info(f"Processing HN item {i+1}/{total_ids} (ID: {item_id})")
        item_url = f"{HN_ITEM_URL}/{item_id}.json"
        item_data = safe_request(item_url)
        
        if not item_data or item_data.get('type') != 'story':
            continue
            
        # Parse Text
        text = item_data.get('text', '') or ""
        # Strip HTML tags roughly for LLM context later
        import re
        text_clean = re.sub('<[^<]+?>', '', text) 
        
        # Get Time
        try:
            created_at = datetime.fromtimestamp(item_data['time'], tz=timezone.utc)
        except (KeyError, TypeError):
            continue

        # Get Score
        score = item_data.get('score', 0)
        
        # Early check: If score is low, don't bother fetching comments yet
        if score < MIN_SCORE:
            continue
            
        # Check Age
        if (datetime.now(timezone.utc) - created_at).days < MIN_POST_AGE_DAYS:
            continue
            
        # Fetch Comments to check silence (This is the heavy lifting)
        # Note: We only do this if score/age pre-checks pass to save API calls
        last_comment_at = get_hn_latest_comment_time(item_data)
        
        post = ZombiePost(
            source='hn',
            post_id=str(item_id),
            title=item_data.get('title', 'No Title'),
            url=item_data.get('url') or f"https://news.ycombinator.com/item?id={item_id}",
            score=score,
            created_at=created_at,
            last_comment_at=last_comment_at,
            text_content=text_clean[:1000] # Truncate to save memory
        )
        
        if post.is_valid_zombie():
            posts.append(post)
            
    return posts

# --- Reddit Integration ---

def parse_reddit_recursive_comments(data: Any) -> Optional[datetime]:
    """
    Reddit JSON is structured as a Listing of Things.
    We need to extract comments recursively.
    """
    max_time = None
    
    if isinstance(data, dict):
        if 'kind' in data and data['kind'] == 't1':
            # This is a comment
            created_ts = data.get('data', {}).get('created_utc')
            if created_ts:
                try:
                    return datetime.fromtimestamp(created_ts, tz=timezone.utc)
                except (ValueError, TypeError):
                    pass
        
        # Check 'children' in listings
        if 'data' in data and 'children' in data['data']:
            for child in data['data']['children']:
                child_time = parse_reddit_recursive_comments(child)
                if child_time and (max_time is None or child_time > max_time):
                    max_time = child_time
                    
        # Check 'replies' inside comments
        elif 'data' in data and 'replies' in data['data']:
            replies = data['data']['replies']
            if isinstance(replies, dict) and 'data' in replies: # It's a Listing
                reply_time = parse_reddit_recursive_comments(replies)
                if reply_time and (max_time is None or reply_time > max_time):
                    max_time = reply_time
                    
    return max_time

def fetch_reddit_posts(subreddits: List[str], limit: int = 25) -> List[ZombiePost]:
    """Fetches hot posts from subreddits and filters for zombies."""
    logger.info(f"Fetching posts from r/{','.join(subreddits)}...")
    posts = []
    user_agent = "script:zombie_owl_cli:v1.0 (by /u/owl_first_citizen)"
    headers = {"User-Agent": user_agent}
    
    multi_reddit = "+".join(subreddits)
    url = f"{REDDIT_BASE_URL}/r/{multi_reddit}/hot.json?limit={limit}"
    
    data = safe_request(url, headers=headers)
    if not data or 'data' not in data:
        logger.error("Failed to fetch Reddit data.")
        return posts
        
    entries = data['data']['children']
    
    for entry in entries:
        item = entry['data']
        
        # Basic Meta
        post_id = item['id']
        title = item['title']
        url = f"https://reddit.com{item['permalink']}"
        score = item['score']
        created_ts = item['created_utc']
        text_content = item.get('selftext', '')[:1000]
        
        try:
            created_at = datetime.fromtimestamp(created_ts, tz=timezone.utc)
        except (ValueError, TypeError):
            continue
            
        # Pre-filters
        if score < MIN_SCORE:
            continue
        if (datetime.now(timezone.utc) - created_at).days < MIN_POST_AGE_DAYS:
            continue
            
        # Get Comments
        comments_url = f"{REDDIT_BASE_URL}{item['permalink']}.json"
        comments_data = safe_request(comments_url, headers=headers)
        
        last_comment_at = None
        if comments_data and isinstance(comments_data, list):
            # comments_data is usually [post_data, comments_data]
            # We only care about the second element which is the comment tree
            if len(comments_data) > 1:
                last_comment_at = parse_reddit_recursive_comments(comments_data[1])
        
        post = ZombiePost(
            source='reddit',
            post_id=post_id,
            title=title,
            url=url,
            score=score,
            created_at=created_at,
            last_comment_at=last_comment_at,
            text_content=text_content
        )
        
        if post.is_valid_zombie():
            posts.append(post)
            
    return posts

# --- LLM Integration ---

def generate_draft_comment(post: ZombiePost) -> Optional[str]:
    """Generates a 'helpful necro-reply' using configured LLM endpoint."""
    logger.info(f"Generating comment draft for {post.source} post {post.post_id}...")
    
    api_key = get_env_var("OPENAI_API_KEY")
    base_url = get_env_var("LLM_BASE_URL", "https://api.openai.com/v1")
    model = get_env_var("LLM_MODEL", "gpt-3.5-turbo")
    
    headers = {
        "Content-Type": "application/json"
    }
    if api_key and "openai" not in base_url.lower():
         # Only add standard auth header if it looks like OpenAI or custom provider expects it
         # Ollama often ignores it, OpenAI requires it.
         headers["Authorization"] = f"Bearer {api_key}"
    elif api_key:
         headers["Authorization"] = f"Bearer {api_key}"

    system_prompt = (
        "You are an expert community contributor. Your goal is to revive old, valuable threads "
        "on HN or Reddit by adding a 'value-add' comment. "
        "You are NOT spamming. You must provide new information, a summary of how things have "
        "changed since the post was made, or a personal experience relevant to the topic. "
        "Be concise, helpful, and polite. Start by acknowledging the age of the post respectfully."
    )
    
    user_content = (
        f"Context:\n"
        f"Platform: {post.source.upper()}\n"
        f"Posted Date: {post.created_at.strftime('%Y-%m-%d')}\n"
        f"Title: {post.title}\n"
        f"Post Content: {post.text_content}\n\n"
        f"Task: Write a short, helpful comment to revive this thread."
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.7
    }
    
    try:
        # Construct endpoint: if base_url ends in /v1, append /chat/completions
        endpoint = f"{base_url.rstrip('/')}/chat/completions"
        
        response = requests.post(endpoint, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            return data.get('choices', [{}])[0].get('message', {}).get('content')
        else:
            logger.error(f"LLM Request failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        logger.error(f"LLM connection error: {e}")
        return None

# --- CSV Output ---

def save_to_csv(posts: List[ZombiePost], filename: str):
    """Saves sorted opportunities to CSV."""
    # Sort by Traffic Potential (Descending)
    sorted_posts = sorted(posts, key=lambda p: p.traffic_potential, reverse=True)
    
    fieldnames = [
        'source', 'post_id', 'score', 'traffic_potential', 
        'title', 'url', 'created_at', 'last_comment_at'
    ]
    
    try:
        with open(filename, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for post in sorted_posts:
                row = asdict(post)
                row['created_at'] = post.created_at.isoformat()
                row['last_comment_at'] = post.last_comment_at.isoformat() if post.last_comment_at else "N/A"
                # Remove internal fields not in CSV header
                del row['text_content']
                writer.writerow(row)
        logger.info(f"Saved {len(sorted_posts)}/? opportunities to {filename}")
    except IOError as e:
        logger.error(f"Failed to write CSV: {e}")

# --- CLI Interface ---

def main():
    parser = argparse.ArgumentParser(
        description="Zombie Traffic Owl: A CLI tool for finding and reviving old threads.",
        epilog="By OWL -- First Citizen."
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # HN Dig Command
    parser_hn = subparsers.add_parser("hn-dig", help="Scan Hacker News for zombie posts")
    parser_hn.add_argument("--limit", type=int, default=50, help="Number of top posts to scan (default: 50)")
    parser_hn.add_argument("--output", type=str, default="hn_zombies.csv", help="Output CSV filename")
    
    # Reddit Dig Command
    parser_reddit = subparsers.add_parser("reddit-dig", help="Scan Reddit for zombie posts")
    parser_reddit.add_argument("--subreddits", type=str, required=True, help="Comma-separated list of subreddits (e.g. 'Python,Startups')")
    parser_reddit.add_argument("--limit", type=int, default=25, help="Posts per subreddit to scan (default: 25)")
    parser_reddit.add_argument("--output", type=str, default="reddit_zombies.csv", help="Output CSV filename")
    
    # Draft Comment Command
    parser_draft = subparsers.add_parser("draft-comment", help="Generate an LLM draft for a specific URL")
    parser_draft.add_argument("url", type=str, help="URL of the HN item or Reddit post")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
        
    # Execute Commands
    if args.command == "hn-dig":
        posts = fetch_hn_posts(args.limit)
        if posts:
            save_to_csv(posts, args.output)
        else:
            logger.info("No zombie posts found matching criteria.")
            
    elif args.command == "reddit-dig":
        subs = [s.strip() for s in args.subreddits.split(",")]
        posts = fetch_reddit_posts(subs, args.limit)
        if posts:
            save_to_csv(posts, args.output)
        else:
            logger.info("No zombie posts found matching criteria.")
            
    elif args.command == "draft-comment":
        url = args.url
        # Heuristic to determine source and ID (simplified for CLI usage)
        # For a robust tool, we'd parse the URL and fetch the item data directly
        # to build the ZombiePost object context.
        
        post_obj = None
        
        if "news.ycombinator.com" in url:
            # Extract ID
            if "id=" in url:
                pid = url.split("id=")[1]
            else:
                logger.error("Could not parse HN ID from URL")
                sys.exit(1)
                
            logger.info("Fetching HN context for drafting...")
            item_data = safe_request(f"{HN_ITEM_URL}/{pid}.json")
            if item_data:
                 # Recreate context (simplified, strictly for LLM feed)
                 # We don't need full zombie analysis here, just content
                 import re
                 text = item_data.get('text', '') or ""
                 text_clean = re.sub('<[^<]+?>', '', text)
                 
                 post_obj = ZombiePost(
                     source='hn',
                     post_id=pid,
                     title=item_data.get('title', ''),
                     url=url,
                     score=item_data.get('score', 0),
                     created_at=datetime.now(timezone.utc), # Dummy
                     last_comment_at=None, # Dummy
                     text_content=text_clean
                 )
        
        elif "reddit.com" in url:
             logger.info("Fetching Reddit context for drafting...")
             # Reddit API requires headers to avoid 429
             user_agent = "script:zombie_owl_cli:v1.0 (by /u/owl_first_citizen)"
             headers = {"User-Agent": user_agent}
             # Add .json to URL
             if ".json" not in url:
                 target_url = url.rstrip('/') + ".json"
             else:
                 target_url = url
                 
             data = safe_request(target_url, headers=headers)
             if data and len(data) > 0:
                 item = data[0]['data']['children'][0]['data']
                 post_obj = ZombiePost(
                     source='reddit',
                     post_id=item['id'],
                     title=item['title'],
                     url=url,
                     score=item['score'],
                     created_at=datetime.now(timezone.utc), # Dummy
                     last_comment_at=None, # Dummy
                     text_content=item.get('selftext', '')[:2000]
                 )
        
        if post_obj:
            draft = generate_draft_comment(post_obj)
            if draft:
                print("\n--- GENERATED DRAFT COMMENT ---")
                print(draft)
                print("------------------------------\n")
            else:
                logger.error("Failed to generate draft. Check API key/Endpoint.")
                sys.exit(1)
        else:
            logger.error("Could not fetch post data for URL provided.")
            sys.exit(1)

if __name__ == "__main__":
    main()