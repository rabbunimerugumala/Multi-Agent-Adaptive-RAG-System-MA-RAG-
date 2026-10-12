"""Dataset loader and text chunker for HotpotQA.

Loads multi-hop question-answering examples from HuggingFace HotpotQA
(distractor setting), chunks context paragraphs with configured overlap,
and provides caching for offline and fast local execution.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from config import config, setup_logger

logger = setup_logger("data_loader")

# Fallback curated HotpotQA multi-hop questions and contexts
# for instant offline development or when HuggingFace is unreachable.
CURATED_HOTPOTQA_FALLBACK: List[Dict[str, Any]] = [
    {
        "id": "5a7a421b55429930f4000010",
        "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
        "answer": "yes",
        "type": "comparison",
        "level": "medium",
        "context": {
            "title": ["Scott Derrickson", "Ed Wood", "Doctor Strange (film)"],
            "sentences": [
                [
                    "Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer.",
                    "He lives in Los Angeles, California.",
                    "He is best known for directing horror films such as Sinister and Deliver Us from Evil.",
                ],
                [
                    "Edward Davis Wood Jr. (October 10, 1924 – December 10, 1978) was an American filmmaker, actor, and pulp novel author.",
                    "In the 1950s, Wood directed several low-budget science fiction and horror films.",
                    "He was posthumously awarded a Golden Turkey Award for Worst Director of All Time.",
                ],
                [
                    "Doctor Strange is a 2016 American superhero film based on the Marvel Comics character of the same name.",
                    "Produced by Marvel Studios and distributed by Walt Disney Studios Motion Pictures.",
                    "It was directed by Scott Derrickson from a screenplay he wrote with Jon Spaihts and C. Robert Cargill.",
                ],
            ],
        },
    },
    {
        "id": "5a8c2d1b55429955e80000a2",
        "question": "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?",
        "answer": "Chief of Protocol of the United States",
        "type": "bridge",
        "level": "medium",
        "context": {
            "title": ["Kiss and Tell (1945 film)", "Shirley Temple"],
            "sentences": [
                [
                    "Kiss and Tell is a 1945 American comedy film starring Shirley Temple as Corliss Archer.",
                    "In the film, two teenage girls cause confusion when they keep secrets about romantic escapades.",
                    "The film was directed by Richard Wallace.",
                ],
                [
                    "Shirley Temple Black (April 23, 1928 – February 10, 2014) was an American actress, singer, dancer, and diplomat.",
                    "She began her diplomatic career in 1969, representing the United States at the United Nations General Assembly.",
                    "Later, she served as the Chief of Protocol of the United States from 1976 to 1977 and U.S. Ambassador to Ghana and Czechoslovakia.",
                ],
            ],
        },
    },
    {
        "id": "5a8286eb55429910d6000088",
        "question": "What magazine did the co-founder of the production company behind 'The Crown' previously edit?",
        "answer": "Prospect",
        "type": "bridge",
        "level": "hard",
        "context": {
            "title": ["The Crown (TV series)", "Left Bank Pictures", "David Aukin"],
            "sentences": [
                [
                    "The Crown is a historical drama television series about the reign of Queen Elizabeth II.",
                    "It was created and principally written by Peter Morgan.",
                    "The series was produced by Left Bank Pictures and Sony Pictures Television for Netflix.",
                ],
                [
                    "Left Bank Pictures is a British film and television production company formed in 2007 by Andy Harries and David Aukin.",
                    "Before co-founding Left Bank, Andy Harries was Controller of Drama at Granada Television.",
                    "David Aukin had previously been Head of Drama at Channel 4 and co-founder of Prospect magazine.",
                ],
                [
                    "Prospect is a British monthly current affairs magazine that specializes in politics, culture, and science.",
                    "It was founded in October 1995 by David Goodhart and David Aukin.",
                ],
            ],
        },
    },
    {
        "id": "5a8e0f9b55429969d2000045",
        "question": "Which band has more members: Radiohead or The Police?",
        "answer": "Radiohead",
        "type": "comparison",
        "level": "easy",
        "context": {
            "title": ["Radiohead", "The Police"],
            "sentences": [
                [
                    "Radiohead are an English rock band formed in Abingdon, Oxfordshire, in 1985.",
                    "The band consists of five members: Thom Yorke, brothers Jonny Greenwood and Colin Greenwood, Ed O'Brien, and Philip Selway.",
                    "They have sold more than 30 million albums worldwide.",
                ],
                [
                    "The Police were an English rock band formed in London in 1977.",
                    "The Police were formed by Sting, Andy Summers, and Stewart Copeland.",
                    "The trio became globally popular in the late 1970s and early 1980s.",
                ],
            ],
        },
    },
    {
        "id": "5a7f92025542992850000012",
        "question": "What is the capital of the country where the director of 'Pan's Labyrinth' was born?",
        "answer": "Mexico City",
        "type": "bridge",
        "level": "medium",
        "context": {
            "title": ["Pan's Labyrinth", "Guillermo del Toro", "Mexico"],
            "sentences": [
                [
                    "Pan's Labyrinth is a 2006 dark fantasy film written and directed by Guillermo del Toro.",
                    "The film takes place in Spain during the summer of 1944, five years after the Spanish Civil War.",
                    "It was produced by Guillermo del Toro, Alfonso Cuarón, and Bertha Navarro.",
                ],
                [
                    "Guillermo del Toro Gómez (born October 9, 1964) is a Mexican filmmaker and author.",
                    "He was born in Guadalajara, Jalisco, Mexico.",
                    "Throughout his career, del Toro has alternated between Spanish-language dark fantasy pieces and mainstream American movies.",
                ],
                [
                    "Mexico, officially the United Mexican States, is a country in the southern portion of North America.",
                    "The capital and largest city of Mexico is Mexico City.",
                ],
            ],
        },
    },
    {
        "id": "5a7a421b55429930f4000015",
        "question": "Which film came out earlier: Kiss and Tell or Doctor Strange?",
        "answer": "Kiss and Tell",
        "type": "comparison",
        "level": "easy",
        "context": {
            "title": ["Kiss and Tell (1945 film)", "Doctor Strange (film)"],
            "sentences": [
                [
                    "Kiss and Tell is a 1945 American comedy film starring Shirley Temple as Corliss Archer.",
                ],
                [
                    "Doctor Strange is a 2016 American superhero film directed by Scott Derrickson.",
                ],
            ],
        },
    },
    {
        "id": "5a7a421b55429930f4000016",
        "question": "What nationality is the director of the superhero film Doctor Strange?",
        "answer": "American",
        "type": "bridge",
        "level": "medium",
        "context": {
            "title": ["Doctor Strange (film)", "Scott Derrickson"],
            "sentences": [
                [
                    "Doctor Strange is a 2016 American superhero film directed by Scott Derrickson.",
                ],
                [
                    "Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer.",
                ],
            ],
        },
    },
    {
        "id": "5a7a421b55429930f4000017",
        "question": "Were Guillermo del Toro and Scott Derrickson born in the same continent?",
        "answer": "yes",
        "type": "comparison",
        "level": "medium",
        "context": {
            "title": ["Guillermo del Toro", "Scott Derrickson"],
            "sentences": [
                [
                    "Guillermo del Toro was born in Guadalajara, Jalisco, Mexico in North America.",
                ],
                [
                    "Scott Derrickson is an American filmmaker born in Denver, Colorado in North America.",
                ],
            ],
        },
    },
    {
        "id": "5a7a421b55429930f4000018",
        "question": "Who is the lead singer of the English band formed in 1985 that sold over 30 million albums?",
        "answer": "Thom Yorke",
        "type": "bridge",
        "level": "hard",
        "context": {
            "title": ["Radiohead", "Thom Yorke"],
            "sentences": [
                [
                    "Radiohead are an English rock band formed in Abingdon in 1985 that sold over 30 million albums.",
                    "The lead singer is Thom Yorke.",
                ],
            ],
        },
    },
    {
        "id": "5a7a421b55429930f4000019",
        "question": "Which English rock band was formed first, The Police or Radiohead?",
        "answer": "The Police",
        "type": "comparison",
        "level": "easy",
        "context": {
            "title": ["The Police", "Radiohead"],
            "sentences": [
                [
                    "The Police were an English rock band formed in London in 1977.",
                ],
                [
                    "Radiohead are an English rock band formed in Abingdon in 1985.",
                ],
            ],
        },
    },
    {
        "id": "5a7a421b55429930f4000020",
        "question": "In what city was the director of Pan's Labyrinth born?",
        "answer": "Guadalajara",
        "type": "bridge",
        "level": "medium",
        "context": {
            "title": ["Pan's Labyrinth", "Guillermo del Toro"],
            "sentences": [
                [
                    "Pan's Labyrinth is a dark fantasy film written and directed by Guillermo del Toro.",
                ],
                [
                    "Guillermo del Toro was born in Guadalajara, Jalisco, Mexico.",
                ],
            ],
        },
    },
    {
        "id": "5a7a421b55429930f4000021",
        "question": "What award did the director of Plan 9 from Outer Space posthumously receive?",
        "answer": "Golden Turkey Award",
        "type": "bridge",
        "level": "medium",
        "context": {
            "title": ["Ed Wood"],
            "sentences": [
                [
                    "Edward Davis Wood Jr. directed Plan 9 from Outer Space and low-budget science fiction films.",
                    "He was posthumously awarded a Golden Turkey Award for Worst Director of All Time.",
                ],
            ],
        },
    },
    {
        "id": "5a7a421b55429930f4000022",
        "question": "Did David Aukin co-found the production company responsible for The Crown?",
        "answer": "yes",
        "type": "bridge",
        "level": "easy",
        "context": {
            "title": ["The Crown (TV series)", "Left Bank Pictures"],
            "sentences": [
                [
                    "The Crown was produced by Left Bank Pictures and Sony Pictures Television for Netflix.",
                ],
                [
                    "Left Bank Pictures is a British film and television production company formed by Andy Harries and David Aukin.",
                ],
            ],
        },
    },
]


@dataclass
class DocumentChunk:
    """Represents a chunked text unit ready for vector store insertion."""

    chunk_id: str
    doc_title: str
    text: str
    metadata: Dict[str, Any]


def chunk_text(
    text: str, chunk_size: int = 500, overlap: int = 50
) -> List[str]:
    """Splits text into overlapping chunks by word/token boundaries.

    Args:
        text: Input string.
        chunk_size: Maximum approximate token/word length.
        overlap: Token/word overlap between consecutive chunks.

    Returns:
        List[str]: Clean text chunks.
    """
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    start = 0
    stride = max(1, chunk_size - overlap)

    while start < len(words):
        chunk_words = words[start : start + chunk_size]
        chunk_str = " ".join(chunk_words)
        if chunk_str.strip():
            chunks.append(chunk_str.strip())
        start += stride

    return chunks


class HotpotQADataLoader:
    """Loader and transformer for HotpotQA dataset."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initializes data loader with optional cache directory."""
        self.cache_dir = cache_dir or Path(config.chroma_persist_directory).parent
        self.cache_file = self.cache_dir / "hotpotqa_cache.json"

    def load_dataset(self, num_samples: int = 200) -> List[Dict[str, Any]]:
        """Loads HotpotQA distractor validation split from HuggingFace or local cache.

        Args:
            num_samples: Target sample limit.

        Returns:
            List[Dict[str, Any]]: Formatted question-answer records.
        """
        # 1. Try local cached file first
        if self.cache_file.exists():
            try:
                logger.info(f"Loading HotpotQA dataset from local cache: {self.cache_file}")
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                    if isinstance(cached_data, list) and len(cached_data) >= min(10, num_samples):
                        return cached_data[:num_samples]
            except Exception as e:
                logger.warning(f"Failed to read dataset cache: {e}")

        # 2. Try loading from HuggingFace datasets library
        try:
            logger.info("Downloading/loading HotpotQA from HuggingFace datasets...")
            from datasets import load_dataset

            split_str = f"validation[:{num_samples}]"
            hf_dataset = load_dataset(
                config.dataset_name,
                config.dataset_config,
                split=split_str,
                trust_remote_code=True,
            )

            records: List[Dict[str, Any]] = []
            for item in hf_dataset:
                records.append(
                    {
                        "id": str(item.get("id", "")),
                        "question": str(item.get("question", "")),
                        "answer": str(item.get("answer", "")),
                        "type": str(item.get("type", "unknown")),
                        "level": str(item.get("level", "medium")),
                        "context": item.get("context", {}),
                    }
                )

            # Save to local cache
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)

            logger.info(f"Loaded {len(records)} HotpotQA samples from HuggingFace.")
            return records

        except Exception as e:
            logger.warning(
                f"Could not load HotpotQA from HuggingFace ({e}). Using curated multi-hop fallback data."
            )
            # Use curated multi-hop fallback
            return CURATED_HOTPOTQA_FALLBACK

    def prepare_chunks(
        self, samples: List[Dict[str, Any]]
    ) -> List[DocumentChunk]:
        """Extracts and chunks all context paragraphs from samples.

        Args:
            samples: List of HotpotQA questions with context.

        Returns:
            List[DocumentChunk]: Indexed chunks ready for ChromaDB.
        """
        chunks: List[DocumentChunk] = []
        seen_titles = set()

        for sample in samples:
            ctx = sample.get("context", {})
            titles = ctx.get("title", [])
            sentences_list = ctx.get("sentences", [])

            for title, sentences in zip(titles, sentences_list):
                if title in seen_titles:
                    continue
                seen_titles.add(title)

                full_text = " ".join(sentences)
                text_splits = chunk_text(
                    full_text,
                    chunk_size=config.chunk_size,
                    overlap=config.chunk_overlap,
                )

                for idx, split in enumerate(text_splits):
                    chunk_id = f"{title}_{idx}".replace(" ", "_")
                    chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_id,
                            doc_title=title,
                            text=split,
                            metadata={
                                "title": title,
                                "chunk_index": idx,
                                "source": "HotpotQA",
                            },
                        )
                    )

        logger.info(f"Generated {len(chunks)} document chunks from {len(seen_titles)} articles.")
        return chunks

    @staticmethod
    def get_sample_questions() -> List[Dict[str, str]]:
        """Returns 5 prominent sample multi-hop questions for the UI sidebar."""
        return [
            {
                "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
                "type": "Comparison",
                "expected": "yes",
            },
            {
                "question": "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?",
                "type": "Bridge",
                "expected": "Chief of Protocol of the United States",
            },
            {
                "question": "What magazine did the co-founder of the production company behind 'The Crown' previously edit?",
                "type": "Bridge",
                "expected": "Prospect",
            },
            {
                "question": "Which band has more members: Radiohead or The Police?",
                "type": "Comparison",
                "expected": "Radiohead",
            },
            {
                "question": "What is the capital of the country where the director of 'Pan's Labyrinth' was born?",
                "type": "Bridge",
                "expected": "Mexico City",
            },
        ]

    @staticmethod
    def get_all_questions() -> List[Dict[str, Any]]:
        """Returns all available HotpotQA benchmark questions from cache or curated records."""
        questions: List[Dict[str, Any]] = []
        for idx, item in enumerate(CURATED_HOTPOTQA_FALLBACK):
            q_type = str(item.get("type", "bridge")).capitalize()
            level = str(item.get("level", "medium")).capitalize()
            questions.append(
                {
                    "id": item.get("id", f"q_{idx+1}"),
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                    "type": q_type,
                    "level": level,
                    "summary": f"[{q_type} | {level}] {item.get('question', '')}",
                }
            )
        return questions
