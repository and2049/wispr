from __future__ import annotations

import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from functools import lru_cache

from wispr.models import AlignedWord

FUZZY_MATCH_THRESHOLD = 0.84
INSERT_DELETE_COST = 1.0
MISMATCH_COST = 2.0


@dataclass(frozen=True)
class MatchingConfig:
    min_line_coverage: float = 0.45
    min_total_coverage: float = 0.35
    search_window: int = 96
    max_candidate_starts: int = 48
    neighbor_window: int = 24


@dataclass(frozen=True)
class WordPiece:
    token: str
    word_index: int
    word: AlignedWord


@dataclass(frozen=True)
class CanonicalToken:
    token: str
    line_number: int


@dataclass(frozen=True)
class TokenMatch:
    canonical_index: int
    piece_index: int
    similarity: float


@dataclass(frozen=True)
class LineMatch:
    line_number: int
    token_count: int
    matches: tuple[TokenMatch, ...] = ()
    score: float = 0.0
    strategy: str = "structured"

    @property
    def coverage(self) -> float:
        return len(self.matches) / self.token_count if self.token_count else 1.0

    @property
    def piece_indices(self) -> tuple[int, ...]:
        return tuple(match.piece_index for match in self.matches)


@dataclass(frozen=True)
class MatchingDiagnostics:
    strategy: str = "structured"
    fallback_used: bool = False
    anchor_count: int = 0
    repeated_line_count: int = 0
    ambiguous_line_count: int = 0
    line_match_scores: dict[int, float] = field(default_factory=dict)


@dataclass(frozen=True)
class MatchingResult:
    line_matches: tuple[LineMatch, ...]
    diagnostics: MatchingDiagnostics


def match_lyrics_to_words(
    lyrics: tuple[str, ...],
    words: tuple[AlignedWord, ...],
    config: MatchingConfig | None = None,
) -> MatchingResult:
    config = config or MatchingConfig()
    pieces = aligned_word_pieces(words)
    if not lyrics:
        return MatchingResult((), MatchingDiagnostics())

    structured = structured_line_matching(lyrics, pieces, config)
    global_result = global_line_matching(lyrics, pieces)
    if should_use_global_fallback(structured, global_result, config):
        return MatchingResult(
            global_result.line_matches,
            MatchingDiagnostics(
                strategy="global",
                fallback_used=True,
                anchor_count=structured.diagnostics.anchor_count,
                repeated_line_count=structured.diagnostics.repeated_line_count,
                ambiguous_line_count=structured.diagnostics.ambiguous_line_count,
                line_match_scores=global_result.diagnostics.line_match_scores,
            ),
        )
    return structured


def structured_line_matching(
    lyrics: tuple[str, ...],
    pieces: tuple[WordPiece, ...],
    config: MatchingConfig,
) -> MatchingResult:
    line_tokens = tuple(lyric_tokens(line) for line in lyrics)
    token_counts = Counter(token for tokens in line_tokens for token in tokens)
    line_counts = Counter(line_signature(tokens) for tokens in line_tokens if tokens)
    repeated_line_count = sum(count for count in line_counts.values() if count > 1)
    anchor_count = sum(1 for count in token_counts.values() if count == 1)
    cursor = 0
    matches: list[LineMatch] = []
    ambiguous = 0

    for index, tokens in enumerate(line_tokens):
        line_number = index + 1
        if not tokens:
            matches.append(LineMatch(line_number=line_number, token_count=0, score=1.0))
            continue
        candidates = line_candidates(
            tokens,
            pieces,
            cursor,
            config,
            token_counts,
            next_tokens(line_tokens, index),
        )
        strong_candidates = [
            candidate for candidate in candidates if candidate.score >= config.min_line_coverage
        ]
        if len(strong_candidates) > 1:
            ambiguous += 1
        selected = candidates[0] if candidates else None
        if selected is None or selected.coverage < config.min_line_coverage:
            matches.append(LineMatch(line_number=line_number, token_count=len(tokens), score=0.0))
            continue
        selected = LineMatch(
            line_number=line_number,
            token_count=selected.token_count,
            matches=selected.matches,
            score=selected.score,
            strategy=selected.strategy,
        )
        matches.append(selected)
        if selected.piece_indices:
            cursor = max(selected.piece_indices) + 1

    diagnostics = MatchingDiagnostics(
        strategy="structured",
        anchor_count=anchor_count,
        repeated_line_count=repeated_line_count,
        ambiguous_line_count=ambiguous,
        line_match_scores={match.line_number: round(match.score, 4) for match in matches},
    )
    return MatchingResult(tuple(matches), diagnostics)


def line_candidates(
    tokens: tuple[str, ...],
    pieces: tuple[WordPiece, ...],
    cursor: int,
    config: MatchingConfig,
    token_counts: Counter[str],
    following_tokens: tuple[str, ...],
) -> tuple[LineMatch, ...]:
    starts = candidate_starts(tokens, pieces, cursor, config)
    candidates = tuple(
        candidate
        for candidate in (
            match_line_from_start(
                tokens,
                pieces,
                start,
                cursor,
                config,
                token_counts,
                following_tokens,
            )
            for start in starts
        )
        if candidate.matches
    )
    return tuple(sorted(candidates, key=lambda item: item.score, reverse=True))


def candidate_starts(
    tokens: tuple[str, ...],
    pieces: tuple[WordPiece, ...],
    cursor: int,
    config: MatchingConfig,
) -> tuple[int, ...]:
    starts = [cursor]
    end = min(len(pieces), cursor + config.search_window)
    for piece_index in range(cursor, end):
        has_match = any(
            token_similarity(token, pieces[piece_index].token) >= FUZZY_MATCH_THRESHOLD
            for token in tokens
        )
        if has_match:
            starts.append(piece_index)
        if len(starts) >= config.max_candidate_starts:
            break
    return tuple(unique(starts))


def match_line_from_start(
    tokens: tuple[str, ...],
    pieces: tuple[WordPiece, ...],
    start: int,
    cursor: int,
    config: MatchingConfig,
    token_counts: Counter[str],
    following_tokens: tuple[str, ...],
) -> LineMatch:
    end = min(len(pieces), start + max(config.search_window, len(tokens) * 8))
    position = start
    matches: list[TokenMatch] = []
    for token_index, token in enumerate(tokens):
        match = next_piece_match(token, pieces, position, end)
        if match is None:
            continue
        piece_index, similarity = match
        matches.append(TokenMatch(token_index, piece_index, similarity))
        position = piece_index + 1
    if not matches:
        return LineMatch(line_number=0, token_count=len(tokens), score=0.0)

    indices = tuple(match.piece_index for match in matches)
    span = max(indices) - min(indices) + 1
    extra_ratio = max(span - len(matches), 0) / max(span, 1)
    coverage = len(matches) / len(tokens)
    similarity = sum(match.similarity for match in matches) / len(matches)
    confidence = sum(pieces[index].word.confidence for index in indices) / len(indices)
    cursor_penalty = min(max(min(indices) - cursor, 0) / max(config.search_window, 1), 1.0)
    rare_bonus = rare_anchor_bonus(tokens, matches, pieces, token_counts)
    neighbor_bonus = following_context_bonus(following_tokens, pieces, max(indices) + 1, config)
    score = (
        coverage * 0.44
        + similarity * 0.2
        + confidence * 0.16
        + rare_bonus * 0.1
        + neighbor_bonus * 0.1
        - extra_ratio * 0.12
        - cursor_penalty * 0.18
    )
    return LineMatch(
        line_number=0,
        token_count=len(tokens),
        matches=tuple(matches),
        score=max(round(score, 4), 0.0),
    )


def next_piece_match(
    token: str,
    pieces: tuple[WordPiece, ...],
    start: int,
    end: int,
) -> tuple[int, float] | None:
    best: tuple[int, float] | None = None
    for index in range(start, end):
        similarity = token_similarity(token, pieces[index].token)
        if similarity < FUZZY_MATCH_THRESHOLD:
            continue
        if similarity == 1.0:
            return index, similarity
        if best is None or similarity > best[1]:
            best = (index, similarity)
    return best


def rare_anchor_bonus(
    tokens: tuple[str, ...],
    matches: list[TokenMatch],
    pieces: tuple[WordPiece, ...],
    token_counts: Counter[str],
) -> float:
    rare = [token for token in tokens if token_counts[token] == 1]
    if not rare:
        return 0.0
    matched = sum(
        token_counts[tokens[match.canonical_index]] == 1
        and tokens[match.canonical_index] == pieces[match.piece_index].token
        for match in matches
    )
    return matched / len(rare)


def following_context_bonus(
    tokens: tuple[str, ...],
    pieces: tuple[WordPiece, ...],
    cursor: int,
    config: MatchingConfig,
) -> float:
    if not tokens:
        return 0.0
    end = min(len(pieces), cursor + config.neighbor_window)
    seen = 0
    position = cursor
    for token in tokens[:3]:
        match = next_piece_match(token, pieces, position, end)
        if match is None:
            continue
        position = match[0] + 1
        seen += 1
    return seen / min(len(tokens), 3)


def next_tokens(line_tokens: tuple[tuple[str, ...], ...], index: int) -> tuple[str, ...]:
    for tokens in line_tokens[index + 1 :]:
        if tokens:
            return tokens
    return ()


def should_use_global_fallback(
    structured: MatchingResult,
    global_result: MatchingResult,
    config: MatchingConfig,
) -> bool:
    structured_coverage = result_coverage(structured)
    global_coverage = result_coverage(global_result)
    return (
        structured_coverage < config.min_total_coverage
        and global_coverage > structured_coverage
    )


def result_coverage(result: MatchingResult) -> float:
    expected = sum(match.token_count for match in result.line_matches)
    matched = sum(len(match.matches) for match in result.line_matches)
    return matched / expected if expected else 1.0


def global_line_matching(
    lyrics: tuple[str, ...],
    pieces: tuple[WordPiece, ...],
) -> MatchingResult:
    canonical = canonical_tokens(lyrics)
    matches, fuzzy = global_token_alignment(canonical, pieces)
    line_matches: list[LineMatch] = []
    canonical_index = 0
    for line_number, text in enumerate(lyrics, start=1):
        tokens = lyric_tokens(text)
        token_matches: list[TokenMatch] = []
        for index in range(canonical_index, canonical_index + len(tokens)):
            if index in matches:
                token_matches.append(
                    TokenMatch(
                        index - canonical_index,
                        matches[index],
                        fuzzy.get(index, 1.0),
                    )
                )
        canonical_index += len(tokens)
        line_matches.append(
            LineMatch(
                line_number=line_number,
                token_count=len(tokens),
                matches=tuple(token_matches),
                score=line_match_score(len(tokens), token_matches, pieces),
                strategy="global",
            )
        )
    diagnostics = MatchingDiagnostics(
        strategy="global",
        line_match_scores={match.line_number: round(match.score, 4) for match in line_matches},
    )
    return MatchingResult(tuple(line_matches), diagnostics)


def line_match_score(
    token_count: int,
    matches: list[TokenMatch],
    pieces: tuple[WordPiece, ...],
) -> float:
    if not token_count:
        return 1.0
    if not matches:
        return 0.0
    coverage = len(matches) / token_count
    similarity = sum(match.similarity for match in matches) / len(matches)
    confidence = sum(pieces[match.piece_index].word.confidence for match in matches) / len(matches)
    return round(coverage * 0.6 + similarity * 0.25 + confidence * 0.15, 4)


def lyric_tokens(text: str) -> tuple[str, ...]:
    return tuple(token for token in (normalize_token(part) for part in split_tokens(text)) if token)


def split_tokens(text: str) -> tuple[str, ...]:
    chars: list[str] = []
    tokens: list[str] = []
    for char in normalize_apostrophes(text):
        if is_token_char(char):
            chars.append(char)
            continue
        if chars:
            tokens.append("".join(chars))
            chars.clear()
    if chars:
        tokens.append("".join(chars))
    return tuple(tokens)


def normalize_apostrophes(text: str) -> str:
    return text.translate(str.maketrans({"’": "'", "`": "'", "´": "'", "ʼ": "'"}))


def is_token_char(char: str) -> bool:
    category = unicodedata.category(char)
    return category[0] in {"L", "N"} or char == "'"


def normalize_token(token: str) -> str:
    folded = unicodedata.normalize("NFKD", token.casefold().replace("'", ""))
    return "".join(char for char in folded if unicodedata.category(char) != "Mn")


@lru_cache(maxsize=2048)
def token_similarity(left: str, right: str) -> float:
    if left == right:
        return 1.0
    return SequenceMatcher(a=left, b=right).ratio()


def canonical_tokens(lyrics: tuple[str, ...]) -> tuple[CanonicalToken, ...]:
    return tuple(
        CanonicalToken(token=token, line_number=line_number)
        for line_number, text in enumerate(lyrics, start=1)
        for token in lyric_tokens(text)
    )


def aligned_word_pieces(words: tuple[AlignedWord, ...]) -> tuple[WordPiece, ...]:
    pieces: list[WordPiece] = []
    for word_index, word in enumerate(words):
        for token in lyric_tokens(word.text):
            pieces.append(WordPiece(token=token, word_index=word_index, word=word))
    return tuple(pieces)


def global_token_alignment(
    canonical: tuple[CanonicalToken, ...],
    pieces: tuple[WordPiece, ...],
) -> tuple[dict[int, int], dict[int, float]]:
    rows = len(canonical) + 1
    cols = len(pieces) + 1
    costs = [[0.0] * cols for _ in range(rows)]
    steps = [[""] * cols for _ in range(rows)]
    for row in range(1, rows):
        costs[row][0] = row * INSERT_DELETE_COST
        steps[row][0] = "up"
    for col in range(1, cols):
        costs[0][col] = col * INSERT_DELETE_COST
        steps[0][col] = "left"
    for row in range(1, rows):
        for col in range(1, cols):
            similarity = token_similarity(canonical[row - 1].token, pieces[col - 1].token)
            diagonal = costs[row - 1][col - 1] + substitution_cost(similarity)
            up = costs[row - 1][col] + INSERT_DELETE_COST
            left = costs[row][col - 1] + INSERT_DELETE_COST
            cost, step = min(
                (diagonal, "diag"),
                (up, "up"),
                (left, "left"),
                key=lambda item: item[0],
            )
            costs[row][col] = cost
            steps[row][col] = step

    matches: dict[int, int] = {}
    fuzzy: dict[int, float] = {}
    row = len(canonical)
    col = len(pieces)
    while row or col:
        step = steps[row][col]
        if step == "diag":
            similarity = token_similarity(canonical[row - 1].token, pieces[col - 1].token)
            if similarity >= FUZZY_MATCH_THRESHOLD:
                matches[row - 1] = col - 1
                if similarity < 1.0:
                    fuzzy[row - 1] = similarity
            row -= 1
            col -= 1
            continue
        if step == "up":
            row -= 1
            continue
        col -= 1
    return matches, fuzzy


def substitution_cost(similarity: float) -> float:
    if similarity >= FUZZY_MATCH_THRESHOLD:
        return 1.0 - similarity
    return MISMATCH_COST


def line_signature(tokens: tuple[str, ...]) -> tuple[str, ...]:
    return tokens


def unique(values: list[int]) -> tuple[int, ...]:
    result: list[int] = []
    seen: set[int] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return tuple(result)
