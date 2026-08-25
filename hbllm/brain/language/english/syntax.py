"""English Syntax & Phrase Structure Parser for A16.

Deterministic phrase structure parser supporting major sentence constructions:
- DECLARATIVE (NP + VP)
- IMPERATIVE (VP with implicit agent)
- COPULAR (NP + BE + Adj / PP)
- EXISTENTIAL (There + BE + NP + PP)
- YES_NO_QUESTION (Aux + NP + VP / PP)
- WH_QUESTION (Wh + Aux + NP / PP)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from hbllm.brain.language.english.lexicon import EnglishPOS
from hbllm.brain.language.english.morphology import MorphToken


class ConstructionType(StrEnum):
    """Syntactic sentence construction types."""

    DECLARATIVE = "declarative"
    IMPERATIVE = "imperative"
    COPULAR = "copular"
    EXISTENTIAL = "existential"
    YES_NO_QUESTION = "yes_no_question"
    WH_QUESTION = "wh_question"


@dataclass
class PrepositionalPhrase:
    """Prepositional Phrase (PP -> Prep + NP)."""

    preposition: MorphToken
    noun_phrase: NounPhrase


@dataclass
class NounPhrase:
    """Noun Phrase (NP -> (Det) (Adj)* Noun (PP)*)."""

    determiner: MorphToken | None = None
    adjectives: list[MorphToken] = field(default_factory=list)
    head_noun: MorphToken | None = None
    pronoun: MorphToken | None = None
    prepositional_phrases: list[PrepositionalPhrase] = field(default_factory=list)
    raw_tokens: list[MorphToken] = field(default_factory=list)


@dataclass
class VerbPhrase:
    """Verb Phrase (VP -> Verb (NP) (PP)*)."""

    head_verb: MorphToken
    direct_object: NounPhrase | None = None
    indirect_object: NounPhrase | None = None
    prepositional_phrases: list[PrepositionalPhrase] = field(default_factory=list)
    adjective_complement: MorphToken | None = None


@dataclass
class SentenceNode:
    """Top-level parsed sentence AST node."""

    construction: ConstructionType
    subject: NounPhrase | None = None
    verb_phrase: VerbPhrase | None = None
    auxiliary: MorphToken | None = None
    wh_word: MorphToken | None = None
    wh_property: MorphToken | None = None
    is_negated: bool = False


class EnglishSyntaxParser:
    """Deterministic recursive-descent / pattern parser for English utterances."""

    def parse(self, tokens: list[MorphToken]) -> SentenceNode | None:
        """Parse a sequence of MorphTokens into a typed SentenceNode AST."""
        # Strip trailing punctuation for syntax checking
        clean_tokens = [t for t in tokens if t.pos != EnglishPOS.PUNCT]
        if not clean_tokens:
            return None

        # 1. Check for Wh-Questions ("Where is the ball?", "What color is the ball?")
        if clean_tokens[0].pos == EnglishPOS.WH:
            return self._parse_wh_question(clean_tokens)

        # 2. Check for Yes/No Questions ("Is the ball on the table?", "Did the robot push the box?")
        if (clean_tokens[0].pos in (EnglishPOS.AUX, EnglishPOS.VERB) and clean_tokens[0].lemma in ("be", "do", "did", "does", "can", "is", "are", "was", "were")) or clean_tokens[0].surface.lower() in ("is", "are", "was", "were", "do", "did", "does", "can"):
            return self._parse_yes_no_question(clean_tokens)

        # 3. Check for Existential ("There is a ball on the table")
        if clean_tokens[0].lemma == "there" and len(clean_tokens) > 1 and clean_tokens[1].lemma == "be":
            return self._parse_existential(clean_tokens)

        # 4. Check for Imperative ("Move the red ball to the table", "Push the box")
        if clean_tokens[0].pos == EnglishPOS.VERB:
            return self._parse_imperative(clean_tokens)

        # 5. Fallback to Declarative / Copular (NP + VP or NP + BE + Adj/PP)
        return self._parse_declarative(clean_tokens)

    # ── Noun Phrase Parser ────────────────────────────────────────────

    def parse_noun_phrase(
        self,
        tokens: list[MorphToken],
        start_idx: int = 0,
        allow_pp: bool = False,
    ) -> tuple[NounPhrase | None, int]:
        """Parse a noun phrase: (Det) (Adj)* Noun (PP)* OR Pronoun."""
        if start_idx >= len(tokens):
            return None, start_idx

        idx = start_idx
        det: MorphToken | None = None
        adjs: list[MorphToken] = []
        head: MorphToken | None = None

        # Check for pronoun
        if tokens[idx].pos == EnglishPOS.PRON:
            pron = tokens[idx]
            np = NounPhrase(pronoun=pron, raw_tokens=[tokens[idx]])
            return np, idx + 1

        # Check determiner
        if tokens[idx].pos == EnglishPOS.DET:
            det = tokens[idx]
            idx += 1

        # Check adjectives
        while idx < len(tokens) and tokens[idx].pos == EnglishPOS.ADJ:
            adjs.append(tokens[idx])
            idx += 1

        # Check head noun
        if idx < len(tokens) and tokens[idx].pos == EnglishPOS.NOUN:
            head = tokens[idx]
            idx += 1
        elif det or adjs:
            # Missing head noun
            return None, start_idx
        else:
            return None, start_idx

        raw = tokens[start_idx:idx]
        np = NounPhrase(determiner=det, adjectives=adjs, head_noun=head, raw_tokens=raw)

        # Check for trailing prepositional phrase attached to NP only if allow_pp is True
        if allow_pp and idx < len(tokens) and tokens[idx].pos == EnglishPOS.PREP:
            prep = tokens[idx]
            nested_np, next_idx = self.parse_noun_phrase(tokens, idx + 1, allow_pp=True)
            if nested_np:
                np.prepositional_phrases.append(PrepositionalPhrase(preposition=prep, noun_phrase=nested_np))
                idx = next_idx

        return np, idx

    # ── Construction Parsers ──────────────────────────────────────────

    def _parse_declarative(self, tokens: list[MorphToken]) -> SentenceNode | None:
        """Parse declarative sentence: NP + VP (or Copular: NP + BE + Adj/PP)."""
        np, idx = self.parse_noun_phrase(tokens, 0)
        if not np or idx >= len(tokens):
            return None

        # Check verb / copula
        verb_tok = tokens[idx]
        idx += 1

        if verb_tok.lemma == "be" or verb_tok.pos == EnglishPOS.AUX:
            # Copular or spatial assertion: NP + BE + Adj OR NP + BE + PP
            is_negated = False
            if idx < len(tokens) and tokens[idx].lemma == "not":
                is_negated = True
                idx += 1

            if idx < len(tokens) and tokens[idx].pos == EnglishPOS.ADJ:
                # Copular property: "The ball is red"
                vp = VerbPhrase(head_verb=verb_tok, adjective_complement=tokens[idx])
                return SentenceNode(construction=ConstructionType.COPULAR, subject=np, verb_phrase=vp, is_negated=is_negated)

            elif idx < len(tokens) and tokens[idx].pos == EnglishPOS.PREP:
                # Spatial assertion: "The ball is on the table"
                prep = tokens[idx]
                target_np, next_idx = self.parse_noun_phrase(tokens, idx + 1)
                if target_np:
                    pp = PrepositionalPhrase(preposition=prep, noun_phrase=target_np)
                    vp = VerbPhrase(head_verb=verb_tok, prepositional_phrases=[pp])
                    return SentenceNode(construction=ConstructionType.DECLARATIVE, subject=np, verb_phrase=vp, is_negated=is_negated)

        if verb_tok.pos == EnglishPOS.VERB:
            # Standard SVO: "The robot pushed the box to the table"
            obj_np, next_idx = self.parse_noun_phrase(tokens, idx)
            idx = next_idx if obj_np else idx

            pps: list[PrepositionalPhrase] = []
            while idx < len(tokens) and tokens[idx].pos == EnglishPOS.PREP:
                prep = tokens[idx]
                p_np, p_next = self.parse_noun_phrase(tokens, idx + 1)
                if p_np:
                    pps.append(PrepositionalPhrase(preposition=prep, noun_phrase=p_np))
                    idx = p_next
                else:
                    break

            vp = VerbPhrase(head_verb=verb_tok, direct_object=obj_np, prepositional_phrases=pps)
            return SentenceNode(construction=ConstructionType.DECLARATIVE, subject=np, verb_phrase=vp)

        return None

    def _parse_imperative(self, tokens: list[MorphToken]) -> SentenceNode | None:
        """Parse imperative: Verb + (NP) + (PP)* ("Move the ball to the table")."""
        verb_tok = tokens[0]
        idx = 1

        obj_np, next_idx = self.parse_noun_phrase(tokens, idx)
        idx = next_idx if obj_np else idx

        pps: list[PrepositionalPhrase] = []
        while idx < len(tokens) and tokens[idx].pos == EnglishPOS.PREP:
            prep = tokens[idx]
            p_np, p_next = self.parse_noun_phrase(tokens, idx + 1)
            if p_np:
                pps.append(PrepositionalPhrase(preposition=prep, noun_phrase=p_np))
                idx = p_next
            else:
                break

        vp = VerbPhrase(head_verb=verb_tok, direct_object=obj_np, prepositional_phrases=pps)
        return SentenceNode(construction=ConstructionType.IMPERATIVE, subject=None, verb_phrase=vp)

    def _parse_yes_no_question(self, tokens: list[MorphToken]) -> SentenceNode | None:
        """Parse Yes/No question: Aux + NP + PP/VP ("Is the ball on the table?")."""
        aux_tok = tokens[0]
        subject_np, idx = self.parse_noun_phrase(tokens, 1)
        if not subject_np or idx >= len(tokens):
            return None

        # Check PP complement ("on the table")
        if idx < len(tokens) and tokens[idx].pos == EnglishPOS.PREP:
            prep = tokens[idx]
            target_np, _ = self.parse_noun_phrase(tokens, idx + 1)
            if target_np:
                pp = PrepositionalPhrase(preposition=prep, noun_phrase=target_np)
                vp = VerbPhrase(head_verb=aux_tok, prepositional_phrases=[pp])
                return SentenceNode(
                    construction=ConstructionType.YES_NO_QUESTION,
                    subject=subject_np,
                    verb_phrase=vp,
                    auxiliary=aux_tok,
                )

        # Check Adj complement ("Is the ball red?")
        if idx < len(tokens) and tokens[idx].pos == EnglishPOS.ADJ:
            vp = VerbPhrase(head_verb=aux_tok, adjective_complement=tokens[idx])
            return SentenceNode(
                construction=ConstructionType.YES_NO_QUESTION,
                subject=subject_np,
                verb_phrase=vp,
                auxiliary=aux_tok,
            )

        return None

    def _parse_wh_question(self, tokens: list[MorphToken]) -> SentenceNode | None:
        """Parse Wh-question: Where is the ball? / What color is the ball?"""
        wh_tok = tokens[0]
        idx = 1
        wh_prop: MorphToken | None = None

        # Check for property specification: "What color is the ball?"
        if wh_tok.lemma == "what" and idx < len(tokens) and tokens[idx].lemma in ("color", "location", "shape", "size"):
            wh_prop = tokens[idx]
            idx += 1

        if idx >= len(tokens) or tokens[idx].pos != EnglishPOS.AUX:
            return None

        aux_tok = tokens[idx]
        idx += 1

        subject_np, _ = self.parse_noun_phrase(tokens, idx)
        if not subject_np:
            return None

        vp = VerbPhrase(head_verb=aux_tok)
        return SentenceNode(
            construction=ConstructionType.WH_QUESTION,
            subject=subject_np,
            verb_phrase=vp,
            auxiliary=aux_tok,
            wh_word=wh_tok,
            wh_property=wh_prop,
        )

    def _parse_existential(self, tokens: list[MorphToken]) -> SentenceNode | None:
        """Parse Existential: There is a ball on the table."""
        # tokens[0] = "there", tokens[1] = "is"
        aux_tok = tokens[1]
        np, idx = self.parse_noun_phrase(tokens, 2)
        if not np:
            return None

        pps: list[PrepositionalPhrase] = []
        if idx < len(tokens) and tokens[idx].pos == EnglishPOS.PREP:
            prep = tokens[idx]
            target_np, _ = self.parse_noun_phrase(tokens, idx + 1)
            if target_np:
                pps.append(PrepositionalPhrase(preposition=prep, noun_phrase=target_np))

        vp = VerbPhrase(head_verb=aux_tok, direct_object=np, prepositional_phrases=pps)
        return SentenceNode(construction=ConstructionType.EXISTENTIAL, subject=None, verb_phrase=vp)
