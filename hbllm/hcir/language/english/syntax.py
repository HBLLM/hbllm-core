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

from hbllm.hcir.language.english.lexicon import EnglishPOS
from hbllm.hcir.language.english.morphology import MorphToken


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
    infinitive_verb: MorphToken | None = None


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
        if (
            clean_tokens[0].pos in (EnglishPOS.AUX, EnglishPOS.VERB)
            and clean_tokens[0].lemma
            in (
                "be",
                "do",
                "did",
                "does",
                "can",
                "could",
                "will",
                "would",
                "is",
                "are",
                "was",
                "were",
            )
        ) or clean_tokens[0].surface.lower() in (
            "is",
            "are",
            "was",
            "were",
            "do",
            "did",
            "does",
            "can",
            "could",
            "will",
            "would",
        ):
            return self._parse_yes_no_question(clean_tokens)

        # 3. Check for Existential ("There is a ball on the table")
        if (
            clean_tokens[0].lemma == "there"
            and len(clean_tokens) > 1
            and clean_tokens[1].lemma == "be"
        ):
            return self._parse_existential(clean_tokens)

        # 4. Check for Imperative ("Move the red ball to the table", "Push the box")
        # Never treat inflected plural/3s/past nouns or NP subjects followed by a verb/aux as imperatives.
        first_tok = clean_tokens[0]
        first_inflected_non_imp = (
            first_tok.features.get("number") == "plural"
            or first_tok.features.get("person") == "3s"
            or first_tok.features.get("tense") == "past"
            or first_tok.features.get("aspect") == "progressive"
        )
        has_later_predicate = len(clean_tokens) >= 2 and (
            clean_tokens[1].pos in (EnglishPOS.AUX, EnglishPOS.VERB)
            or (
                "noun" in first_tok.features.get("alt_pos", "")
                and any(t.pos in (EnglishPOS.AUX, EnglishPOS.VERB) for t in clean_tokens[1:])
            )
        )
        if (
            first_tok.pos == EnglishPOS.VERB
            and not first_inflected_non_imp
            and not has_later_predicate
        ):
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

        # Check head noun (allow compound nouns up to 5 consecutive NOUN tokens)
        if idx < len(tokens) and tokens[idx].pos == EnglishPOS.NOUN:
            compound_parts = [tokens[idx]]
            idx += 1
            while (
                idx < len(tokens)
                and len(compound_parts) < 5
                and tokens[idx].pos == EnglishPOS.NOUN
                and tokens[idx].features.get("is_path") != "true"
                and not (idx + 1 < len(tokens) and tokens[idx + 1].pos == EnglishPOS.DET)
            ):
                compound_parts.append(tokens[idx])
                idx += 1
            adjs.extend(compound_parts[:-1])
            head = compound_parts[-1]
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
                np.prepositional_phrases.append(
                    PrepositionalPhrase(preposition=prep, noun_phrase=nested_np)
                )
                idx = next_idx

        return np, idx

    # ── Construction Parsers ──────────────────────────────────────────

    def _parse_declarative(self, tokens: list[MorphToken]) -> SentenceNode | None:
        """Parse declarative sentence: NP + VP (or Copular: NP + BE + Adj/PP)."""
        np, idx = self.parse_noun_phrase(tokens, 0)
        if not np or idx >= len(tokens):
            return None

        # Check negative determiner ("no dog", "no cars")
        is_negated = np.determiner is not None and np.determiner.lemma == "no"

        # Optional pre-verbal adverb ("never", "always", "often", "usually")
        while idx < len(tokens) and tokens[idx].pos == EnglishPOS.ADV:
            if tokens[idx].lemma == "never":
                is_negated = True
            idx += 1
        if idx >= len(tokens):
            return None

        # Check verb / copula
        verb_tok = tokens[idx]
        idx += 1

        if verb_tok.lemma == "be" or verb_tok.pos == EnglishPOS.AUX:
            # Check negation marker "not" / "never" or benign adverbs
            while idx < len(tokens) and tokens[idx].pos == EnglishPOS.ADV:
                if tokens[idx].lemma in ("not", "never"):
                    is_negated = True
                idx += 1

            # Check if followed by a main lexical verb (e.g. "Heat does not cause rust")
            if idx < len(tokens) and tokens[idx].pos == EnglishPOS.VERB:
                main_verb = tokens[idx]
                idx += 1
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

                vp = VerbPhrase(
                    head_verb=main_verb,
                    direct_object=obj_np,
                    prepositional_phrases=pps,
                )
                return SentenceNode(
                    construction=ConstructionType.DECLARATIVE,
                    subject=np,
                    verb_phrase=vp,
                    auxiliary=verb_tok,
                    is_negated=is_negated,
                )

            if idx < len(tokens) and tokens[idx].pos == EnglishPOS.ADJ:
                # Copular property: "The ball is red", "A drell is warm-blooded"
                # Unless followed immediately by a noun (e.g. "Florbs are small aquatic creatures")
                adj_tok = tokens[idx]
                next_i = idx + 1
                if next_i < len(tokens) and tokens[next_i].pos in (EnglishPOS.ADJ, EnglishPOS.NOUN):
                    pred_np, next_idx = self.parse_noun_phrase(tokens, idx)
                    if pred_np:
                        vp = VerbPhrase(head_verb=verb_tok, direct_object=pred_np)
                        return SentenceNode(
                            construction=ConstructionType.COPULAR,
                            subject=np,
                            verb_phrase=vp,
                            is_negated=is_negated,
                        )
                vp = VerbPhrase(head_verb=verb_tok, adjective_complement=adj_tok)
                return SentenceNode(
                    construction=ConstructionType.COPULAR,
                    subject=np,
                    verb_phrase=vp,
                    is_negated=is_negated,
                )

            elif idx < len(tokens) and tokens[idx].pos == EnglishPOS.PREP:
                # Spatial assertion: "The ball is on the table"
                prep = tokens[idx]
                target_np, next_idx = self.parse_noun_phrase(tokens, idx + 1)
                if target_np:
                    pp = PrepositionalPhrase(preposition=prep, noun_phrase=target_np)
                    vp = VerbPhrase(head_verb=verb_tok, prepositional_phrases=[pp])
                    return SentenceNode(
                        construction=ConstructionType.DECLARATIVE,
                        subject=np,
                        verb_phrase=vp,
                        is_negated=is_negated,
                    )

            else:
                # Predicative nominal copular: "Focus is a cognitive process"
                pred_np, next_idx = self.parse_noun_phrase(tokens, idx)
                if pred_np:
                    vp = VerbPhrase(head_verb=verb_tok, direct_object=pred_np)
                    return SentenceNode(
                        construction=ConstructionType.COPULAR,
                        subject=np,
                        verb_phrase=vp,
                        is_negated=is_negated,
                    )

        if verb_tok.pos == EnglishPOS.VERB:
            # Standard or ditransitive SVO:
            # "Alice gave Bob the brass key", "The robot pushed the box to the table", "Florbs live in caves"
            obj_np, next_idx = self.parse_noun_phrase(tokens, idx)
            idx = next_idx if obj_np else idx

            indirect_np: NounPhrase | None = None
            if (
                obj_np is not None
                and idx < len(tokens)
                and tokens[idx].pos in (EnglishPOS.DET, EnglishPOS.ADJ, EnglishPOS.NOUN)
            ):
                second_np, next_idx2 = self.parse_noun_phrase(tokens, idx)
                if second_np is not None:
                    indirect_np = obj_np
                    obj_np = second_np
                    idx = next_idx2

            # Coordinated direct object NPs ("converts sunlight and water into glucose")
            if (
                obj_np is not None
                and idx + 1 < len(tokens)
                and tokens[idx].pos == EnglishPOS.CONJ
                and tokens[idx].lemma in ("and", "or")
                and tokens[idx + 1].pos in (EnglishPOS.DET, EnglishPOS.ADJ, EnglishPOS.NOUN)
                and not (
                    idx + 2 < len(tokens)
                    and tokens[idx + 2].pos in (EnglishPOS.VERB, EnglishPOS.AUX)
                )
            ):
                coord_np, c_next = self.parse_noun_phrase(tokens, idx + 1)
                if coord_np is not None:
                    obj_np.raw_tokens = [*obj_np.raw_tokens, tokens[idx], *coord_np.raw_tokens]
                    idx = c_next

            # Causal / object-control infinitive: "Heavy rain caused the river to flood"
            inf_verb: MorphToken | None = None
            if (
                obj_np is not None
                and idx + 1 < len(tokens)
                and tokens[idx].lemma == "to"
                and tokens[idx + 1].pos == EnglishPOS.VERB
            ):
                inf_verb = tokens[idx + 1]
                idx += 2

            pps = []
            while idx < len(tokens) and tokens[idx].pos == EnglishPOS.PREP:
                prep = tokens[idx]
                p_np, p_next = self.parse_noun_phrase(tokens, idx + 1)
                if p_np:
                    pps.append(PrepositionalPhrase(preposition=prep, noun_phrase=p_np))
                    idx = p_next
                else:
                    break

            vp = VerbPhrase(
                head_verb=verb_tok,
                direct_object=obj_np,
                indirect_object=indirect_np,
                prepositional_phrases=pps,
                infinitive_verb=inf_verb,
            )
            return SentenceNode(
                construction=ConstructionType.DECLARATIVE,
                subject=np,
                verb_phrase=vp,
                is_negated=is_negated,
            )

        return None

    def _parse_imperative(self, tokens: list[MorphToken]) -> SentenceNode | None:
        """Parse imperative: Verb + (NP) + (PP)* ("Move the ball to the table")."""
        verb_tok = tokens[0]
        idx = 1

        # Skip polite/beneficiary pronouns on conversational commands ("list me the files")
        if verb_tok.lemma in {
            "list",
            "read",
            "show",
            "display",
            "find",
            "search",
            "explain",
            "describe",
            "tell",
            "fetch",
            "get",
        }:
            if (
                idx + 1 < len(tokens)
                and tokens[idx].lemma == "for"
                and tokens[idx + 1].lemma in ("me", "us")
                and idx + 2 < len(tokens)
            ):
                idx += 2
            elif idx < len(tokens) and tokens[idx].lemma in ("me", "us") and idx + 1 < len(tokens):
                idx += 1

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
        if not subject_np:
            return None

        # Prevent greedy compound-NP from swallowing the predicate complement ("Is a drell warm-blooded?")
        if idx >= len(tokens):
            if aux_tok.lemma == "be" and len(tokens) >= 3:
                is_two_cap_name = (
                    tokens[-1].surface[:1].isupper() and tokens[-2].surface[:1].isupper()
                )
                if not is_two_cap_name:
                    alt_subj, alt_idx = self.parse_noun_phrase(tokens[:-1], 1)
                    if alt_subj is not None and alt_idx == len(tokens) - 1:
                        subject_np = alt_subj
                        idx = alt_idx
            if idx >= len(tokens):
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

        # Check Adj complement ("Is the ball red?", "Is a drell warm-blooded?")
        if idx < len(tokens) and tokens[idx].pos == EnglishPOS.ADJ:
            adj_tok = tokens[idx]
            next_i = idx + 1
            if next_i < len(tokens) and tokens[next_i].pos in (EnglishPOS.ADJ, EnglishPOS.NOUN):
                pred_np, _ = self.parse_noun_phrase(tokens, idx)
                if pred_np:
                    vp = VerbPhrase(head_verb=aux_tok, direct_object=pred_np)
                    return SentenceNode(
                        construction=ConstructionType.YES_NO_QUESTION,
                        subject=subject_np,
                        verb_phrase=vp,
                        auxiliary=aux_tok,
                    )
            vp = VerbPhrase(head_verb=aux_tok, adjective_complement=adj_tok)
            return SentenceNode(
                construction=ConstructionType.YES_NO_QUESTION,
                subject=subject_np,
                verb_phrase=vp,
                auxiliary=aux_tok,
            )

        # Check Verb + (NP)? + (PP)* complement ("Do florbs live in caves?", "Do florbs fly?")
        if idx < len(tokens) and tokens[idx].pos == EnglishPOS.VERB:
            verb_tok = tokens[idx]
            obj_np, next_idx = self.parse_noun_phrase(tokens, idx + 1)
            cur = next_idx if obj_np else idx + 1
            pps: list[PrepositionalPhrase] = []
            while cur < len(tokens) and tokens[cur].pos == EnglishPOS.PREP:
                prep = tokens[cur]
                p_np, p_next = self.parse_noun_phrase(tokens, cur + 1)
                if p_np:
                    pps.append(PrepositionalPhrase(preposition=prep, noun_phrase=p_np))
                    cur = p_next
                else:
                    break
            vp = VerbPhrase(head_verb=verb_tok, direct_object=obj_np, prepositional_phrases=pps)
            return SentenceNode(
                construction=ConstructionType.YES_NO_QUESTION,
                subject=subject_np,
                verb_phrase=vp,
                auxiliary=aux_tok,
            )

        # Check NP nominal complement ("Is Pluto a planet?")
        pred_np, _ = self.parse_noun_phrase(tokens, idx)
        if pred_np:
            vp = VerbPhrase(head_verb=aux_tok, direct_object=pred_np)
            return SentenceNode(
                construction=ConstructionType.YES_NO_QUESTION,
                subject=subject_np,
                verb_phrase=vp,
                auxiliary=aux_tok,
            )

        return None

    def _parse_wh_question(self, tokens: list[MorphToken]) -> SentenceNode | None:
        """Parse Wh-question: Where is the ball? / What do florbs eat? / Who was tired?"""
        wh_tok = tokens[0]
        idx = 1
        wh_prop: MorphToken | None = None

        # Check for property specification: "What color is the ball?"
        if (
            wh_tok.lemma == "what"
            and idx < len(tokens)
            and tokens[idx].lemma in ("color", "location", "shape", "size")
        ):
            wh_prop = tokens[idx]
            idx += 1

        if idx >= len(tokens):
            return None

        # Check Wh + Main Verb ("What causes crash?", "Who walked home?", "Who closed the gate?")
        if tokens[idx].pos == EnglishPOS.VERB:
            verb_tok = tokens[idx]
            obj_np, next_idx = self.parse_noun_phrase(tokens, idx + 1)
            cur = next_idx if obj_np else idx + 1
            pps: list[PrepositionalPhrase] = []
            while cur < len(tokens) and tokens[cur].pos == EnglishPOS.PREP:
                prep = tokens[cur]
                p_np, p_next = self.parse_noun_phrase(tokens, cur + 1)
                if p_np:
                    pps.append(PrepositionalPhrase(preposition=prep, noun_phrase=p_np))
                    cur = p_next
                else:
                    break
            vp = VerbPhrase(head_verb=verb_tok, direct_object=obj_np, prepositional_phrases=pps)
            return SentenceNode(
                construction=ConstructionType.WH_QUESTION,
                subject=None,
                verb_phrase=vp,
                wh_word=wh_tok,
            )

        if tokens[idx].pos != EnglishPOS.AUX:
            return None

        aux_tok = tokens[idx]
        idx += 1
        if idx >= len(tokens):
            return None

        # Subject-seeking "Who was tired?", "Who is in the cave?", "Who is a wizard?"
        if wh_tok.lemma == "who" and aux_tok.lemma == "be":
            if tokens[idx].pos == EnglishPOS.ADJ:
                adj_tok = tokens[idx]
                if idx + 1 >= len(tokens):
                    vp = VerbPhrase(head_verb=aux_tok, adjective_complement=adj_tok)
                    return SentenceNode(
                        construction=ConstructionType.WH_QUESTION,
                        subject=None,
                        verb_phrase=vp,
                        auxiliary=aux_tok,
                        wh_word=wh_tok,
                    )
            elif tokens[idx].pos == EnglishPOS.PREP:
                prep = tokens[idx]
                target_np, _ = self.parse_noun_phrase(tokens, idx + 1)
                if target_np:
                    pp = PrepositionalPhrase(preposition=prep, noun_phrase=target_np)
                    vp = VerbPhrase(head_verb=aux_tok, prepositional_phrases=[pp])
                    return SentenceNode(
                        construction=ConstructionType.WH_QUESTION,
                        subject=None,
                        verb_phrase=vp,
                        auxiliary=aux_tok,
                        wh_word=wh_tok,
                    )
            elif tokens[idx].pos == EnglishPOS.DET and tokens[idx].lemma in ("a", "an"):
                pred_np, _ = self.parse_noun_phrase(tokens, idx)
                if pred_np:
                    vp = VerbPhrase(head_verb=aux_tok, direct_object=pred_np)
                    return SentenceNode(
                        construction=ConstructionType.WH_QUESTION,
                        subject=None,
                        verb_phrase=vp,
                        auxiliary=aux_tok,
                        wh_word=wh_tok,
                    )

        subject_np, next_idx = self.parse_noun_phrase(tokens, idx)
        if not subject_np:
            return None

        # Guard if compound-NP swallowed trailing verb after do/does/did ("What do florbs eat?")
        if (
            next_idx >= len(tokens)
            and aux_tok.lemma in ("do", "does", "did", "can", "could", "will", "would")
            and len(tokens) > idx + 1
            and tokens[-1].pos in (EnglishPOS.VERB, EnglishPOS.NOUN)
        ):
            alt_subj, alt_idx = self.parse_noun_phrase(tokens[:-1], idx)
            if alt_subj is not None and alt_idx == len(tokens) - 1:
                subject_np = alt_subj
                next_idx = alt_idx

        # Check Wh + Aux + Subject + Main Verb ("What does encryption require?", "Where do florbs live?")
        if next_idx < len(tokens) and (
            tokens[next_idx].pos == EnglishPOS.VERB
            or (
                aux_tok.lemma in ("do", "does", "did", "can", "could", "will", "would")
                and "verb" in tokens[next_idx].features.get("alt_pos", "")
            )
        ):
            verb_tok = tokens[next_idx]
            obj_np, obj_next = self.parse_noun_phrase(tokens, next_idx + 1)
            cur = obj_next if obj_np else next_idx + 1
            pps = []
            while cur < len(tokens) and tokens[cur].pos == EnglishPOS.PREP:
                prep = tokens[cur]
                p_np, p_next = self.parse_noun_phrase(tokens, cur + 1)
                if p_np:
                    pps.append(PrepositionalPhrase(preposition=prep, noun_phrase=p_np))
                    cur = p_next
                else:
                    break
            vp = VerbPhrase(head_verb=verb_tok, direct_object=obj_np, prepositional_phrases=pps)
            return SentenceNode(
                construction=ConstructionType.WH_QUESTION,
                subject=subject_np,
                verb_phrase=vp,
                auxiliary=aux_tok,
                wh_word=wh_tok,
            )

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
