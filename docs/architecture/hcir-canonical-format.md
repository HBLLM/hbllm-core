---
title: "Canonical .hcir Format Specification (v1)"
description: "Lossless, self-describing S-expression specification for HCIR instruction streams, graph patterns, symbolic/fuzzy expressions, cognitive nodes, and seed bundles."
---

# Canonical `.hcir` Format Specification (v1)

> **Status:** Phase 1 Specification ([ADR 007](../adr/0007-canonical-hcir-textual-representation.md))  
> **Scope:** Authoritative textual bijection for `hbllm.hcir` IR models (`bytecode.py`, `unification.py`, `skills/declarative.py`, `fuzzy.py`, `nodes.py`, `graph.py`, `seed/loader.py`).

---

## 1. Scope, Framing & Rule 0 (Zero Hidden Semantics)

**`.hcir` v1 is a canonical, lossless textual serialization format for HCIR** — not a new programming language with independent runtime semantics.

```mermaid
flowchart LR
    EXP["Experience / Author"] --> IR["HCIR Objects (InstructionStream, GraphPattern, SkillNode)"]
    IR -->|"format_hcir (canonical / exact)"| TXT["Canonical .hcir Source"]
    TXT -->|"Git diff / CLI inspect"| REV["Human Audit & Validation"]
    TXT -->|"parse_hcir"| IR2["Equivalent HCIR Objects"]
    IR2 -->|"HCIRInterpreter / TieredWorkspace"| VM["Deterministic VM Execution"]
```

### Rule 0 — Zero Hidden Semantics

1. Every `.hcir` v1 construct maps **1:1** to an existing HCIR Pydantic model or `seed/loader.py` bundle structure.
2. `.hcir` v1 contains **no syntactic sugar, no implicit control flow, and no hidden coercion** (for example, `(if ...)` is disallowed in v1; use explicit `(branch ...)`).
3. Canonicalization is deterministic and idempotent: formatting a parsed document produces byte-identical UTF-8 output.

---

## 2. Lexical Grammar & Complete EBNF

### 2.1 Encoding & Whitespace

- Input must be valid **UTF-8**.
- Whitespace (`' '`, `'\t'`, `'\r'`, `'\n'`) separates adjacent tokens and has no semantic meaning outside string literals.
- Line comments begin with `;` outside string literals and run until `\n` or EOF. The reader discards comments; the canonical printer never emits comments.

### 2.2 EBNF

```ebnf
document    := ws* ( envelope | form ) ws* ;
envelope    := "(" ws* "hcir" ws+ ":version" ws+ "1" ws+ ":type" ws+ doc_type ws+ form ws* ")" ;
form        := "(" ws* head ( ws+ item )* ws* ")" ;
item        := form | atom ;
atom        := nil | boolean | float | integer | register | var | keyword | string | symbol ;

nil         := "nil" ;
boolean     := "true" | "false" ;
integer     := [ "+" | "-" ] digit+ ;
float       := [ "+" | "-" ] digit+ "." digit+ [ ("e" | "E") [ "+" | "-" ] digit+ ]
             | "+inf" | "-inf" ;
register    := "$r" ("0" | "1" | "2" | "3" | "4" | "5" | "6" | "7") ;
var         := "?" var_char+ ;
keyword     := ":" key_start key_char* ;
string      := '"' ( unescaped_char | escape_seq )* '"' ;
symbol      := sym_start sym_char* ;        (* excluding "nil", "true", "false" *)

digit       := "0".."9" ;
var_char    := "A".."Z" | "a".."z" | "0".."9" | "_" | "." | "-" ;
key_start   := "A".."Z" | "a".."z" | "_" ;
key_char    := "A".."Z" | "a".."z" | "0".."9" | "_" | "-" ;
sym_start   := "A".."Z" | "a".."z" | "_" ;
sym_char    := "A".."Z" | "a".."z" | "0".."9" | "_" | "." | "/" | "*" | "+" | "-" | "?" ;
escape_seq  := '\"' | "\\" | "\n" | "\r" | "\t" | ( "\u" hex hex hex hex ) ;
```

### 2.3 Exact Python Type Bijection (Value Position)

In untyped payload positions (`Instruction.params`, `NodePattern.attributes`, `HCIRNode.properties`, `ConstructionNode.semantic_mapping`, etc.), every `.hcir` expression maps uniquely to a Python value without type drift:

| `.hcir` Form / Token                 | Python Type              | Canonical Formatting Rule                         | Notes                                                                                                                                                |
| :----------------------------------- | :----------------------- | :------------------------------------------------ | :--------------------------------------------------------------------------------------------------------------------------------------------------- |
| `nil`                                | `None`                   | `nil`                                             | Never quoted                                                                                                                                         |
| `true` / `false`                     | `bool`                   | `true` / `false`                                  | Checked **before** `int` (`bool` subclasses `int` in Python)                                                                                         |
| `INT` (e.g. `42`, `-1`)              | `int`                    | `str(v)`                                          | No decimal point                                                                                                                                     |
| `FLOAT` (e.g. `1.0`, `0.85`, `+inf`) | `float`                  | `repr(v)` with guaranteed `.` (or `+inf`/`-inf`)  | Integer-valued floats **must** emit `.0` (e.g. `1.0`) so `int` and `float` never swap on round-trip; `NaN` is rejected                               |
| `REGISTER` (`$r0`..`$r7`)            | `str` (`"$r0"`..`"$r7"`) | Bare `$r0`..`$r7`                                 | Matches `REGISTER_NAMES` in `hbllm/hcir/registers.py`                                                                                                |
| `VAR` (`?name`)                      | `str` (`"?name"`)        | Bare `?name`                                      | Matches `is_symbolic_var`                                                                                                                            |
| `STRING` (`"..."`)                   | `str`                    | Double-quoted JSON-escaped string                 | Canonical printer quotes all ordinary `str` values in untyped value position so `"true"`, `"42"`, `"nil"`, and `":k"` never collide with typed atoms |
| `SYMBOL`                             | `str`                    | Accepted by reader in identifier/enum/key slots   | In untyped value position, reader parses bare `SYMBOL` as `str`; printer emits `STRING` unless slot is a typed identifier/enum                       |
| `(list <v1> ...)`                    | `list[Any]`              | `(list)` when empty                               | Eliminates ambiguity with `()`/`map`/AST forms                                                                                                       |
| `(map (<k> <v>) ...)`                | `dict[str, Any]`         | `(map)` when empty; keys sorted lexicographically | Keys `<k>` are `STRING` (or `SYMBOL` accepted by reader)                                                                                             |

> **Why `(list ...)` and `(map ...)`?**  
> In untyped value slots, plain `(...)` cannot distinguish `[]` from `{}` or `[["a", 1]]`. Explicit `(list ...)` and `(map ...)` guarantee lossless round-tripping for arbitrary nested JSON/Pydantic payloads while typed structural slots (like `(tags ...)`, `(params ...)`, `(attrs ...)`) remain clean and unprefixed.

---

## 3. Self-Describing Envelope & Document Types

Top-level `.hcir` files use a self-describing envelope:

```lisp
(hcir
  :version 1
  :type <doc-type>
  <payload>)
```

Supported `:type` values and their authoritative Python targets:

| `:type`               | Payload Head                | Target Model / Module                               |
| :-------------------- | :-------------------------- | :-------------------------------------------------- |
| `stream`              | `(stream ...)`              | `InstructionStream` (`hbllm/hcir/bytecode.py`)      |
| `pattern`             | `(pattern ...)`             | `GraphPattern` (`hbllm/hcir/unification.py`)        |
| `expr`                | `<symbolic-expr>`           | `SymbolicExpr` (`hbllm/hcir/skills/declarative.py`) |
| `fuzzy-expr`          | `<fuzzy-expr>`              | `FuzzyExpr` (`hbllm/hcir/fuzzy.py`)                 |
| `skill`               | `(skill ...)`               | `SkillNode` (`hbllm/hcir/nodes.py`)                 |
| `procedure`           | `(procedure ...)`           | `ProcedureNode` (`hbllm/hcir/nodes.py`)             |
| `rule`                | `(rule ...)`                | `LearnedRuleNode` (`hbllm/hcir/nodes.py`)           |
| `construction`        | `(construction ...)`        | `ConstructionNode` (`hbllm/hcir/nodes.py`)          |
| `concept`             | `(concept ...)`             | `ConceptNode` (`hbllm/hcir/nodes.py`)               |
| `adaptive-membership` | `(adaptive-membership ...)` | `AdaptiveMembershipNode` (`hbllm/hcir/nodes.py`)    |
| `composition-policy`  | `(composition-policy ...)`  | `RelationCompositionPolicy` (`hbllm/hcir/fuzzy.py`) |
| `edge`                | `(edge-def ...)`            | `HCIREdge` (`hbllm/hcir/graph.py`)                  |
| `seed-bundle`         | `(seed-bundle ...)`         | Seed spec dict (`hbllm/hcir/seed/loader.py`)        |

- **Reader behavior:** `parse_hcir(text)` accepts both `(hcir :version 1 :type ... <payload>)` and a bare top-level `<payload>` form. Unsupported `:version` values raise `HCIRParseError`.
- **Printer behavior:** `format_hcir(obj, *, envelope=True)` emits the `(hcir :version 1 :type ...)` wrapper by default (`envelope=False` emits `<payload>` directly).

---

## 4. Family 1 — `InstructionStream` & ISA v2 Opcodes

Maps 1:1 to `InstructionStream` and `Instruction` in `hbllm/hcir/bytecode.py` and `HCIRInterpreter` in `hbllm/hcir/interpreter.py`.

### 4.1 Stream Container, Labels & Costs

```lisp
(stream
  :author "wake_sleep_learner"
  :description "Navigate to target room"
  (unify (pattern-map ("intent" "room") ("target" ?dest)) :bind $r0)
  (branch :when (empty? $r0) :goto not_found)
  (call (goal (pattern-map ("intent" "choose_route") ("to" $r0))) :bind $r1)
  (execute motor.move :params (("target" $r0) ("route" $r1)) :cost 10 :bind $r2)
  (return $r2)
  (label "not_found"
    (return)))
```

- **Stream header slots:** optional `:id "<str>"` (`exact` mode or explicit non-default ID), `:author "<str>"` (omitted when `""`), `:description "<str>"` (omitted when `""`).
- **Labels:** `(label "<label-str>" <instruction-form>)` sets `Instruction.label = "<label-str>"` directly on that instruction without creating a synthetic no-op instruction, preserving PC indices and `InstructionStream.semantic_hash()`.
- **Instruction ID & Cost:** optional `:id "<ins-id>"` (`exact` mode) and `:cost <int>` (`Instruction.cost_estimate`, omitted when `0`).
- **Bind targets (`:bind` / `:bind-all`):**
  - `:bind $r0` -> `"bind": "$r0"`
  - `:bind ?var` -> `"bind": "?var"`
  - `:bind (map ("key" $r0) ...)` -> `"bind": {"key": "$r0", ...}` (matching `_bind_output` in `interpreter.py`)
  - `:bind-all $r1` -> `"bind_all": "$r1"` (`UNIFY`)
- **Lossless overflow:** Any unrecognized keys in `Instruction.params` (such as template placeholders in `ConstructionNode.compiled_hcir_template`) are emitted under `:extra ((<k> <val>) ...)` sorted by key so every instruction round-trips with zero data loss.

### 4.2 Exact Forms for All 12 ISA v2 Opcodes

| `Opcode`   | `.hcir` Form                                                                                                                                                                         | `Instruction.params` Bijection                                                                                                                                                                                                       |
| :--------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ASSERT`   | `(assert [:node <val>] [:edge <val>] [:author <val>] [:branch <val>] [:bind <spec>])`                                                                                                | `node_data`, `edge_data`, `author`, `branch_name`, `bind`                                                                                                                                                                            |
| `RETRACT`  | `(retract [:node-id <val>] [:edge-id <val>] [:author <val>] [:branch <val>] [:bind <spec>])`                                                                                         | `node_id`, `edge_id`, `author`, `branch_name`, `bind`                                                                                                                                                                                |
| `QUERY`    | `(query [:node-type <val>] [:category <val>] [:lifecycle <val>] [:scope-tenant <val>] [:text-contains <val>] [:min-confidence <val>] [:limit <val>] [:branch <val>] [:bind <spec>])` | `node_type`, `category`, `lifecycle`, `scope_tenant`, `text_contains`, `min_confidence`, `limit`, `branch_name`, `bind`                                                                                                              |
| `EXECUTE`  | `(execute <capability> [:params ((<k> <val>) ...)] [:branch <val>] [:bind <spec>])`                                                                                                  | `capability`, `params`, `branch_name`, `bind`                                                                                                                                                                                        |
| `WAIT`     | `(wait [:event-type <val>] [:event <val>] [:condition <val>] [:timeout-ms <val>] [:bind <spec>])`                                                                                    | `event_type`, `event`, `condition`, `timeout_ms`, `bind`                                                                                                                                                                             |
| `FORK`     | `(fork [:branch <val>] [:bind <spec>])`                                                                                                                                              | `branch_name`, `bind`                                                                                                                                                                                                                |
| `MERGE`    | `(merge [:branch <val>] [:author <val>] [:bind <spec>])`                                                                                                                             | `branch_name`, `author`, `bind`                                                                                                                                                                                                      |
| `ROLLBACK` | `(rollback [:branch <val>] [:target-version <val>] [:reason <val>] [:bind <spec>])`                                                                                                  | `branch_name`, `target_version`, `reason`, `bind`                                                                                                                                                                                    |
| `UNIFY`    | `(unify <pattern-or-map> [:source <val>] [:fuzzy <bool>] [:profile-id <val>] [:min-degree <val>] [:limit <val>] [:branch <val>] [:bind <spec>] [:bind-all <reg>])`                   | `pattern` (`(pattern ...)` or `(pattern-map ...)` or raw value), `source`, `fuzzy`, `profile_id`, `min_degree`, `limit`, `branch_name`, `bind`, `bind_all`                                                                           |
| `BRANCH`   | `(branch [:when <cond>] [:goto <label>] [:goto-pc <int>])`                                                                                                                           | `condition` + optional `register`/`var`/`threshold`/`value` (see §4.3), `target_label`, `target_pc`                                                                                                                                  |
| `CALL`     | `(call <callee-spec> [:args <list-or-map>] [:fuzzy <bool>] [:profile-id <val>] [:min-degree <val>] [:eval-context <val>] [:branch <val>] [:bind <spec>])`                            | `<callee-spec>`: `(id <val>)` -> `skill_id`, `(name <val>)` -> `skill_name`, `(goal <pat>)` -> `goal_pattern`; plus `args` (`(list ...)` or `(map ...)`), `fuzzy`, `profile_id`, `min_degree`, `eval_context`, `branch_name`, `bind` |
| `RETURN`   | `(return <v1> ...)` / `(return :value <val>)` / `(return :empty)`                                                                                                                    | `(return <v> ...)` <-> `{"yield": [...]}`; `(return)` <-> `{"yield": []}`; `(return :value <v>)` <-> `{"value": <v>}`; `(return :empty)` <-> `{}`                                                                                    |

### 4.3 `BRANCH` Condition Bijection (`_eval_branch_condition`)

All 15 condition modes in `hbllm/hcir/interpreter.py` (`_eval_branch_condition`) map 1:1 as follows, where `<probe?>` is omitted (probe `last_result`), a register `$r0..$r7` (`params["register"]`), or a variable `?var` (`params["var"]`):

| `params["condition"]`          | `.hcir` `:when` Form                            | Notes                                                                                                                                       |
| :----------------------------- | :---------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------ |
| `"unconditional"` (or omitted) | `always` (or `:when` omitted)                   | Canonical printer omits `:when` when `"condition"` is absent; emits `:when always` if `"condition": "unconditional"` was explicitly present |
| `"if_empty"`                   | `(empty? [<probe>])`                            |                                                                                                                                             |
| `"if_not_empty"`               | `(not-empty? [<probe>])`                        |                                                                                                                                             |
| `"if_true"`                    | `(true? [<probe>])`                             |                                                                                                                                             |
| `"if_false"`                   | `(false? [<probe>])`                            |                                                                                                                                             |
| `"if_equals"`                  | `(eq? [<probe>] <value>)`                       | Maps to `params["value"]`                                                                                                                   |
| `"if_not_equals"`              | `(neq? [<probe>] <value>)`                      | Maps to `params["value"]`                                                                                                                   |
| `"if_lt"`                      | `(lt? [<probe>] <num> [:key value])`            | Default key is `threshold`; `:key value` emitted only if `params` used `"value"` instead of `"threshold"`                                   |
| `"if_lte"`                     | `(lte? [<probe>] <num> [:key value])`           | Same                                                                                                                                        |
| `"if_gt"`                      | `(gt? [<probe>] <num> [:key value])`            | Same                                                                                                                                        |
| `"if_gte"`                     | `(gte? [<probe>] <num> [:key value])`           | Same                                                                                                                                        |
| `"if_confidence_lt"`           | `(confidence-lt? [<probe>] <num> [:key value])` | Probes `confidence` / `uncertainty.confidence`                                                                                              |
| `"if_surprise_gt"`             | `(surprise-gt? [<probe>] <num> [:key value])`   | Probes `surprise` / `prediction_error`                                                                                                      |
| `"if_degree_gte"`              | `(degree-gte? [<probe>] <num> [:key value])`    | ADR 006 `ALPHA_CUT` gate (`>=`)                                                                                                             |
| `"if_degree_lt"`               | `(degree-lt? [<probe>] <num> [:key value])`     | ADR 006 `ALPHA_CUT` gate (`<`)                                                                                                              |

---

## 5. Family 2 — `GraphPattern` & Dual Crisp/Fuzzy Matching

Maps 1:1 to `GraphPattern`, `NodePattern`, and `EdgePattern` in `hbllm/hcir/unification.py`.

### 5.1 Canonical `GraphPattern` Form

```lisp
(pattern
  :distinct-nodes true
  (nodes
    (node ?agent
      :type physical_entity
      :id "ent_1"
      :min-confidence 0.8
      :tags ("tracked" "mobile")
      (attrs
        ("status" "operational")
        ("x" ?ax))
      (fuzzy
        ("near_goal" 0.75))
      (tolerances
        ("distance" (("sigma" 1.5) ("target" 0.0))))))
  (edges
    (edge ?agent :located_at ?dest :min-weight 0.5 :min-semantic-degree 0.7))
  (any-of
    (pattern (nodes (node ?dest :type location))))
  (none-of
    (pattern (nodes (node ?hazard :tags ("lethal"))))))
```

Slot mappings:
- `GraphPattern.distinct_nodes`: omitted when `true` (default), `:distinct-nodes false` when `false`.
- `NodePattern`: `(node <var> [:type <node_type>] [:id <id>] [:min-confidence <float>] [:tags (<s1> ...)] [(attrs (<k> <v>) ...)] [(fuzzy (<k> <float>) ...)] [(tolerances (<attr> (<sk> <float>) ...) ...)])`.
- `EdgePattern`: `(edge <source> <edge-type-kw-or-:any> <target> [:min-weight <float>] [:min-semantic-degree <float>])` where `edge_type=None` prints as `:any` and a string `"located_at"` prints as `:located_at`.
- `any_of` / `none_of`: `(any-of <pattern> ...)` / `(none-of <pattern> ...)` (omitted when empty).

### 5.2 Pattern Containers on Skills & Instructions

In `SkillNode.precondition_pattern`, `SkillNode.postcondition_pattern`, `Instruction.params["pattern"]`, and `Instruction.params["goal_pattern"]`, the underlying Python field is a `dict[str, Any]` that may hold one of three distinct shapes:

1. **Full `GraphPattern` dict** (any key in `{"nodes", "edges", "any_of", "none_of", "distinct_nodes"}`): serialized as `(pattern ...)`.
2. **Shorthand dict pattern** (e.g. `{"intent": "collision_cone", "target": "?avatar_trajectory"}` in `core_priors.hcir.yaml`): serialized as `(pattern-map ("intent" "collision_cone") ("target" ?avatar_trajectory))` so it does **not** silently coerce into `{"nodes": [...]}` on round-trip.
3. **Declarative `SymbolicExpr` guard dict** (`{"symbolic_expr": <dict>, ...}` in `interpreter.py` / `declarative.py`): serialized as `(expr-guard <symbolic-expr-form> [:capabilities (<c1> ...)] [:signature (map ...)] [:extra ((<k> <v>) ...)])`.

---

## 6. Family 3 — `SymbolicExpr` & `FuzzyExpr` ASTs

### 6.1 `SymbolicExpr` (`hbllm/hcir/skills/declarative.py`)

Every `SymbolicExpr(op, args, children, alu_syscall)` maps to a Lisp-like prefix AST form:

```lisp
(all-of
  (affordance :required (1 2 3 4))
  (grid-dim :min-height 4 :max-width 32)
  (entity-bbox :color 2 :min-count 1)
  (symmetry :axis "vertical" :max-deviation 1.5)
  (causal-do :variable "door_open" :operator "==" :expected-value true)
  (fuzzy-expr
    (alpha-cut 0.6
      (hedge very
        (membership :var "distance" :kernel gaussian :scale entity_extent
          :params (("center" 0.0) ("sigma" 1.0)))))
    :profile-id product
    :alpha 0.6))
```

Complete `SymbolicExpr.op` bijection:

| `SymbolicExpr.op`         | `.hcir` Head            | Canonical Layout                                                                                                |
| :------------------------ | :---------------------- | :-------------------------------------------------------------------------------------------------------------- |
| `"all_of"`                | `all-of`                | `(all-of <child> ...)`                                                                                          |
| `"any_of"`                | `any-of`                | `(any-of <child> ...)`                                                                                          |
| `"not"`                   | `not`                   | `(not [<child>])`                                                                                               |
| `"affordance"`            | `affordance`            | `(affordance [:exact (...)] [:required (...)] [:forbidden (...)])`                                              |
| `"grid_dim"`              | `grid-dim`              | `(grid-dim [:exact-shape (h w)] [:min-height n] [:max-height n] [:min-width n] [:max-width n])`                 |
| `"entity_bbox"`           | `entity-bbox`           | `(entity-bbox <arg-pairs...>)`                                                                                  |
| `"entity_count"`          | `entity-count`          | `(entity-count <arg-pairs...>)`                                                                                 |
| `"color_dist"`            | `color-dist`            | `(color-dist [:min-distinct-colors n] [:max-distinct-colors n] [:present-colors (...)] [:absent-colors (...)])` |
| `"cut_set"`               | `cut-set`               | `(cut-set [:min-partitions n])`                                                                                 |
| `"homogeneous_region"`    | `homogeneous-region`    | `(homogeneous-region <arg-pairs...>)`                                                                           |
| `"pixel_density"`         | `pixel-density`         | `(pixel-density <arg-pairs...>)`                                                                                |
| `"symmetry"`              | `symmetry`              | `(symmetry <arg-pairs...>)`                                                                                     |
| `"metadata"`              | `metadata`              | `(metadata :key <k> [:expected-value <v>])`                                                                     |
| `"causal_do"`             | `causal-do`             | `(causal-do :variable <str> [:operator <str>] [:expected-value <v>])`                                           |
| `"fuzzy_expr"`            | `fuzzy-expr`            | `(fuzzy-expr <fuzzy-ast-form> [:profile-id <id>] [:alpha <float>])`                                             |
| `"predictive_confidence"` | `predictive-confidence` | `(predictive-confidence [:max-free-energy <float>] [:min-confidence <float>])`                                  |
| *(any other `op`)*        | `sexpr`                 | `(sexpr :op "<raw>" [:alu-syscall "<sys>"] [:args ((<k> <v>) ...)] [:children (<c> ...)])`                      |

If `SymbolicExpr.alu_syscall` is not `None`, `:alu-syscall "<str>"` is emitted as an attribute on the form; any extra keys in `args` not listed as named keywords are preserved sorted via `:extra ((<k> <v>) ...)`.

### 6.2 `FuzzyExpr` (`hbllm/hcir/fuzzy.py`)

Complete bijection over all 10 `FuzzyExpr.op` literals (`"const"`, `"var"`, `"membership"`, `"and"`, `"or"`, `"not"`, `"implies"`, `"equiv"`, `"hedge"`, `"alpha_cut"`):

| `FuzzyExpr.op` | `.hcir` Form                                                                                                                                                     | Notes                                                                                                 |
| :------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------- |
| `"const"`      | `(const <float>)`                                                                                                                                                | `args["value"]`                                                                                       |
| `"var"`        | `(var "<name>" [:default <float>])`                                                                                                                              | `args["name"]`, optional `args["default"]`                                                            |
| `"membership"` | `(membership [:model-id "<id>"] [:var "<name>"] [:value <float>] [:kernel <family>] [:scale <scale_kind>] [:scale-context <val>] [:params ((<k> <float>) ...)])` | `kernel` -> `kernel_family`, `scale` -> `reference_scale_kind`, `params` -> `parameters`              |
| `"and"`        | `(and <f1> <f2> ...)`                                                                                                                                            | `children`                                                                                            |
| `"or"`         | `(or <f1> <f2> ...)`                                                                                                                                             | `children`                                                                                            |
| `"not"`        | `(not [<f1>])`                                                                                                                                                   | `children`                                                                                            |
| `"implies"`    | `(implies <f1> <f2>)`                                                                                                                                            | `children[0..1]`                                                                                      |
| `"equiv"`      | `(equiv <f1> <f2>)`                                                                                                                                              | `children[0..1]`                                                                                      |
| `"hedge"`      | `(hedge <hedge-kind> <f1>)`                                                                                                                                      | `args["hedge"]` (`identity`, `very`, `extremely`, `somewhat`, `slightly`, `intensify`), `children[0]` |
| `"alpha_cut"`  | `(alpha-cut <alpha> [:strict <bool>] <f1>)`                                                                                                                      | `args["alpha"]`, optional `args["strict"]`, `children[0]`                                             |

When a `FuzzyExpr` appears at top level (`:type fuzzy-expr`) or inside `(fuzzy-expr ...)` without ambiguity, the parser parses it as `FuzzyExpr`. Outside a `FuzzyExpr` context (for instance in an untyped `(map ...)`), `(fuzzy-ast <fuzzy-expr>)` disambiguates `FuzzyExpr` vs `SymbolicExpr` (for shared heads like `not`).

---

## 7. Family 4 — Cognitive Nodes, Fuzzy Policies & Seed Bundles

### 7.1 Shared Homoiconic Fields & Metadata Modes

`SkillNode`, `ProcedureNode`, `ConstructionNode`, and `LearnedRuleNode` all inherit from `_HomoiconicProgramFields` + `HCIRNode` (`hbllm/hcir/nodes.py`).

#### Parameter Specifications
Each entry in `parameters: list[dict[str, Any]]` formats as:

```lisp
(params
  (param "grid_shape" :required true)
  (param "profile_id" :required false :default "product"))
```

Any additional keys on a parameter dict (such as `"node_type"` or `"role"`) are emitted alphabetically as keyword pairs (`:node-type ...`, `:role ...`, or `:extra ((<k> <v>) ...)`).

#### Canonical vs. Exact Mode (`mode="canonical"` vs `mode="exact"`)
Because `HCIRNode` carries runtime metadata with dynamic defaults (`Provenance.timestamp = time.time()`, `Instruction.id = ins_<hex>`, `InstructionStream.id = stream_<hex>`):

- **`mode="canonical"` (default — Git, seeds, CLI inspection):**
  - Emits stable node `:id` (if non-empty) and semantic fields.
  - Omits ephemeral `Instruction.id`, `InstructionStream.id`, and `Provenance.timestamp`.
  - Omits `HCIRNode` base fields (`lifecycle`, `provenance`, `uncertainty`, `attention`, `scope`, `tags`, `properties`, `confidence`) when equal to model defaults, or emits only non-default overrides in `(meta ...)`.
- **`mode="exact"` (checkpoint / strict `==` round-trip):**
  - Emits `(meta ...)` with complete `id`, `category`, `lifecycle`, `provenance` (including `timestamp`), `uncertainty`, `attention`, `scope`, `tags`, `properties`, `confidence`, plus `:id` on `InstructionStream` and every `Instruction`.
  - Guarantees `parse_hcir(format_hcir(node, mode="exact")) == node` for Pydantic equality.

### 7.2 Cognitive Node Forms

1. **`SkillNode`:**
   ```lisp
   (skill "innate_geodesic_field"
     :id "prior_skill_geodesic_field"
     :description "Compute exact L1 shortest-path distance field over passable lattice via Layer-0 ALU."
     (params
       (param "grid_shape" :required true)
       (param "targets" :required true))
     (postcondition
       (pattern-map ("intent" "geodesic_distance_field") ("targets" "?targets")))
     (reliability :alpha 5.0 :beta 1.0)
     (body
       :author "core_priors"
       :description "Invoke math.geodesic_distance_field ALU"
       (execute math.geodesic_distance_field
         :params (("grid_shape" $r0) ("targets" $r1))
         :bind $r2)
       (return $r2)))
   ```
   - `bytecode_stream=None` -> `(body ...)` omitted; `bytecode_stream={"instructions": []}` -> empty `(body)`.
   - Legacy string lists `preconditions` / `postconditions`, plus `success_rate` and `invocation_count`, are emitted when non-default.

2. **`ProcedureNode`:**
   `(procedure "<procedure_name>" [:id "<id>"] [:description "..."] [:atomic <bool>] [(steps ("..."))] [(params ...)] [(precondition ...)] [(postcondition ...)] [(reliability ...)] [(body ...)] [(meta ...)])`

3. **`LearnedRuleNode`:**
   `(rule "<id>" :condition "<if-clause>" :prediction "<then-clause>" [:domain "<domain>"] [:confidence <float>] [:source-error-count <int>] [(applies-to ("..."))] [(params ...)] [(precondition ...)] [(postcondition ...)] [(reliability ...)] [(body ...)] [(meta ...)])`

4. **`ConstructionNode`:**
   `(construction "<construction_name>" [:id "<id>"] :language "<lang>" :intent-type "<intent>" :template "<syntactic_template>" [(triggers ("..."))] [(slot-constraints ((<k> <v>) ...))] [(semantic-mapping ((<k> <v>) ...))] [(compiled-template (<ins-1> ...))] [(params ...)] [(precondition ...)] [(postcondition ...)] [(reliability ...)] [(body ...)] [(meta ...)])`

5. **`ConceptNode`:**
   `(concept "<id>" :label "<label>" [:domain "<domain>"] [:definition "<def>"] [:confidence <float>] [(meta ...)])`

6. **`AdaptiveMembershipNode` (`hbllm/hcir/nodes.py`, ADR 006):**
   `(adaptive-membership "<id>" :model-id "<model_id>" :target-id "<target_id>" [:contextual-scope "<scope>"] :kernel <kernel_family> :scale <reference_scale_kind> [:sample-count <int>] [:version <int>] [:parent-version <int>] [:parent-hash "<hash>"] [(learned-params ((<k> <float>) ...))] [(version-snapshots (("<ver>" ((<k> <float>) ...)) ...))] [(meta ...)])`

7. **`RelationCompositionPolicy` (`hbllm/hcir/fuzzy.py`):**
   ```lisp
   (composition-policy "default_spatial_policy"
     (compose-rule "inside" "inside" -> "inside"
       :directionality forward
       :guard topological_closure
       :operator godel
       :attenuation 1.0)
     (compose-rule "near" "near" -> nil
       :directionality symmetric
       :guard none
       :operator godel
       :attenuation 0.0))
   ```

8. **`HCIREdge` (`hbllm/hcir/graph.py`):**
   `(edge-def "<id>" :type <edge_type> :sources ("<id1>" ...) :targets ("<id2>" ...) [:weight <float>] [(properties ...)] [(provenance ...)])`

### 7.3 Seed Bundles (`core_priors.hcir`, `constructions_<lang>.hcir`)

```lisp
(hcir
  :version 1
  :type seed-bundle
  (seed-bundle
    :kind core-priors
    :seed-version "2.0.0"
    :authority "docs/architecture/priors.md"
    :description "..."
    (concepts ...)
    (rules ...)
    (skills ...)
    (composition-policy ...)
    (edges ...)))
```

For multilingual construction seeds (`:kind constructions :language "en"`), `(lexical-rules ...)` stores the language's `lexical_rules` map and `(constructions ...)` holds `ConstructionNode` entries. When loading `core_priors.hcir`, `hbllm/hcir/seed/loader.py` runs `_assert_domain_purity()` on the raw text exactly as for `core_priors.hcir.yaml`.

---

## 8. Deterministic Canonical Formatting Rules

1. **Indentation:** 2 spaces per nesting depth; no tabs.
2. **Line Endings:** LF (`\n`); a single trailing `\n` at EOF for top-level output.
3. **Map Ordering:** All unordered dictionaries (`(map ...)`, `(attrs ...)`, `(fuzzy ...)`, `(tolerances ...)`, `:params`, `:extra`, `learned-params`, `version-snapshots`) sort keys in ascending UTF-8 codepoint order.
4. **Float Formatting:** Emit `repr(float(v))`, appending `.0` if neither `.` nor `e`/`E` appears (and `+inf` / `-inf` for infinities; reject `NaN`).
5. **Single-Line vs. Multi-Line Layout:**
   - Short leaf forms with scalar arguments only (e.g. `(return $r2)`, `(const 0.5)`, `(label "done" (return))`, `(empty? $r0)`) remain single-line when <= 88 columns.
   - Structural containers (`hcir`, `seed-bundle`, `skill`, `procedure`, `construction`, `stream`, `pattern`, `nodes`, `edges`, `composition-policy`) emit one child form per line indented by 2 spaces.
   - No trailing whitespace on any line.

---

## 9. Verification Suite & Phase 2–8 Implementation Contract

### 9.1 Canonical Semantic Digest (`semantic_digest` & `exact_digest`)

Define `semantic_digest(obj) -> str` (alias `canonical_digest`) and `exact_digest(obj) -> str` as 64-hex **BLAKE3** digests in `hbllm/hcir/sexpr.py` (backed by `hbllm_hcir_graph.blake3_hash` with a bit-identical pure-Python BLAKE3 fallback):
- **`semantic_digest(obj)` (`HCIR_SEMANTIC_V1`):** Normalizes floats (`-0.0 -> 0.0`), sorts map keys, normalizes `GraphPattern` variants, preserves fuzzy profiles, instruction ordering, labels, costs, and `$r0..$r7`/`?var` bindings, and strips ephemeral instance `id`s, `InstructionStream.id`, `Instruction.id`, `Provenance.timestamp`, `attention`, `lifecycle`, `invocation_count`, and `success_rate`.
- **`exact_digest(obj)` (`HCIR_EXACT_V1`):** Hashes the full type-tagged exact model projection preserving generated IDs, timestamps, and runtime counters so `exact_digest(A) != exact_digest(B)` while `semantic_digest(A) == semantic_digest(B)` for structurally identical nodes learned from distinct episodes.

### 9.2 Required Test Matrix (Phases 4–5)

1. **Structural Round-Trip:** `parse_hcir(format_hcir(x, mode="exact")) == x` across all 12 opcodes, all 15 `BRANCH` conditions, all 16 `SymbolicExpr` ops, all 10 `FuzzyExpr` ops, `GraphPattern` (crisp + fuzzy), `SkillNode`, `ProcedureNode`, `LearnedRuleNode`, `ConstructionNode`, `ConceptNode`, `AdaptiveMembershipNode`, `RelationCompositionPolicy`, `HCIREdge`, and seed bundles.
2. **Canonical Idempotence:** `format_hcir(parse_hcir(text)) == format_hcir(parse_hcir(format_hcir(parse_hcir(text))))`.
3. **Semantic Identity:** `semantic_digest(parse_hcir(format_hcir(x, mode="canonical"))) == semantic_digest(x)`.
4. **Execution Equivalence (Unit + Hypothesis Property Tests):**
   Generate random valid streams combining `FORK -> UNIFY (crisp & fuzzy) -> BRANCH (ALPHA_CUT) -> CALL -> MERGE / ROLLBACK` and assert identical `ExecutionResult.success`, `return_value`, `registers`, `annotations`, and post-execution `workspace.graph.content_hash()` between `x` and `parse_hcir(format_hcir(x))`.

### 9.3 Seed Loader & CLI Contract (Phases 6–7)

- `hbllm/hcir/seed/loader.py`: support `.hcir` alongside `.hcir.yaml` (preferring `.hcir` when explicitly passed or when canonical migration opts in) while producing identical `TransactionOperation` lists.
- CLI (`hbllm hcir`):
  - `hbllm hcir inspect skill <id>` — prints the `SkillNode` from workspace in canonical `.hcir`
  - `hbllm hcir print trace <id>` — prints an execution/replay `InstructionStream` in canonical `.hcir`
  - `hbllm hcir validate <file.hcir>` — parses, schema-validates, and checks domain purity (if seed)
  - `hbllm hcir roundtrip <file.hcir>` — verifies canonical idempotence and `semantic_digest` stability
