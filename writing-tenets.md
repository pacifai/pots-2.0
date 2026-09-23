# Writing tenets

The Google developer documentation style guide, synthesized for everyday use. Follow
these whenever you write to the user or on their behalf: terminal replies, commit
messages and PR descriptions, code comments and doc prose, `AskUserQuestion` prompts
and their options, and any message, report, or draft meant for someone else to read.

Full guide: https://developers.google.com/style

## The three that win ties

Clarity, coherence, conciseness — in that order — override every rule below. When a
guideline would make a sentence harder to understand, break the guideline. This is
house style, not law: it optimizes for a global audience reading technical material
under time pressure. Write for that reader.

## Voice and tone

- Sound like a knowledgeable colleague: conversational, plain, direct. Not formal,
  not chatty, not cute.
- No hype, no filler enthusiasm, no exclamation marks.
- Cut self-congratulation and pre-announcement. Don't call your own work robust,
  comprehensive, powerful, or seamless. Don't describe planned work as done.
- Drop throat-clearing: "please note", "it's worth noting that", "at this time",
  "as you can see".
- Don't tell the reader a task is "easy", "simple", "quick", or "just" one step. If
  it isn't, you've mocked them; if it is, they'll notice without being told.
- Contractions are fine and preferred, especially negatives — "don't", "isn't",
  "can't" read faster and are harder to misread than a lone "not".
- Skip "Let's..." and "we" for instructions. The reader acts; you address them.

## Person and address

- Second person. "You" configure the deployment, not "we" configure it.
- First person plural ("we") only for the authoring party's own choices or
  recommendations, and only with a clear antecedent.
- Reserve "user" for the end user of the software the reader is building — not the
  reader.
- Use the imperative for steps: "Run the migration", not "You should run the
  migration" or "The migration should be run".

## Verbs

- Active voice. Name who acts: "The scheduler retries the job", not "The job is
  retried". Passive is acceptable only when the actor is unknown, irrelevant, or
  better de-emphasized ("The row was deleted in January").
- Present tense. "The server returns a 404", not "will return". Reserve "will" for
  something genuinely later in time than the surrounding action.
- No hypothetical "would": "If the token expires, the server rejects the request",
  not "would reject".

## Sentences and paragraphs

- Put the condition, context, or goal before the instruction. "To reset the cache,
  delete the directory." Not "Delete the directory to reset the cache." "If you use
  Postgres, set this flag." Not the flag first.
- Short sentences, one idea each. Break compound sentences rather than stacking
  clauses.
- Lead each paragraph with its point; readers don't reach the end. One idea per
  paragraph, up to about five sentences. A one-sentence paragraph is fine.
- Vary sentence openers, but not at the cost of clarity.

## Word choice

Prefer the plain word. Common swaps:

| Instead of | Use |
|---|---|
| utilize, leverage | use |
| in order to | to |
| ingest | import, load, read |
| off-the-shelf | ready-made, prebuilt |
| e.g. | for example, such as |
| i.e. | that is |
| via | through, with, by using |
| allows you to, enables you to | lets you |
| a number of | some, many, several |
| kill, hang, hit | stop, stop responding, press or click |
| in the event that | if |
| prior to | before |

- Don't use "and/or"; pick one, or write "x, or y, or both".
- Avoid "etc." and a trailing "and so on" — either the list is complete or it names
  a category.
- Define any acronym on first use, or avoid it. Project rule from pacifai: never
  "KAT" — write "known-answer test".
- No internet shorthand: "tl;dr", "ymmv", "iirc", "RTFM".

## Claims and hedging

- No superlatives about the work: best, fastest, simplest, always, never,
  guaranteed.
- "Ensure" and "guarantee" only when something truly is guaranteed. Otherwise
  "helps", "is designed to", "in most cases".
- Cite the source for any specific performance or cost number.
- Use "must" for requirements, "we recommend" for recommendations, "can" for
  options, "might" for possible outcomes. Avoid "should" — it blurs the line between
  required and advised.

## Timeless writing

- Drop words that pin text to a moment: currently, now, new, recently, soon, as of
  this writing, at present, does not yet.
- "The API supports pagination", not "the API now supports pagination". If a version
  or date matters, name it: "Added in v2.3".
- Don't write about unreleased features or roadmap.

## Inclusive and accessible language

- Singular "they" for a person of unknown gender. Never "he/she" or "(s)he".
- Replace: whitelist/blacklist with allowlist/blocklist; master/slave with
  primary/replica or controller/worker; sanity check with quick check or confidence
  check; dummy value with placeholder; man-hours with person-hours.
- No ableist or violent metaphors: not "crazy", "insane", "blind to", "cripples",
  "hang", "hit" or "kill" as UI verbs. Say what happens: "the connection stops
  responding".
- Don't rely on visual or directional cues alone: not "the button on the right" or
  "see above" — use "the Save button", "the preceding section", "the following
  example".
- No "click here", "this link", or "this document" as link text. Link text names its
  target.

## Writing for a global audience

- Short sentences translate and skim better.
- No idioms, sports metaphors, or culture-bound references: not "ballpark", "home
  stretch", "punt on", "out of the box", "August is summer".
- Keep terminology consistent — the same thing gets the same name every time, with
  the same capitalization. Don't vary "the worker" / "the node" / "the runner" for
  one concept.
- Keep optional relative pronouns for clarity: "the rules that you defined", not
  "the rules you defined".
- Don't stack more than two nouns as modifiers: "a pipeline for cloud-native builds"
  beats "a cloud-native build pipeline config".
- Make pronoun antecedents unambiguous; follow "this" or "that" with a noun: "this
  buffer", not a bare "this".

## Structure

- Headings and titles in sentence case. Task headings start with a bare verb:
  "Configure the prover". Concept headings are noun phrases, not "-ing": "Memory
  residency", not "Verifying memory residency". No trailing punctuation, no links,
  no numbering for sequence.
- Use the plain, conventional section name. Readers scan for labels they already
  know — "Background", "Change", "Scope", "Measurements", "Testing", "Open
  questions", "Usage" — and find them faster than a heading that argues a point or
  asks a question. Write "Implementation choice", not "Why vendor the whole
  standard". Write "Scope", not "What this branch does not do". A heading is a
  label, not a claim: if it only makes sense to someone who has already read the
  section, it belongs in the body as a sentence.
- Never press a noun into service as a verb in a heading: not "Vendor the
  library", "Action the request", "Architect the system". Use the verb that
  already exists — "bundle", "handle", "design". Prefer the same in prose, but
  keep a term the project has already standardized on: an established "vendored
  dependency" reads fine as a fixed adjective, and renaming it costs more clarity
  than it buys.
- Numbered lists for sequences and priorities; bulleted lists for sets where order
  doesn't matter; description lists for term-and-definition pairs.
- Introduce a list with a full sentence, not a fragment the items complete. "The
  pipeline has three stages:", not "The pipeline:".
- Parallel structure across list items — same part of speech, same tense.
- Procedures: one action per numbered step, imperative verb first, state where
  before what, state the result after the action. A single-step procedure is a
  bullet, not "1.".
- Notes and warnings: use sparingly, never stacked. Don't put information the reader
  needs to succeed inside an aside — put it in the main flow.

## Mechanics

- Serial comma always: "zones, regions, and multi-regions".
- Comma before a coordinating conjunction joining two independent clauses, unless
  both are very short.
- Sentence case everywhere except proper nouns; don't capitalize for emphasis.
  Lowercase after a colon unless a proper noun or a full quoted sentence follows.
- Spell out zero through nine; numerals for 10 and up. Always numerals for versions,
  measurements, percentages, and anything with a unit. Spell out a number that
  starts a sentence, or rewrite.
- Dates: spell the month — "January 19, 2026". Never all-numeric (04/05/09 is
  ambiguous). Use ISO 8601 (2026-01-19) when a compact form is needed.
- Don't use "e.g.", "i.e.", or "etc." — write "for example", "that is", and finish
  the list.
- Bold for literal UI labels the reader clicks. Code font for filenames, commands,
  flags, identifiers, and literal values. Italics only for a term being defined or
  genuine emphasis.
- American spelling: color, behavior, canceled, toward.

## Code comments

Default to none. Code is read far more often than comments are, and a comment that
restates the code rots into a lie the first time someone edits around it. Direction,
rationale, and the alternatives you rejected belong in the commit message and the
decision record, not the source file — those are versioned against the change that
made them true, and nobody has to scroll past them to read the code.

Write a comment only when it carries something the code cannot:

- The spec and section a block implements: `RFC 9106 section 3.4.1.2`.
- A genuine trap: a bound that looks off by one and isn't, a parameter that means
  something other than its name suggests, an order that must not change.
- A file header of a few lines at most, naming what the file is and which spec or
  standard it follows.

Never write a comment that explains why the project wants the feature, argues for
the approach taken, or summarizes what the function plainly does. If a block needs
paragraphs to justify it, the justification is a decision record and the code should
link to it in one line.

## When rules conflict

Reach for the reader, not the rulebook. If following a tenet here produces something
stilted, ambiguous, or longer, write the clear version instead. Consistency is a
tiebreaker between equally good options — never a reason to keep a worse one.
