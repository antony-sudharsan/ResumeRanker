"""Technology skills dictionary and extraction utilities."""

import re
from dataclasses import dataclass, field

# Comprehensive technology skills organized by category
TECH_SKILLS: dict[str, list[str]] = {
    "programming_languages": [
        "python",
        "java",
        "java 8",
        "java 11",
        "java 17",
        "java 21",
        "javascript",
        "typescript",
        "c++",
        "c#",
        "c",
        "ruby",
        "go",
        "golang",
        "rust",
        "swift",
        "kotlin",
        "scala",
        "php",
        "perl",
        "r",
        "matlab",
        "dart",
        "lua",
        "haskell",
        "elixir",
        "clojure",
        "objective-c",
        "groovy",
        "shell",
        "bash",
        "powershell",
        "sql",
        "plsql",
        "vb.net",
        "visual basic",
        "cobol",
        "fortran",
        "assembly",
    ],
    "web_frameworks": [
        "react",
        "reactjs",
        "react.js",
        "angular",
        "angularjs",
        "vue",
        "vuejs",
        "vue.js",
        "svelte",
        "next.js",
        "nextjs",
        "nuxt",
        "nuxtjs",
        "gatsby",
        "django",
        "flask",
        "fastapi",
        "spring",
        "spring boot",
        "springboot",
        "spring cloud",
        "spring mvc",
        "spring security",
        "spring data",
        "hibernate",
        "jpa",
        "express",
        "expressjs",
        "nestjs",
        "rails",
        "ruby on rails",
        "asp.net",
        "asp.net core",
        "laravel",
        "symfony",
        "gin",
        "fiber",
        "blazor",
        "remix",
        "astro",
    ],
    "databases": [
        "mysql",
        "postgresql",
        "postgres",
        "mongodb",
        "redis",
        "elasticsearch",
        "sqlite",
        "oracle",
        "sql server",
        "mssql",
        "cassandra",
        "dynamodb",
        "couchdb",
        "neo4j",
        "mariadb",
        "cockroachdb",
        "firebase",
        "firestore",
        "supabase",
        "influxdb",
        "timescaledb",
        "memcached",
    ],
    "cloud_platforms": [
        "aws",
        "amazon web services",
        "azure",
        "microsoft azure",
        "gcp",
        "google cloud",
        "google cloud platform",
        "heroku",
        "digitalocean",
        "linode",
        "vercel",
        "netlify",
        "cloudflare",
        "alibaba cloud",
    ],
    "devops_tools": [
        "docker",
        "kubernetes",
        "k8s",
        "terraform",
        "ansible",
        "jenkins",
        "github actions",
        "gitlab ci",
        "circleci",
        "travis ci",
        "chef",
        "puppet",
        "vagrant",
        "helm",
        "argocd",
        "argo cd",
        "prometheus",
        "grafana",
        "datadog",
        "new relic",
        "splunk",
        "nginx",
        "apache",
        "caddy",
        "istio",
        "envoy",
    ],
    "data_science_ml": [
        "tensorflow",
        "pytorch",
        "keras",
        "scikit-learn",
        "sklearn",
        "pandas",
        "numpy",
        "scipy",
        "matplotlib",
        "seaborn",
        "jupyter",
        "spark",
        "pyspark",
        "hadoop",
        "hive",
        "airflow",
        "mlflow",
        "kubeflow",
        "sagemaker",
        "nltk",
        "spacy",
        "hugging face",
        "huggingface",
        "transformers",
        "opencv",
        "computer vision",
        "nlp",
        "natural language processing",
        "machine learning",
        "deep learning",
        "neural network",
        "random forest",
        "xgboost",
        "lightgbm",
        "catboost",
        "regression",
        "classification",
        "clustering",
        "generative ai",
        "llm",
        "large language model",
        "gpt",
        "bert",
    ],
    "mobile": [
        "react native",
        "flutter",
        "ionic",
        "xamarin",
        "swiftui",
        "android",
        "ios",
        "mobile development",
        "cordova",
        "capacitor",
    ],
    "version_control": [
        "git",
        "github",
        "gitlab",
        "bitbucket",
        "svn",
        "mercurial",
        "github copilot",
    ],
    "testing": [
        "junit",
        "pytest",
        "jest",
        "mocha",
        "chai",
        "cypress",
        "selenium",
        "playwright",
        "puppeteer",
        "testng",
        "rspec",
        "minitest",
        "unittest",
        "phpunit",
        "tdd",
        "bdd",
        "test driven development",
    ],
    "messaging_streaming": [
        "kafka",
        "rabbitmq",
        "activemq",
        "sqs",
        "sns",
        "pubsub",
        "celery",
        "sidekiq",
        "nats",
        "pulsar",
    ],
    "api_protocols": [
        "rest",
        "restful",
        "graphql",
        "grpc",
        "soap",
        "websocket",
        "openapi",
        "swagger",
        "postman",
    ],
    "security": [
        "oauth",
        "oauth2",
        "jwt",
        "saml",
        "ldap",
        "sso",
        "encryption",
        "ssl",
        "tls",
        "https",
        "owasp",
        "penetration testing",
        "vulnerability assessment",
    ],
    "other_tools": [
        "jira",
        "confluence",
        "slack",
        "trello",
        "figma",
        "sketch",
        "adobe xd",
        "webpack",
        "vite",
        "babel",
        "eslint",
        "prettier",
        "maven",
        "gradle",
        "npm",
        "yarn",
        "pnpm",
        "pip",
        "poetry",
        "linux",
        "unix",
        "windows server",
        "agile",
        "scrum",
        "kanban",
        "ci/cd",
        "cicd",
        "microservices",
        "serverless",
        "event driven",
        "design patterns",
        "sdlc",
        "solid",
        "oop",
        "functional programming",
        "data structures",
        "algorithms",
    ],
}

# Flatten all skills for quick lookup
ALL_SKILLS: set[str] = set()
for category_skills in TECH_SKILLS.values():
    ALL_SKILLS.update(category_skills)


@dataclass
class SemanticMatchResult:
    """Result of matching a single unknown JD skill against resume sentences.

    Provides full transparency into how a skill was matched (or rejected),
    including the sentence that triggered the match, the similarity score,
    the match type (literal / normalized / semantic), and the confidence
    level derived from context analysis.
    """

    skill: str
    matched_sentence: str
    similarity_score: float
    match_type: str  # "literal" | "normalized_literal" | "semantic"
    confidence: str  # "high" | "medium" | "low"
    context: str  # "strong" | "negated" | "learning_only" | "weak_exposure" | "unknown"


def detect_skill_context(sentence: str, skill: str) -> str:
    """Classify how a skill is mentioned in a sentence.

    Checks for three types of non-committal contexts:
      - negated:      candidate explicitly states they lack the skill
                      (e.g. "no experience with Docker")
      - learning_only: candidate is still acquiring the skill
                      (e.g. "currently learning Kubernetes")
      - weak_exposure: candidate has only superficial familiarity
                      (e.g. "basic exposure to AWS")

    Returns one of: "negated", "learning_only", "weak_exposure", "strong", "unknown".
    """
    if not sentence or not skill:
        return "unknown"

    sent_lower = sentence.lower()
    skill_lower = skill.lower()

    if skill_lower not in sent_lower:
        return "unknown"

    # Pre-process to avoid false triggers from compound skill names.
    # Replace common compound terms so their component words don't
    # individually match learning/weak patterns (e.g. "learning" in
    # "machine learning" should not flag as learning_only).
    processed = sent_lower
    processed = processed.replace("deep learning", " [[deep_learning_compound]] ")
    processed = processed.replace("machine learning", " [[machine_learning_compound]] ")

    # 1. Negation patterns — candidate explicitly says they lack this skill
    # Use .*? to allow intervening words (e.g. "no docker experience" → matches)
    NEGATION_PATTERNS = [
        r"\bno\b.*?\bexperience\b",
        r"\bnot\b.*?\bexperienced\b",
        r"\bnever\b.*?\bworked\b",
        r"\bnot\b.*?\bworked\b",
        r"\bno\b.*?\bhands?-?\s*on\b",
        r"\bwithout\b.*?\bexperience\b",
        r"\black\s+of\b.*?\bexperience\b",
        r"\bunfamiliar\b.*?\bwith\b",
    ]

    for pat in NEGATION_PATTERNS:
        if re.search(pat, processed):
            return "negated"

    # 2. Learning-only patterns — candidate is still acquiring the skill
    #    Uses \blearn\w*\b to cover "learn", "learning", "learned", "learnt", etc.
    #    The check is skipped when the skill name itself contains "learning"
    #    to avoid false flags for skills like "machine learning".
    LEARNING_PATTERNS = [
        r"currently\s+(?:learning|studying|taking)",
        r"\blearn\w*\b",
        r"\bbeginner\b",
        r"basic\s+understanding",
        r"basic\s+knowledge",
        r"just\s+begun",
    ]

    for pat in LEARNING_PATTERNS:
        if re.search(pat, processed):
            if "learning" in skill_lower:
                continue  # skill is something like "machine learning"
            return "learning_only"

    # 3. Weak exposure patterns — surface-level / beginner familiarity
    WEAK_PATTERNS = [
        r"familiar\s+with",
        r"exposure\s+to",
        r"interested\s+(?:in|to)",
        r"\bexploring\b",
        r"self.?learning",
        r"some\s+knowledge",
        r"basic\s+exposure",
        r"basic\s+experience",
    ]

    for pat in WEAK_PATTERNS:
        if re.search(pat, processed):
            return "weak_exposure"

    return "strong"


def _get_overall_skill_context(resume_text: str, skill: str) -> str:
    """Determine the overall context of a skill across ALL its mentions in the resume.

    Checks every sentence that mentions the skill and returns the aggregate context:
      - "negated"       → every mention is negated
      - "learning_only" → mentions include learning signals (but no strong signal)
      - "weak_exposure" → mentions include weak signals (but no strong signal)
      - "strong"        → at least one mention is positive/strong
      - "unknown"       → skill not found in any sentence
    """
    # Split on sentence punctuation AND newlines to handle bullet-pointed text
    # (e.g. "• i have no docker experience" with no period at end)
    raw_parts = re.split(r"(?:[.!?\n]|^\s*[•\-**])\s*", resume_text, flags=re.MULTILINE)
    sentences = [p.strip(" \t\n\r•\-*") for p in raw_parts if len(p.strip(" \t\n\r•\-*")) > 10]
    if not sentences:
        sentences = [resume_text[:500]]

    contexts: list[str] = []
    for sent in sentences:
        if skill.lower() in sent.lower():
            ctx = detect_skill_context(sent, skill)
            contexts.append(ctx)

    if not contexts:
        return "unknown"

    # If ANY mention is strong, the overall context is strong (positive outweighs negative)
    if any(c == "strong" for c in contexts):
        return "strong"

    # Mixed signals where at least one mention is not negated → benefit of the doubt
    if any(c != "negated" for c in contexts) and any(c == "negated" for c in contexts):
        return "weak_exposure"

    # All mentions are the same context
    if all(c == "negated" for c in contexts):
        return "negated"
    if all(c == "learning_only" for c in contexts):
        return "learning_only"
    if all(c == "weak_exposure" for c in contexts):
        return "weak_exposure"

    # Fallback: return the most common non-strong context
    return max(set(contexts), key=contexts.count)


def _find_sentence_with_skill(sentences: list[str], skill: str) -> str | None:
    """Return the first sentence that contains the given skill (case-insensitive)."""
    skill_lower = skill.lower()
    for sent in sentences:
        if skill_lower in sent.lower():
            return sent
    return None


_SECTION_HEADER_LIKE = re.compile(
    r"^(?:(?:required|preferred|core|key|technical|essential|must\s+have|nice\s+to\s+have)\s+)?"
    r"(?:skills?|qualifications?|competencies|experience|requirements?)"
    r"(?:\s*[&]\s*(?:qualifications?|experience|competencies|requirements?))?\s*:?\s*$",
    re.I,
)

_SUB_HEADER_LIKE = re.compile(
    r"^(?:technical|leadership|management|core|functional|domain|industry|additional|"
    r"general|professional|cross-functional|soft|analytical|business|education|"
    r"certifications?)\s+(?:expertise|skills?|experience|competencies|knowledge|"
    r"qualifications?|abilities?)"
    r"(?:\s*[&]\s*(?:management|experience|skills?|knowledge))?\s*$",
    re.I,
)

# Words that are common in JD headings / descriptions but are NOT skills
_NON_SKILL_WORDS: set[str] = {
    "leadership",
    "management",
    "communication",
    "collaboration",
    "interpersonal",
    "organizational",
    "analytical",
    "problem-solving",
    "problem solving",
    "teamwork",
    "mentorship",
    "stakeholder",
    "cross-functional",
    "cross functional",
    "decision-making",
    "decision making",
    "strategic",
    "planning",
    "innovation",
    "ownership",
    "accountability",
    "initiative",
    "adaptability",
    "flexibility",
    "creativity",
    "critical thinking",
    "attention to detail",
    "time management",
    "project management",
    "people management",
    "team management",
    "technical expertise",
    "technical skills",
    "soft skills",
    "core competencies",
    "system architecture",
    "systems architecture",
    "software engineering",
    "production systems",
    "high-traffic systems",
    "scalable infrastructure",
    "deployment strategies",
    "development workflows",
    "engineering teams",
    "development lifecycle",
    "product development",
    "business goals",
    "company goals",
    "technical roadmaps",
    "resource planning",
    "operational excellence",
    "continuous improvement",
    "best practices",
    "coding standards",
    "performance optimization",
    "production systems",
    "high availability",
    "cross-functional teams",
}

_DESC_PHRASE = re.compile(
    r"^(?:experience\s+with|experience\s+in|expertise\s+in|knowledge\s+of|"
    r"understanding\s+of|proficiency\s+in|familiarity\s+with|background\s+in|"
    r"proven\s+experience|ability\s+to|proficient\s+in|skilled\s+in|"
    r"exposure\s+to|hands?.on\s+experience|working\s+knowledge|"
    r"strong\s+(?:background|knowledge|understanding|experience|proficiency|familiarity|skills|sense|track\s+record|problem.solving|analytical|communication|leadership|technical|foundation|expertise)|"
    r"deep\s+(?:understanding|knowledge|expertise|experience|familiarity)|"
    r"solid\s+(?:understanding|knowledge|background|experience|expertise)|"
    r"extensive\s+(?:experience|knowledge|background|expertise)|"
    r"demonstrated\s+(?:experience|ability|expertise|success|track\s+record)|"
    r"proven\s+(?:ability|track\s+record|success|experience)|"
    r"hand[-\s]on\s+(?:experience|knowledge|expertise))\b",
    re.I,
)


def _clean_phrase(text: str) -> str | None:
    text = text.strip().strip("*").strip()
    text = re.sub(r"\s*\(.*?\)\s*", "", text).strip()
    if not text or len(text) <= 1 or len(text) > 50:
        return None
    # Strip leading conjunctions like "and ", "or "
    text = re.sub(r"^(?:and|or)\s+", "", text, flags=re.I).strip()
    # Strip leading filler like "such as ", "including "
    text = re.sub(r"^(?:such\s+as|including|e\.g\.|i\.e\.)\s+", "", text, flags=re.I).strip()
    if not text or len(text) <= 1:
        return None
    return text


def _extract_skill_phrases(text: str) -> list[str]:
    phrases: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        cleaned = re.sub(r"^[\s#*•\-·]+", "", line).strip()
        if not cleaned:
            continue
        if _SECTION_HEADER_LIKE.match(cleaned):
            continue
        if _SUB_HEADER_LIKE.match(cleaned):
            continue
        parts = re.split(r"\s*,\s*", cleaned)
        for part in parts:
            part = part.strip()
            sub_parts = re.split(r"\s+/\s+|\s+&\s+", part)
            for sub in sub_parts:
                sub = _clean_phrase(sub)
                if sub is None:
                    continue
                if _DESC_PHRASE.match(sub):
                    continue
                if sub.lower() in _NON_SKILL_WORDS:
                    continue
                phrases.append(sub)
    return phrases


def extract_unknown_skills_semantic(
    required_skills_text: str,
    resume_text: str,
    threshold: float = 0.45,
) -> tuple[set[str], set[str], list[SemanticMatchResult]]:
    """Detect JD skills not covered by the dictionary using literal + semantic matching.

    Applies a 4-tier matching pipeline:
      Tier 2 — literal substring match
      Tier 3 — normalized literal match (strip punctuation)
      Tier 4 — sentence-level semantic match via sentence-transformers

    Before accepting any match, the resume sentence is checked for negation,
    learning-only, or weak-exposure context. Negated skills are never matched.
    Learning-only and weak-exposure skills are excluded from the matched set
    but are recorded in the match_details list with low confidence.

    Returns (matched_skills, unknown_skills, match_details) where:
      - matched        -> unknown skills the resume genuinely covers (high confidence)
      - unknown        -> every unknown skill phrase found in the JD
      - match_details  -> full SemanticMatchResult list for every attempted match
                          (including rejected ones with context and confidence)
    """
    phrases = _extract_skill_phrases(required_skills_text)

    unknown: set[str] = set()
    for phrase in phrases:
        if phrase.lower() not in ALL_SKILLS:
            unknown.add(phrase.lower())

    if not unknown:
        return set(), set(), []

    # Split resume into sentences (including on newlines for bullet-pointed text)
    raw_parts = re.split(r"(?:[.!?\n]|^\s*[•\-**])\s*", resume_text, flags=re.MULTILINE)
    sentences = [p.strip(" \t\n\r•\-*") for p in raw_parts if len(p.strip(" \t\n\r•\-*")) > 10]
    if not sentences:
        sentences = [resume_text[:500]]

    resume_lower = resume_text.lower()
    matched: set[str] = set()
    needs_semantic: set[str] = set()
    details: list[SemanticMatchResult] = []

    # ------------------------------------------------------------------
    # Tier 2: Literal substring match
    # ------------------------------------------------------------------
    for skill in unknown:
        if skill in resume_lower:
            match_sent = _find_sentence_with_skill(sentences, skill)
            ctx = detect_skill_context(match_sent, skill) if match_sent else "strong"

            if ctx == "negated":
                # Negated skills are completely excluded from matched set
                details.append(
                    SemanticMatchResult(
                        skill=skill,
                        matched_sentence=match_sent or "",
                        similarity_score=1.0,
                        match_type="literal",
                        confidence="low",
                        context=ctx,
                    )
                )
                continue

            confidence = "high" if ctx == "strong" else "medium"
            # Weak/learning-only literal matches still need to be counted
            # since the skill name literally appears; context is noted.
            matched.add(skill)
            details.append(
                SemanticMatchResult(
                    skill=skill,
                    matched_sentence=match_sent or "",
                    similarity_score=1.0,
                    match_type="literal",
                    confidence=confidence,
                    context=ctx,
                )
            )
        else:
            needs_semantic.add(skill)

    # ------------------------------------------------------------------
    # Tier 3: Normalized literal match (strip punctuation/parentheses)
    # ------------------------------------------------------------------
    if needs_semantic:
        resume_normalized = re.sub(r"[()\[\]{}]", " ", resume_lower)
        resume_normalized = re.sub(r"\s+", " ", resume_normalized)

        still_needs: set[str] = set()
        for skill in needs_semantic:
            skill_normalized = re.sub(r"[()\[\]{}]", " ", skill)
            skill_normalized = re.sub(r"\s+", " ", skill_normalized).strip()
            if skill_normalized in resume_normalized:
                # Try finding the sentence with original or normalized skill text
                match_sent = _find_sentence_with_skill(sentences, skill)
                if not match_sent:
                    match_sent = _find_sentence_with_skill(sentences, skill_normalized)
                ctx = detect_skill_context(match_sent, skill) if match_sent else "strong"

                if ctx == "negated":
                    details.append(
                        SemanticMatchResult(
                            skill=skill,
                            matched_sentence=match_sent or "",
                            similarity_score=1.0,
                            match_type="normalized_literal",
                            confidence="low",
                            context=ctx,
                        )
                    )
                    continue

                confidence = "high" if ctx == "strong" else "medium"
                matched.add(skill)
                details.append(
                    SemanticMatchResult(
                        skill=skill,
                        matched_sentence=match_sent or "",
                        similarity_score=1.0,
                        match_type="normalized_literal",
                        confidence=confidence,
                        context=ctx,
                    )
                )
            else:
                still_needs.add(skill)

        # ------------------------------------------------------------------
        # Tier 4: Sentence-level semantic match (all-MiniLM-L6-v2)
        # ------------------------------------------------------------------
        if still_needs:
            from sklearn.metrics.pairwise import cosine_similarity
            from resume_ranker.semantic import _get_model

            model = _get_model()
            sent_embs = model.encode(sentences, convert_to_numpy=True)

            for skill in still_needs:
                skill_emb = model.encode([skill], convert_to_numpy=True)
                sims = cosine_similarity(skill_emb, sent_embs)[0]
                best_idx = int(sims.argmax())
                sim = float(sims[best_idx])

                if sim < threshold:
                    continue  # Below similarity threshold, no match

                best_sentence = sentences[best_idx] if sentences else ""
                ctx = detect_skill_context(best_sentence, skill) if best_sentence else "unknown"

                if ctx == "negated":
                    # Negated semantic matches are discarded entirely
                    details.append(
                        SemanticMatchResult(
                            skill=skill,
                            matched_sentence=best_sentence,
                            similarity_score=sim,
                            match_type="semantic",
                            confidence="low",
                            context=ctx,
                        )
                    )
                    continue

                if ctx in ("learning_only", "weak_exposure"):
                    # Learning-only and weak semantic matches are excluded from the
                    # matched set because the sentence does not demonstrate genuine
                    # proficiency — it indicates the candidate is still learning or
                    # has only superficial familiarity. The match is still recorded
                    # with low confidence for debugging/transparency.
                    details.append(
                        SemanticMatchResult(
                            skill=skill,
                            matched_sentence=best_sentence,
                            similarity_score=sim,
                            match_type="semantic",
                            confidence="low",
                            context=ctx,
                        )
                    )
                    continue

                # Strong semantic match — candidate demonstrates genuine proficiency
                matched.add(skill)
                details.append(
                    SemanticMatchResult(
                        skill=skill,
                        matched_sentence=best_sentence,
                        similarity_score=sim,
                        match_type="semantic",
                        confidence="high",
                        context=ctx,
                    )
                )

    return matched, unknown, details


def extract_skills(text: str) -> dict[str, list[str]]:
    """Extract technology skills from text, grouped by category.

    Returns a dict mapping category name to list of matched skills.
    """
    text_lower = text.lower()
    found: dict[str, list[str]] = {}

    for category, skills in TECH_SKILLS.items():
        matched = []
        for skill in skills:
            pattern = r"(?<![a-zA-Z])" + re.escape(skill) + r"(?![a-zA-Z])"
            if re.search(pattern, text_lower):
                matched.append(skill)
        if matched:
            found[category] = matched

    return found


def flatten_skills(categorized: dict[str, list[str]]) -> set[str]:
    """Flatten categorized skills into a single set."""
    result: set[str] = set()
    for skills in categorized.values():
        result.update(skills)
    return result


# Skill inference graph: if a candidate has skill X, they likely also know Y.
# Maps a skill to a set of inferred skills.
SKILL_INFERENCE: dict[str, set[str]] = {
    "spring boot": {"distributed systems", "backend engineering", "rest", "microservices"},
    "spring cloud": {"distributed systems", "microservices", "cloud-native"},
    "kafka": {"distributed systems", "event-driven architecture", "streaming"},
    "rabbitmq": {"distributed systems", "event-driven architecture", "messaging"},
    "docker": {"containerization", "devops", "cloud-native"},
    "kubernetes": {"container orchestration", "devops", "cloud-native", "infrastructure"},
    "microservices": {"distributed systems", "api design", "backend engineering"},
    "aws": {"cloud computing", "infrastructure", "cloud-native"},
    "azure": {"cloud computing", "infrastructure", "cloud-native"},
    "gcp": {"cloud computing", "infrastructure", "cloud-native"},
    "react": {"frontend development", "spa", "component-based architecture"},
    "angular": {"frontend development", "spa", "component-based architecture"},
    "vue": {"frontend development", "spa", "component-based architecture"},
    "django": {"backend engineering", "web development", "orm"},
    "fastapi": {"backend engineering", "api design", "async programming"},
    "flask": {"backend engineering", "web development"},
    "python": {"scripting", "automation"},
    "java": {"enterprise development", "oop"},
    "tensorflow": {"deep learning", "ai", "neural networks"},
    "pytorch": {"deep learning", "ai", "neural networks"},
    "postgresql": {"relational databases", "data modeling"},
    "mongodb": {"nosql", "document databases"},
    "redis": {"caching", "in-memory databases"},
    "jenkins": {"ci/cd", "automation", "devops"},
    "github actions": {"ci/cd", "automation", "devops"},
    "terraform": {"infrastructure as code", "cloud automation", "devops"},
    "rest": {"api design", "web services"},
    "restful": {"api design", "web services"},
    "graphql": {"api design", "web services"},
    "agile": {"project management", "scrum"},
    "scrum": {"agile", "project management"},
    "ci/cd": {"automation", "devops", "deployment"},
    "elasticsearch": {"search", "analytics", "distributed systems"},
    "git": {"version control", "collaboration"},
    "github copilot": {"ai-assisted development", "ai tools", "productivity"},
}


def infer_skills(explicit_skills: set[str]) -> set[str]:
    """Infer additional skills from explicitly mentioned skills.

    Returns only the inferred skills (not the explicit ones).
    """
    inferred: set[str] = set()
    for skill in explicit_skills:
        if skill in SKILL_INFERENCE:
            inferred.update(SKILL_INFERENCE[skill])
    return inferred - explicit_skills
