# app.py — InterviewSense 2.0: GPT-Driven, Voice-Interactive AI Interviewer
# Run with:  streamlit run app.py
#
# Recommended requirements.txt:
# streamlit
# langchain-openai>=0.2.0
# langchain-core>=0.2.0
# SpeechRecognition
# pyaudio              # (if installing is hard on Windows, use prebuilt wheels)
# textblob
# python-dotenv        # optional
#
# Optional: initialize TextBlob corpora once in Python REPL:
# >>> import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')
#
# Set your OpenAI key (optional, for GPT questions & feedback):
# Linux/Mac: export OPENAI_API_KEY=sk-...
# Windows  : setx OPENAI_API_KEY "sk-..."

import os
import json
import time
import random
from typing import Dict, List, Tuple, Optional

import streamlit as st

# ---- Optional LLM (GPT) support via LangChain-OpenAI ----
try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None

# ---- Voice recognition (local mic) ----
try:
    import speech_recognition as sr
except Exception:
    sr = None

# ---- Tone / Sentiment ----
try:
    from textblob import TextBlob
except Exception:
    TextBlob = None

# ------------------------------
# Config & Constants
# ------------------------------

st.set_page_config(page_title="InterviewSense 2.0 — AI Interviewer", page_icon="💬", layout="wide")
FLAG_WORDS = [
    "umm", "uh", "er", "hmm", "ah", "like", "you know", "i guess",
    "kinda", "sort of", "maybe", "i think", "perhaps", "not sure"
]
HEDGE_WORDS = ["maybe", "i think", "not sure", "perhaps", "guess", "kind of", "sort of"]

LEVELS = ["Beginner", "Intermediate", "Advanced"]

# Fallback static bank used if GPT not available (also for keyword hints in scoring)
QUESTION_BANK: Dict[str, Dict[str, List[Dict[str, List[str]]]]] = {
    "Machine Learning": {
        "Beginner": [
            {"q": "What is supervised learning?", "kw": ["labeled", "target", "mapping", "inputs", "outputs"]},
            {"q": "Define overfitting.", "kw": ["train", "test", "generalize", "high variance", "complex"]},
            {"q": "Classification vs regression?", "kw": ["discrete", "continuous", "labels", "numeric"]},
        ],
        "Intermediate": [
            {"q": "Explain bias-variance tradeoff.", "kw": ["bias", "variance", "underfitting", "overfitting", "complexity"]},
            {"q": "How does regularization help?", "kw": ["penalty", "L1", "L2", "weights", "overfitting"]},
            {"q": "Walk me through cross-validation.", "kw": ["k-fold", "validation", "holdout", "leakage", "generalization"]},
        ],
        "Advanced": [
            {"q": "Compare XGBoost vs Random Forest.", "kw": ["boosting", "bagging", "trees", "overfitting", "regularization"]},
            {"q": "Explain attention in transformers.", "kw": ["queries", "keys", "values", "weights", "context"]},
            {"q": "Design an anomaly detector for payments.", "kw": ["imbalance", "precision", "recall", "threshold", "ROC"]},
        ],
    },
    "Web Development": {
        "Beginner": [
            {"q": "What is a REST API?", "kw": ["HTTP", "resources", "GET", "POST", "stateless"]},
            {"q": "Explain client vs server.", "kw": ["browser", "backend", "requests", "responses"]},
        ],
        "Intermediate": [
            {"q": "Walk through HTTP status codes.", "kw": ["200", "404", "500", "status", "error"]},
            {"q": "What is CORS and why needed?", "kw": ["origin", "headers", "security", "browsers"]},
        ],
        "Advanced": [
            {"q": "Design scalable web app architecture.", "kw": ["load balancer", "cache", "database", "stateless", "microservices"]},
        ],
    },
    "Data Structures": {
        "Beginner": [
            {"q": "Stack vs Queue?", "kw": ["LIFO", "FIFO", "operations", "push", "pop", "enqueue", "dequeue"]},
            {"q": "What is a hash table?", "kw": ["key", "value", "collision", "map"]},
        ],
        "Intermediate": [
            {"q": "Balance in BSTs and why?", "kw": ["height", "log n", "rotation", "AVL", "red-black"]},
        ],
        "Advanced": [
            {"q": "Design an LRU cache.", "kw": ["evict", "recent", "capacity", "doubly", "hash"]},
        ],
    },
    "HR": {
        "Beginner": [
            {"q": "Tell me about yourself.", "kw": ["background", "experience", "skills"]},
            {"q": "Strengths and weaknesses?", "kw": ["strength", "weakness", "improve"]},
        ],
        "Intermediate": [
            {"q": "Describe a conflict and resolution.", "kw": ["conflict", "resolve", "communication", "team"]},
        ],
        "Advanced": [
            {"q": "How do you handle tight deadlines?", "kw": ["prioritize", "plan", "communicate", "deliver"]},
        ],
    },
}

# ------------------------------
# LLM + Memory Utilities
# ------------------------------

def get_llm() -> Optional[object]:
    """Instantiate ChatOpenAI if available and API key is set."""
    if ChatOpenAI is None:
        return None
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        return None
    try:
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    except Exception:
        return None

LLM = get_llm()

def load_memory(path: str = "interviewsense_memory.json") -> Dict:
    if os.path.exists(path):
        try:
            return json.load(open(path, "r", encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_memory(mem: Dict, path: str = "interviewsense_memory.json") -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(mem, f, indent=2)
    except Exception:
        pass

# ------------------------------
# Question Agent (GPT + fallback)
# ------------------------------

def sample_bank_question(domain: str, level_idx: int, asked: List[str]) -> Tuple[Dict, int]:
    level = LEVELS[level_idx]
    pool = [q for q in QUESTION_BANK[domain][level] if q["q"] not in asked]
    if not pool and level_idx < len(LEVELS) - 1:
        return sample_bank_question(domain, level_idx + 1, asked)
    if not pool and level_idx > 0:
        return sample_bank_question(domain, level_idx - 1, asked)
    if not pool:
        return {"q": "No more questions in this domain/level.", "kw": []}, level_idx
    q = random.choice(pool)
    return q, level_idx

def generate_gpt_question(domain: str, level: str, last_answer: str = "") -> str:
    """Use GPT to generate a single interview question."""
    if LLM is None:
        return ""
    prompt = (
        f"You are a professional interviewer for the domain: {domain}.\n"
        f"Generate ONE {level.lower()}-level interview question.\n"
        f"If a candidate's previous answer is given, make this question a natural follow-up that's slightly harder, but concise.\n"
        f"Only output the question text.\n"
    )
    if last_answer:
        prompt += f"\nCandidate previous answer (for context): {last_answer}\n"
    try:
        q = LLM.invoke(prompt).content.strip()
        # Ensure it ends with a question mark for neatness
        if not q.endswith("?"):
            q = q.rstrip(".") + "?"
        return q
    except Exception:
        return ""

def pick_next_question(domain: str, level_idx: int, asked: List[str], use_gpt: bool, last_answer: str) -> Tuple[Dict, int]:
    """Decide next question; use GPT when available else fallback to bank."""
    level = LEVELS[level_idx]
    if use_gpt and LLM is not None:
        q_text = generate_gpt_question(domain, level, last_answer)
        if q_text:
            # Try to infer keywords from domain templates (lightweight hinting)
            kw = []
            fallback = QUESTION_BANK.get(domain, {}).get(level, [])
            if fallback:
                # take union of up to 2 random templates' keywords as hints
                hints = random.sample(fallback, k=min(2, len(fallback)))
                for h in hints:
                    for k in h.get("kw", []):
                        if k not in kw:
                            kw.append(k)
            return {"q": q_text, "kw": kw[:6]}, level_idx
    # Fallback to static bank
    return sample_bank_question(domain, level_idx, asked)

# ------------------------------
# Voice Agent (Speech to Text)
# ------------------------------

def record_voice(timeout: int = 5, phrase_time_limit: int = 30) -> str:
    """Capture audio from microphone and transcribe using Google Web Speech API."""
    if sr is None:
        st.error("SpeechRecognition module is not available. Install 'SpeechRecognition' and 'pyaudio'.")
        return ""
    try:
        r = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("🎤 Listening... Speak now.")
            audio = r.listen(source, timeout=15, phrase_time_limit=90)
        st.caption("Transcribing…")
        text = r.recognize_google(audio)  # free web recognizer
        st.success("✅ Recognized: " + text)
        return text
    except Exception as e:
        st.error(f"Voice recognition failed: {e}")
        return ""

# ------------------------------
# Evaluator Agent
# ------------------------------

def detect_flags(answer: str) -> List[str]:
    ans = answer.lower()
    flags = []
    for f in FLAG_WORDS:
        # match as substring (simple, effective for demo)
        if f in ans:
            flags.append(f)
    # Deduplicate while preserving order
    seen = set()
    unique = [f for f in flags if not (f in seen or seen.add(f))]
    return unique

def evaluate_answer(answer: str, keywords: List[str]) -> Dict:
    aw = answer.strip().lower()

    # Technical score via keyword coverage
    tech = 0.0
    for k in keywords:
        if k.lower() in aw:
            tech += 10.0 / max(1, len(keywords))
    tech = round(max(0.0, min(10.0, tech)), 1)

    # Clarity: length + basic structure heuristic
    clarity = round(min(10.0, 6.0 + (len(answer.split()) / 40.0) * 4.0), 1)

    # Confidence: hedges & filler words reduce
    flags = detect_flags(answer)
    conf = 9.5
    if any(h in aw for h in HEDGE_WORDS):
        conf -= 1.0
    conf -= min(2.0, 0.5 * len(flags))  # cap penalty
    conf = round(max(0.0, min(10.0, conf)), 1)

    # Sentiment/tone nudge
    tone_comment = ""
    if TextBlob is not None:
        try:
            polarity = TextBlob(answer).sentiment.polarity  # -1..+1
            if polarity < -0.25:
                conf = max(0.0, conf - 0.3)
                tone_comment = "Tone slightly negative/uncertain."
            elif polarity > 0.4:
                conf = min(10.0, conf + 0.2)
                tone_comment = "Positive, confident tone detected."
        except Exception:
            pass

    # Optional LLM commentary
    commentary = ""
    if LLM is not None:
        try:
            commentary = LLM.invoke(
                "Evaluate this interview answer in 1-2 constructive sentences. "
                "Be specific, concise, and focus on what to add:\n"
                f"Answer: {answer}"
            ).content
        except Exception:
            commentary = ""

    overall = round((tech + clarity + conf) / 3.0, 1)
    return {
        "technical": tech,
        "clarity": clarity,
        "confidence": conf,
        "overall": overall,
        "flags": flags,
        "tone": tone_comment,
        "commentary": commentary
    }

# ------------------------------
# Feedback Agent
# ------------------------------

def feedback_agent(question: str, answer: str, keywords: List[str], eval_res: Dict) -> str:
    missing = [k for k in keywords if k.lower() not in answer.lower()]
    base = []

    if missing:
        base.append("Consider mentioning: " + ", ".join(missing[:6]))
    if eval_res["clarity"] < 7:
        base.append("Structure your response: definition → key points → short example.")
    if eval_res["technical"] < 7:
        base.append("Tighten the technical core and define key terms explicitly.")
    if eval_res.get("flags"):
        base.append(f"Reduce filler words: {', '.join(eval_res['flags'])}. Replace with brief pauses.")
    if eval_res.get("tone"):
        base.append(eval_res["tone"])

    if LLM is not None:
        try:
            tips = LLM.invoke(
                "You are a senior interviewer and coach. "
                "Provide 2 crisp bullet tips (<=20 words each) to improve this answer:\n"
                f"Question: {question}\n"
                f"Answer: {answer}\n"
            ).content
            base.append(tips)
        except Exception:
            pass

    if not base:
        base = ["Excellent answer. Add a concrete example to make it outstanding."]
    return "\n- " + "\n- ".join(base)

# ------------------------------
# Reporter Agent
# ------------------------------

def build_report(transcript: List[Dict]) -> Tuple[Dict, str]:
    if not transcript:
        return {}, "No answers recorded."

    tech = sum(t["scores"]["technical"] for t in transcript) / len(transcript)
    clar = sum(t["scores"]["clarity"] for t in transcript) / len(transcript)
    conf = sum(t["scores"]["confidence"] for t in transcript) / len(transcript)
    overall = round((tech + clar + conf) / 3.0, 1)

    summary = {
        "Technical Accuracy": round(tech, 1),
        "Clarity": round(clar, 1),
        "Confidence": round(conf, 1),
        "Overall": overall,
        "Questions Answered": len(transcript),
    }

    md = ["## 🧾 InterviewSense Report", ""]
    md.append(f"Technical Accuracy: *{summary['Technical Accuracy']} / 10*")
    md.append(f"Clarity: *{summary['Clarity']} / 10*")
    md.append(f"Confidence: *{summary['Confidence']} / 10*")
    md.append(f"Overall: *{summary['Overall']} / 10*")
    md.append("")
    md.append("### Highlights & Tips")
    for t in transcript[-3:]:
        md.append(f"- *Q:* {t['question']} — *Score:* {t['scores']['overall']}/10")
        if t["scores"].get("flags"):
            md.append(f"  \n  ⚠ Flags: {', '.join(t['scores']['flags'])}")
        if t.get("feedback"):
            md.append(f"  \n  {t['feedback']}")
    return summary, "\n".join(md)

# ------------------------------
# UI — Streamlit App
# ------------------------------

st.title("💬 InterviewSense 2.0 — GPT + Voice Agentic Interviewer")

with st.sidebar:
    st.subheader("🎯 Session Setup")
    domain = st.selectbox("Domain", list(QUESTION_BANK.keys()), index=0)
    level = st.selectbox("Level", LEVELS, index=1)
    max_q = st.slider("Number of Questions", 1, 10, 5)
    use_gpt = st.toggle("Use GPT for dynamic questions (if API key set)", value=True if LLM else False)
    use_voice = st.toggle("Use voice input (SpeechRecognition)", value=bool(sr))
    st.caption("Tip: Set OPENAI_API_KEY for GPT features. Install SpeechRecognition + PyAudio for mic.")

# Load memory and greet
memory = load_memory()
if memory:
    st.info(f"📈 Welcome back! Previous sessions — average overall: {memory.get('avg_overall', 'N/A')}, "
            f"filler count total: {memory.get('filler_total', 0)}")

# Session state
if "state" not in st.session_state:
    st.session_state.state = {
        "level_idx": LEVELS.index(level),
        "asked": [],
        "transcript": [],
        "current": None,
        "started": False,
        "done": False,
        "domain": domain,
        "recognized": "",  # voice transcript buffer
        "last_answer": "",
    }

# Reset if domain/level changed
if st.session_state.state["domain"] != domain:
    st.session_state.state = {
        "level_idx": LEVELS.index(level),
        "asked": [],
        "transcript": [],
        "current": None,
        "started": False,
        "done": False,
        "domain": domain,
        "recognized": "",
        "last_answer": "",
    }

col1, col2, col3 = st.columns(3)
with col1: st.header("🧠 Strategy / Question")
with col2: st.header("🧪 Evaluation")
with col3: st.header("🔧 Feedback & Report")

# Controls
start_col, next_col, finish_col = st.columns([1,1,1])
with start_col:
    if not st.session_state.state["started"] and st.button("Start Interview", use_container_width=True):
        st.session_state.state["started"] = True

# Main flow
if st.session_state.state["started"] and not st.session_state.state["done"]:

    # Get/prepare current question
    if st.session_state.state["current"] is None and len(st.session_state.state["asked"]) < max_q:
        q, lvl = pick_next_question(
            domain,
            st.session_state.state["level_idx"],
            st.session_state.state["asked"],
            use_gpt=use_gpt and (LLM is not None),
            last_answer=st.session_state.state["last_answer"]
        )
        st.session_state.state["current"] = q
        st.session_state.state["level_idx"] = lvl
        st.session_state.state["asked"].append(q["q"])

    current = st.session_state.state["current"]

    if current:
        with col1:
            st.markdown(f"*Level:* {LEVELS[st.session_state.state['level_idx']]}  ")
            st.markdown(f"*Question {len(st.session_state.state['asked'])}/{max_q}:* {current['q']}")

            # Answer input (voice + text)
            user_ans_key = f"ans_{len(st.session_state.state['asked'])}"
            default_text = st.session_state.state["recognized"] or ""
            user_ans = st.text_area("Your Answer", key=user_ans_key, value=default_text, height=140)

            voice_cols = st.columns([1,1,2])
            with voice_cols[0]:
                if use_voice and st.button("🎤 Record Answer"):
                    text = record_voice()
                    if text:
                        st.session_state.state["recognized"] = text
                        st.rerun()
            with voice_cols[1]:
                if st.button("🧹 Clear"):
                    st.session_state.state["recognized"] = ""
                    st.session_state[user_ans_key] = ""

            submit = st.button("Submit Answer", type="primary")

        # On submit → evaluate, adapt, store
        if submit and (user_ans.strip() or st.session_state.state["recognized"].strip()):
            final_ans = user_ans.strip() or st.session_state.state["recognized"].strip()
            # Evaluate
            scores = evaluate_answer(final_ans, current.get("kw", []))
            fb = feedback_agent(current["q"], final_ans, current.get("kw", []), scores)

            # Adaptive level movement
            if scores["overall"] >= 8.0 and st.session_state.state["level_idx"] < len(LEVELS) - 1:
                st.session_state.state["level_idx"] += 1
            elif scores["overall"] <= 4.5 and st.session_state.state["level_idx"] > 0:
                st.session_state.state["level_idx"] -= 1

            record = {
                "question": current["q"],
                "answer": final_ans,
                "scores": scores,
                "feedback": fb,
                "level": LEVELS[st.session_state.state["level_idx"]],
            }
            st.session_state.state["transcript"].append(record)
            st.session_state.state["last_answer"] = final_ans
            st.session_state.state["current"] = None
            st.session_state.state["recognized"] = ""

        # Show last eval/feedback live
        if st.session_state.state["transcript"]:
            last = st.session_state.state["transcript"][-1]
            with col2:
                st.subheader("Latest Evaluation")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Technical", last["scores"]["technical"])
                m2.metric("Clarity", last["scores"]["clarity"])
                m3.metric("Confidence", last["scores"]["confidence"])
                m4.metric("Overall", last["scores"]["overall"])
                if last["scores"].get("flags"):
                    st.warning("⚠ Filler/hedge words detected: " + ", ".join(last["scores"]["flags"]))
                if last["scores"].get("commentary"):
                    st.caption("LLM commentary:")
                    st.write(last["scores"]["commentary"])
                if last["scores"].get("tone"):
                    st.caption("Tone:")
                    st.write(last["scores"]["tone"])

            with col3:
                st.subheader("Feedback")
                st.write(last["feedback"])

    # Finish control
    with next_col:
        if len(st.session_state.state["asked"]) >= max_q and st.button("Finish Session", use_container_width=True):
            st.session_state.state["done"] = True

# Final report + memory update
if st.session_state.state["done"]:
    transcript = st.session_state.state["transcript"]
    summary, md = build_report(transcript)

    # Update memory
    filler_total = sum(len(t["scores"].get("flags", [])) for t in transcript)
    prev_sessions = memory.get("sessions", 0)
    prev_avg = memory.get("avg_overall")
    this_overall = summary.get("Overall", 0)
    if prev_sessions and prev_avg is not None:
        new_avg = round((prev_avg * prev_sessions + this_overall) / (prev_sessions + 1), 2)
    else:
        new_avg = this_overall
    memory.update({
        "sessions": prev_sessions + 1,
        "avg_overall": new_avg,
        "filler_total": memory.get("filler_total", 0) + filler_total
    })
    save_memory(memory)

    with col3:
        st.subheader("Session Report")
        st.markdown(md)
        fname = "interviewsense_report.json"
        st.download_button(
            "Download JSON Report",
            data=json.dumps({"summary": summary, "transcript": transcript}, indent=2),
            file_name=fname
        )

    with col2:
        st.subheader("Scores (All Questions)")
        for i, t in enumerate(transcript, 1):
            st.write(f"*Q{i}.* {t['question']}")
            st.write(
                f"Score: {t['scores']['overall']}/10  —  "
                f"Technical {t['scores']['technical']}, "
                f"Clarity {t['scores']['clarity']}, "
                f"Confidence {t['scores']['confidence']}"
            )
            if t["scores"].get("flags"):
                st.caption("Flags: " + ", ".join(t["scores"]["flags"]))
            st.divider()

    with col1:
        st.subheader("Restart")
        if st.button("Start New Interview"):
            st.session_state.pop("state", None)
            st.rerun()

# Footer notes
if LLM is None:
    st.sidebar.info("ℹ GPT features are disabled (no OPENAI_API_KEY detected). Using curated question bank + heuristic feedback.")
if sr is None:
    st.sidebar.info("ℹ Voice input disabled (SpeechRecognition/PyAudio not installed). Use the text box to answer.")
if TextBlob is None:
    st.sidebar.info("ℹ Sentiment/tone analysis disabled (TextBlob not installed).")