#!/usr/bin/env python3
"""
STORM Wiki Article Generator — uses local LLM + Qdrant second brain.

Usage:
    python3 generate.py "Topic Name"
    python3 generate.py "Topic Name" --output /path/to/output/dir
    python3 generate.py "Topic Name" --no-polish   # skip polishing step (faster)

Requires: knowledge-storm, qdrant-client
LLM: local Qwen at http://localhost:11435/v1
Retrieval: Qdrant second_brain via Ollama embeddings
"""

import argparse
import os
import sys
import shutil

# Add parent dir for imports
sys.path.insert(0, os.path.dirname(__file__))

from knowledge_storm import STORMWikiRunner, STORMWikiRunnerArguments, STORMWikiLMConfigs, OpenAIModel
from qdrant_rm import QdrantLocalRM


def _patch_openai_for_thinking_models():
    """Monkey-patch OpenAI client to pass enable_thinking=false for Qwen 3.5.
    Only affects requests made through this script — not server-wide."""
    import openai
    _orig_create = openai.resources.chat.completions.Completions.create

    def _patched_create(self, *args, **kwargs):
        if "extra_body" not in kwargs:
            kwargs["extra_body"] = {}
        kwargs["extra_body"]["chat_template_kwargs"] = {"enable_thinking": False}
        return _orig_create(self, *args, **kwargs)

    openai.resources.chat.completions.Completions.create = _patched_create

    # Also patch async version if used
    try:
        _orig_acreate = openai.resources.chat.completions.AsyncCompletions.create

        async def _patched_acreate(self, *args, **kwargs):
            if "extra_body" not in kwargs:
                kwargs["extra_body"] = {}
            kwargs["extra_body"]["chat_template_kwargs"] = {"enable_thinking": False}
            return await _orig_acreate(self, *args, **kwargs)

        openai.resources.chat.completions.AsyncCompletions.create = _patched_acreate
    except AttributeError:
        pass


_patch_openai_for_thinking_models()


def make_lm_configs(api_base: str, model: str) -> STORMWikiLMConfigs:
    """Configure all STORM LM slots to use local model."""
    lm_configs = STORMWikiLMConfigs()

    kwargs = {
        "api_key": "not-needed",
        "api_base": api_base,
        "temperature": 0.7,
        "top_p": 0.9,
    }

    # All roles use the same local model
    conv_lm = OpenAIModel(model=model, model_type="chat", max_tokens=1000, **kwargs)
    question_lm = OpenAIModel(model=model, model_type="chat", max_tokens=500, **kwargs)
    outline_lm = OpenAIModel(model=model, model_type="chat", max_tokens=1000, **kwargs)
    article_lm = OpenAIModel(model=model, model_type="chat", max_tokens=2000, **kwargs)
    polish_lm = OpenAIModel(model=model, model_type="chat", max_tokens=3000, **kwargs)

    lm_configs.set_conv_simulator_lm(conv_lm)
    lm_configs.set_question_asker_lm(question_lm)
    lm_configs.set_outline_gen_lm(outline_lm)
    lm_configs.set_article_gen_lm(article_lm)
    lm_configs.set_article_polish_lm(polish_lm)

    return lm_configs


def generate_article(
    topic: str,
    output_dir: str = None,
    api_base: str = "http://localhost:11435/v1/",
    model: str = "qwen3.5-122b-a10b",
    do_polish: bool = True,
):
    """Generate a wiki article on the given topic."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "wiki")

    os.makedirs(output_dir, exist_ok=True)

    # Working directory for STORM intermediate files
    work_dir = os.path.join(output_dir, ".storm_work")
    os.makedirs(work_dir, exist_ok=True)

    print(f"[STORM] Generating article: '{topic}'")
    print(f"[STORM] LLM: {api_base} / {model}")
    print(f"[STORM] Output: {output_dir}")

    # Configure LMs
    lm_configs = make_lm_configs(api_base, model)

    # Configure retrieval
    rm = QdrantLocalRM(
        collection_name="second_brain",
        qdrant_url="http://localhost:6333",
        ollama_url="http://localhost:11434",
        embed_model="nomic-embed-text",
        k=5,
    )

    # Runner arguments
    args = STORMWikiRunnerArguments(
        output_dir=work_dir,
        max_conv_turn=3,       # keep it short for local LLM
        max_perspective=2,     # fewer perspectives = faster
        search_top_k=5,
    )

    # Create and run
    runner = STORMWikiRunner(args=args, lm_configs=lm_configs, rm=rm)

    runner.run(
        topic=topic,
        do_research=True,
        do_generate_outline=True,
        do_generate_article=True,
        do_polish_article=do_polish,
    )

    runner.post_run()

    # Find the generated article and copy to wiki/
    safe_topic = topic.replace("/", "_").replace(" ", "_")
    article_path = None

    # STORM writes to work_dir/topic/...
    for root, dirs, files in os.walk(work_dir):
        for f in files:
            if "storm_gen_article" in f.lower():
                article_path = os.path.join(root, f)
                break
            # Also check for polished version
            if ("polished" in f.lower()) and (f.endswith(".md") or f.endswith(".txt")):
                article_path = os.path.join(root, f)
                break
        if article_path:
            break

    # If no specific file found, look for any .md in the topic folder
    if not article_path:
        topic_dir = os.path.join(work_dir, topic)
        if os.path.isdir(topic_dir):
            for f in os.listdir(topic_dir):
                if f.endswith(".md"):
                    article_path = os.path.join(topic_dir, f)
                    break

    if article_path and os.path.exists(article_path):
        dest = os.path.join(output_dir, f"{safe_topic}.md")
        shutil.copy2(article_path, dest)
        print(f"[STORM] Article saved to: {dest}")
        with open(dest) as f:
            content = f.read()
        print(f"[STORM] Article length: {len(content)} chars, {len(content.split())} words")
        return dest
    else:
        # List what was actually created
        print(f"[STORM] Warning: Could not find generated article. Contents of {work_dir}:")
        for root, dirs, files in os.walk(work_dir):
            for f in files:
                fp = os.path.join(root, f)
                print(f"  {fp} ({os.path.getsize(fp)} bytes)")

        # Try to find ANY output
        for root, dirs, files in os.walk(work_dir):
            for f in files:
                if f.endswith(".txt") or f.endswith(".md"):
                    src = os.path.join(root, f)
                    dest = os.path.join(output_dir, f"{safe_topic}.md")
                    shutil.copy2(src, dest)
                    print(f"[STORM] Copied {src} -> {dest}")
                    return dest

        print("[STORM] No article output found.")
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate wiki article using STORM + local LLM")
    parser.add_argument("topic", help="Topic to generate article about")
    parser.add_argument("--output", "-o", default=None, help="Output directory")
    parser.add_argument("--api-base", default="http://localhost:11435/v1/", help="LLM API base URL")
    parser.add_argument("--model", default="qwen3.5-122b-a10b", help="Model name")
    parser.add_argument("--no-polish", action="store_true", help="Skip polishing step")

    args = parser.parse_args()
    result = generate_article(
        topic=args.topic,
        output_dir=args.output,
        api_base=args.api_base,
        model=args.model,
        do_polish=not args.no_polish,
    )

    if result:
        print(f"\n✅ Done! Article at: {result}")
    else:
        print("\n❌ Generation failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
