from typing import List


def chunk_sentences(nlp, text):
    doc = nlp(text)
    sentence_chunks = [sent.text.strip() for sent in doc.sents]
    return sentence_chunks


@transformer
def transform(file_paths: List[str], *args, **kwargs):
    """
    Chunks for 1 document
    [
        "---\ntitle: \"Core Design Principles\"\nsidebarTitle: \"Design\"\nicon: \"pen-nib\"\ndescription:\n \"Every user experience and technical design decision adheres to these principles.",
        "\"\n\"og:image\": \"https://user-images.githubusercontent.com/78053898/198752891-1e823231-f5eb-48ea-8a6d-91e60ec368c9.svg\"\n---\n\n<Frame>\n <img\n alt=\"Core design principles\"\n src=\"https://user-images.githubusercontent.com/78053898/198752891-1e823231-f5eb-48ea-8a6d-91e60ec368c9.svg\"\n />\n</Frame>\n\n## 💻",
        "Easy developer experience\n\nOpen-source engine that comes with a custom notebook UI for building data\npipelines.",
        "- Mage comes with a specialized notebook UI for building data pipelines.",
        "- Use Python and SQL (more languages coming soon) together in the same pipeline\n for ultimate flexibility.",
        "- Set up locally and get started developing with a single command.",
        "- Deploying to production is fast using native integrations with major cloud\n providers.",
        "## 🚢 Engineering best practices built-in\n\nBuild and deploy data pipelines using modular code.",
        "No more writing throwaway\ncode or trying to turn notebooks into scripts.",
        "- Writing reusable code is easy because every block in your data pipeline is a\n standalone file.",
        "- Data validation is written into each block and tested every time a block is\n run.\n\n-",
        "Operationalizing your data pipelines is easy with built-in observability, data\n quality monitoring, and lineage.",
        "- Each block of code has a single responsibility: load data from a source,\n transform data, or export data anywhere.",
        "## 💳 Data is a first class citizen\n\nDesigned from the ground up specifically for running data-intensive workflows.\n\n-",
        "Every block run produces a data product (e.g. dataset, unstructured data,\n etc.)\n\n-",
        "Every data product can be automatically partitioned.",
        "- Each pipeline and data product can be versioned.",
        "- Backfilling data products is a core function and operation.",
        "## 🪐 Scaling is made simple\n\nAnalyze and process large data quickly for rapid iteration.",
        "- Transform very large datasets through a native integration with Spark.",
        "- Handle data intensive transformations with built-in distributed computing\n (e.g. Dask, Ray) \\[coming soon\\].\n\n- Run thousands of pipelines simultaneously and manage transparently through a\n collaborative UI.\n\n- Execute SQL queries in your data warehouse to process heavy workloads."
    ]
    """
    nlp = list(kwargs.get('factory_items_mapping').values())[0][0]

    processed_texts = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            text = f.read()
            chunks = chunk_sentences(nlp, text)
            processed_texts.append(chunks)

    return [processed_texts]