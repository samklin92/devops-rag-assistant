import re


def chunk_document(
    document: dict[str, str],
    max_chunk_size: int = 1200,
    overlap: int = 150,
) -> list[dict[str, str]]:
    """Split a Markdown document into chunks while preserving section context."""

    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be greater than zero.")

    if overlap < 0 or overlap >= max_chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than max_chunk_size."
        )

    content = document["content"].strip()
    source = document["source"]

    if not content:
        return []

    headings = list(re.finditer(r"(?m)^#{1,6}\s+.+$", content))

    if not headings:
        sections = [("", content)]
    else:
        sections = []

        # Preserve any text appearing before the first heading.
        if content[:headings[0].start()].strip():
            sections.append(
                ("", content[:headings[0].start()].strip())
            )

        for index, heading in enumerate(headings):
            start = heading.start()
            end = (
                headings[index + 1].start()
                if index + 1 < len(headings)
                else len(content)
            )

            section_text = content[start:end].strip()

            if section_text:
                sections.append((heading.group().strip(), section_text))

    chunks = []

    for heading, section_text in sections:
        start = 0

        while start < len(section_text):
            end = min(start + max_chunk_size, len(section_text))

            # Prefer breaking at a paragraph boundary when possible.
            if end < len(section_text):
                boundary = section_text.rfind("\n\n", start, end)

                if boundary > start:
                    end = boundary

            chunk_text = section_text[start:end].strip()

            if chunk_text:
                chunks.append(
                    {
                        "source": source,
                        "section": heading,
                        "content": chunk_text,
                    }
                )

            if end >= len(section_text):
                break

            # Move forward while retaining the configured overlap.
            next_start = max(end - overlap, start + 1)
            start = next_start

    return chunks