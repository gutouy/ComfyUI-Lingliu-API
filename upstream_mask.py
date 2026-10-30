"""Hide upstream supplier names in text shown to users.

The platform may echo supplier names or hosts in error messages.  Users only
need to know the request failed upstream, so those words are replaced before
any response or error reaches the node outputs.  The word list is stored as
salted digests: the plugin needs to recognise the words, not to spell them.
"""

import hashlib
import re

SALT = "lingliu-mask-v1:"
UPSTREAM = "上游"
URL_LABEL = "<上游地址>"
JL = "JL"

DIGESTS = {
    "4931b6e1f29997d2026d7328": UPSTREAM,
    "939c943962fa615c3a53614f": UPSTREAM,
    "a0474c9acdc7ee0501b22158": UPSTREAM,
    "4b3296cf72adf35114bac908": JL,
    "bef2e8110b42b9e1cad48e18": UPSTREAM,
    "79ebbfa97eb90e340bd1efc8": UPSTREAM,
    "ba5c7c149ac56aab84e94734": UPSTREAM,
    "bdf753a4dbfa5294487cff4c": UPSTREAM,
    "0d67d2f6e3090273e0858d35": UPSTREAM,
    "51856f6ab3d5244e450e02f9": UPSTREAM,
    "35fce275f50065aea486bab0": UPSTREAM,
    "f106b5be21f817694d04d1c8": UPSTREAM,
    "61ec5f1ae0e2b8e6724cac26": UPSTREAM,
    "43d223bb14058ddcc9599610": UPSTREAM,
    "668ccaee253cda98fbd90e1e": UPSTREAM,
    "a2d4658f5189c3439f477a2b": UPSTREAM,
    "35e303aaab77aec4c17cc7de": UPSTREAM,
    "571ecaad29aad2d01c93c70a": UPSTREAM,
}
ASCII_LENGTHS = (4, 5, 6, 7, 9, 10, 11)
CJK_LENGTHS = (2, 3)

WORD = re.compile(r"[A-Za-z0-9]+")
CJK = re.compile(r"[一-鿿]{2,}")
# ASCII-only so a masked URL never swallows the Chinese text that follows it.
URL = re.compile(r"https?://([A-Za-z0-9.-]+)[!#-&(-;=?-~]*")


def _digest(word):
    return hashlib.sha256((SALT + word).encode("utf-8")).hexdigest()[:24]


def label_for(word):
    """Return the replacement label when ``word`` starts with an upstream name."""
    word = str(word or "").lower()
    for length in ASCII_LENGTHS:
        if length > len(word):
            break
        label = DIGESTS.get(_digest(word[:length]))
        if label:
            return label
    return None


def is_group(group, label):
    """True when a pricing group belongs to the upstream named by ``label``."""
    head = re.split(r"[-_]", str(group or "").strip().lower(), maxsplit=1)[0]
    return bool(head) and DIGESTS.get(_digest(head)) == label


def _mask_words(text):
    out = []
    pos = 0
    matches = list(WORD.finditer(text))
    index = 0
    while index < len(matches):
        match = matches[index]
        label = label_for(match.group())
        end = match.end()
        if not label and index + 1 < len(matches):
            following = matches[index + 1]
            if following.start() - end == 1 and text[end] in " -_":
                label = label_for(match.group() + following.group())
                if label:
                    end = following.end()
                    index += 1
        if label:
            if end < len(text) and text[end] == "_":
                end += 1
                label = "" if label == UPSTREAM else label
            out.append(text[pos:match.start()])
            out.append(label)
            pos = end
        index += 1
    out.append(text[pos:])
    return "".join(out)


def _mask_cjk(match):
    run = match.group()
    out = []
    index = 0
    while index < len(run):
        for length in CJK_LENGTHS:
            label = DIGESTS.get(_digest(run[index:index + length])) if index + length <= len(run) else None
            if label:
                # Adjacent names (company + product) collapse into one label.
                if not out or out[-1] != label:
                    out.append(label)
                index += length
                break
        else:
            out.append(run[index])
            index += 1
    return "".join(out)


def _mask_url(match):
    host = match.group(1)
    if any(label_for(part) for part in WORD.findall(host)):
        return URL_LABEL
    return match.group()


def mask_upstream(value):
    text = URL.sub(_mask_url, str(value))
    return CJK.sub(_mask_cjk, _mask_words(text))
