"""Tiện ích dùng chung cho `critic` và `citation_checker`.

STUDENT-OWNED. Không có lớp nào ở đây — chỉ vài hàm nhỏ mà hai lớp bằng
chứng đều cần, đặt chung một chỗ để hai bên KHÔNG BAO GIỜ trả lời khác
nhau cho cùng một câu hỏi "câu này có phải trích dẫn của tài liệu kia
không". Ranh giới critic/citation_checker chỉ chặt chẽ khi cả hai đo bằng
đúng một cái thước.

Cái thước đó là thước của bộ chấm, chép lại chứ không phát minh:

  * `arena.scorer._norm`      -> NFC + casefold + gộp khoảng trắng
  * `arena.scorer._norm_lines`-> chuẩn hoá THEO TỪNG DÒNG
  * `arena.scorer._supports`  -> trích dẫn phải nằm gọn trong MỘT dòng,
                                 và dài ít nhất `MIN_SUPPORT_CHARS`

Chép lại thay vì import `arena.scorer`: bộ chấm là module đóng băng và
các hàm đó là nội bộ (`_` ở đầu tên). Một lớp phụ thuộc vào nội bộ của bộ
chấm sẽ vỡ ở vòng tính điểm nếu bản đóng băng ở đó khác một chữ.

KHÔNG hàm nào ở đây SỬA chữ của claim. Chuẩn hoá chỉ dùng để SO SÁNH —
xem README §8.2: đổi citation hoặc bỏ claim thì được, đổi chữ thì mất
provenance.
"""

from __future__ import annotations

import re
import unicodedata

#: Giống `arena.scorer.MIN_SUPPORT_CHARS`: ngắn hơn thế thì một chuỗi
#: khớp được với nửa kho tài liệu, nên nó không "đỡ" cho cái gì cả.
MIN_SUPPORT_CHARS = 12

_WS_RE = re.compile(r"\s+")

#: Khoá cache trong `ctx.state` — chuẩn hoá 120 tài liệu tốn vài mili
#: giây, và mỗi claim lại hỏi một lần.
_INDEX_KEY = "_evidence_index"
_OBSERVED_KEY = "_evidence_observed"


def norm(text) -> str:
    """NFC + casefold + gộp khoảng trắng — dạng mà mọi so sánh chạy trên."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    return _WS_RE.sub(" ", unicodedata.normalize("NFC", text).casefold()).strip()


def norm_lines(text) -> tuple:
    """Một chuỗi đã chuẩn hoá cho MỖI DÒNG của tài liệu.

    Tách dòng ở đây chính là chỗ khác biệt giữa "câu này có trong tài
    liệu" và "câu này là một trích dẫn": bộ chấm không nhận một câu vắt
    qua hai dòng.
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    return tuple(line for line in (norm(raw) for raw in text.splitlines()) if line)


def quotes_a_line(lines, normalised_claim: str) -> bool:
    """Câu (đã chuẩn hoá) có nằm gọn trong một dòng nào của tài liệu không?"""
    if len(normalised_claim) < MIN_SUPPORT_CHARS:
        return False
    return any(normalised_claim in line for line in lines)


def claim_text(claim) -> str:
    if not isinstance(claim, dict):
        return ""
    value = claim.get("text")
    return value if isinstance(value, str) else ""


def claim_doc_id(claim) -> str:
    if not isinstance(claim, dict):
        return ""
    value = claim.get("doc_id")
    return value.strip() if isinstance(value, str) else ""


def citations_for(claims) -> list:
    """`citations` khớp với danh sách claim còn lại, đã sắp xếp."""
    seen = {claim_doc_id(claim) for claim in claims}
    return sorted(doc_id for doc_id in seen if doc_id)


def observed(ctx) -> str:
    """Toàn bộ quan sát agent đã thấy, đã chuẩn hoá (có cache)."""
    cached = ctx.state.get(_OBSERVED_KEY)
    text = ctx.observed_text
    if isinstance(cached, tuple) and cached[0] == len(text):
        return cached[1]
    normalised = norm(text)
    ctx.state[_OBSERVED_KEY] = (len(text), normalised)
    return normalised


class Index:
    """Kho tài liệu đã chuẩn hoá + biết tài liệu nào lượt chạy đã đọc.

    `retrieved` là tài liệu mà lượt chạy CHỨNG MINH được là đã nhìn thấy,
    chia làm hai hạng — và thứ tự hai hạng đó là có chủ ý:

      1. `whole`  — toàn văn body có mặt nguyên vẹn trong quan sát, tức
                    một lần `fetch_doc` sạch. Đây là nguồn đáng tin nhất.
      2. `named`  — doc_id có xuất hiện trong quan sát (kết quả search,
                    hoặc một lần fetch bị cắt / bị `injection_guard` làm
                    sạch). Bộ chấm coi những tài liệu này là ĐÃ TRUY XUẤT
                    (nó phát lại truy vấn search qua đúng hàm BM25 cũ),
                    nên gắn citation vào đây KHÔNG bị chấm `UNRETRIEVED`.

    Không bao giờ vượt ra ngoài hai hạng đó: trích một tài liệu lượt chạy
    chưa từng chạm là `UNRETRIEVED`, phạt 0.75.
    """

    def __init__(self, ctx) -> None:
        corpus = ctx.corpus
        docs = list(getattr(corpus, "docs", None) or [])
        seen = observed(ctx)
        self.lines: dict = {}
        self.whole: list = []
        self.named: list = []
        for doc in docs:
            doc_id = getattr(doc, "doc_id", "")
            body = getattr(doc, "body", "")
            if not isinstance(doc_id, str) or not doc_id:
                continue
            self.lines[doc_id] = norm_lines(body)
            if norm(body) and norm(body) in seen:
                self.whole.append(doc_id)
            elif norm(doc_id) in seen:
                self.named.append(doc_id)

    def supports(self, doc_id: str, normalised_claim: str) -> bool:
        return quotes_a_line(self.lines.get(doc_id, ()), normalised_claim)

    def sources(self, normalised_claim: str) -> list:
        """Tài liệu ĐÃ ĐỌC có chứa câu này nguyên văn trên một dòng.

        Hạng `whole` trước, `named` sau, trong mỗi hạng giữ nguyên thứ tự
        của corpus — tất định, không phụ thuộc `set` hay `hash`.
        """
        if len(normalised_claim) < MIN_SUPPORT_CHARS:
            return []
        return [
            doc_id
            for doc_id in self.whole + self.named
            if self.supports(doc_id, normalised_claim)
        ]


def index(ctx) -> Index:
    """`Index` của lượt chạy này, dựng một lần rồi dùng lại."""
    cached = ctx.state.get(_INDEX_KEY)
    if isinstance(cached, tuple) and cached[0] == len(ctx.observed_text):
        return cached[1]
    built = Index(ctx)
    ctx.state[_INDEX_KEY] = (len(ctx.observed_text), built)
    return built
