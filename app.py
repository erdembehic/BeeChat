"""BeeChat — İTÜ Mevzuat Asistanı (Chainlit)."""
import chainlit as cl
from chainlit.input_widget import Select

from rag.pipeline  import RAGPipeline
from rag.situation import DurumAnalizoru
from rag           import llm

_rag:    RAGPipeline    | None = None
_analiz: DurumAnalizoru | None = None


def get_rag() -> RAGPipeline:
    global _rag
    if _rag is None:
        _rag = RAGPipeline()
    return _rag


def get_analiz() -> DurumAnalizoru:
    global _analiz
    if _analiz is None:
        _analiz = DurumAnalizoru()
    return _analiz


# ── Oturum başlangıcı ─────────────────────────────────────────────────────── #

@cl.on_chat_start
async def start():
    await cl.ChatSettings(
        [
            Select(
                id="mod",
                label="Mod",
                values=["Mevzuat Arama", "Durum Analizi"],
                initial_value="Mevzuat Arama",
            )
        ]
    ).send()

    await cl.Message(
        content=(
            "## İTÜ Mevzuat Asistanı\n\n"
            "**Mevzuat Arama** modunda yönetmelik ve kanunlar hakkında soru sorabilirsiniz.\n\n"
            "**Durum Analizi** modunda kendi durumunuzu yazın; "
            "ilgili maddeler ve önerilen adımlar size özel hazırlanır.\n\n"
            "---\n"
            "**Örnek sorular:**\n"
            "- Doktora azami öğrenim süresi nedir?\n"
            "- Sınav sonucuna nasıl itiraz ederim?\n"
            "- *Durum analizi:* Lisans 4. yılım, GNO'm 1.78, bu dönem 2 ders daha geçemedim."
        ),
        author="BeeChat",
    ).send()


@cl.on_settings_update
async def settings_update(settings: dict):
    cl.user_session.set("mod", settings["mod"])


# ── Mesaj işleme ──────────────────────────────────────────────────────────── #

@cl.on_message
async def main(message: cl.Message):
    soru = message.content.strip()
    mod  = cl.user_session.get("mod", "Mevzuat Arama")

    if soru.lower().startswith("durum:"):
        soru = soru[6:].strip()
        mod  = "Durum Analizi"

    if mod == "Durum Analizi":
        await _handle_analiz(soru)
    else:
        await _handle_rag(soru)


# ── RAG sorgusu ───────────────────────────────────────────────────────────── #

async def _handle_rag(soru: str):
    rag = get_rag()

    async with cl.Step(name="Retrieval", type="retrieval") as step:
        chunks = rag.retriever.retrieve(soru)
        step.output = f"{len(chunks)} madde bulundu"

    if not chunks:
        await cl.Message(content="İlgili mevzuat bulunamadı.", author="BeeChat").send()
        return

    context  = _build_context(chunks)
    user_msg = f"Kaynaklar:\n{context}\n\nSoru: {soru}"

    msg = cl.Message(content="", author="BeeChat")
    await msg.send()

    full: list[str] = []
    for token, kind in llm.stream(rag.cfg.system_prompt, user_msg,
                                  rag.cfg.claude_model, rag.cfg.max_tokens):
        if kind == "thinking":
            # thinking tokenları ayrı step'te göster
            pass   # biriktirilir, aşağıda thinking step açılır
        else:
            full.append(token)
            msg.content = "".join(full)
            await msg.update()

    msg.elements = _build_source_elements(chunks)
    await msg.update()


# ── Durum analizi ─────────────────────────────────────────────────────────── #

async def _handle_analiz(durum: str):
    analizor = get_analiz()

    # Adım 1: Fact extraction
    async with cl.Step(name="Bilgi çıkarma", type="tool") as step:
        faktler = analizor._fact_extraction(durum)
        step.output = analizor._faktler_str(faktler)

    # Adım 2: Query generation + retrieval
    async with cl.Step(name="Mevzuat arama", type="retrieval") as step:
        queries   = analizor._queries_from_facts(faktler)
        kaynaklar = analizor._multi_retrieve(queries)
        step.input  = "\n".join(f"• {q}" for q in queries)
        step.output = f"{len(kaynaklar)} ilgili madde bulundu"

    # Adım 3: Analiz (streaming) — thinking varsa ayrı step
    from rag.situation import _ANALYSIS_SYSTEM, _ANALYSIS_USER

    user_msg = _ANALYSIS_USER.format(
        ham_giris=durum,
        faktler=analizor._faktler_str(faktler),
        kaynaklar=analizor._build_source_text(kaynaklar),
    )

    # Thinking tokenlarını ve yanıt tokenlarını ayır
    thinking_parts: list[str] = []
    answer_parts:   list[str] = []

    msg = cl.Message(content="", author="BeeChat")
    await msg.send()

    thinking_step: cl.Step | None = None

    for token, kind in llm.stream(_ANALYSIS_SYSTEM, user_msg,
                                  analizor.cfg.claude_model, analizor.cfg.max_tokens):
        if kind == "thinking":
            thinking_parts.append(token)
            if thinking_step is None:
                thinking_step = cl.Step(name="Düşünce süreci", type="llm")
                await thinking_step.__aenter__()
        else:
            # thinking bitti, step'i kapat
            if thinking_step is not None:
                thinking_step.output = "".join(thinking_parts)
                await thinking_step.__aexit__(None, None, None)
                thinking_step = None

            answer_parts.append(token)
            msg.content = "".join(answer_parts)
            await msg.update()

    if thinking_step is not None:
        thinking_step.output = "".join(thinking_parts)
        await thinking_step.__aexit__(None, None, None)

    msg.elements = _build_source_elements(kaynaklar)
    await msg.update()


# ── Yardımcılar ───────────────────────────────────────────────────────────── #

def _build_context(chunks) -> str:
    parts = []
    for rc in chunks:
        m   = rc.chunk.metadata
        tag = " (çapraz referans)" if rc.source == "ref_expansion" else ""
        parts.append(f"[{m['baslik']} — Madde {m['maddeNo']}]{tag}\n{rc.chunk.content}")
    return "\n\n---\n\n".join(parts)


def _build_source_elements(chunks) -> list[cl.Text]:
    seen, elements = set(), []
    for rc in chunks:
        m   = rc.chunk.metadata
        lbl = f"{m['baslik']} — Madde {m['maddeNo']}"
        if lbl in seen:
            continue
        seen.add(lbl)
        tag = " *(çapraz ref)*" if rc.source == "ref_expansion" else ""
        elements.append(cl.Text(
            name=lbl,
            content=f"**{lbl}**{tag}\n\n{rc.chunk.content}",
            display="side",
        ))
    return elements
