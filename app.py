"""BeeChat — İTÜ Mevzuat Asistanı (Chainlit)."""
import chainlit as cl
from chainlit.input_widget import Select

from rag.pipeline  import RAGPipeline
from rag.situation import DurumAnalizoru

# Singleton'lar — uygulama başladığında bir kez yükle
_rag:    RAGPipeline   | None = None
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


# ── Oturum başlangıcı ─────────────────────────────────────────────────── #

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


# ── Mesaj işleme ──────────────────────────────────────────────────────── #

@cl.on_message
async def main(message: cl.Message):
    soru = message.content.strip()
    mod  = cl.user_session.get("mod", "Mevzuat Arama")

    # "durum:" prefix de analiz modunu tetikler
    if soru.lower().startswith("durum:"):
        soru = soru[6:].strip()
        mod  = "Durum Analizi"

    if mod == "Durum Analizi":
        await _handle_analiz(soru)
    else:
        await _handle_rag(soru)


# ── RAG sorgusu ───────────────────────────────────────────────────────── #

async def _handle_rag(soru: str):
    msg = cl.Message(content="", author="BeeChat")
    await msg.send()

    rag    = get_rag()
    chunks = rag.retriever.retrieve(soru)

    if not chunks:
        msg.content = "İlgili mevzuat bulunamadı."
        await msg.update()
        return

    # Kaynakları Chainlit element olarak hazırla
    source_elements = _build_source_elements(chunks)

    # Streaming yanıt
    context  = _build_context(chunks)
    user_msg = f"Kaynaklar:\n{context}\n\nSoru: {soru}"

    from rag import llm
    full = []
    for text in llm.stream(rag.cfg.system_prompt, user_msg,
                            rag.cfg.claude_model, rag.cfg.max_tokens):
        full.append(text)
        msg.content = "".join(full)
        await msg.update()

    msg.content  = "".join(full)
    msg.elements = source_elements
    await msg.update()


# ── Durum analizi ─────────────────────────────────────────────────────── #

async def _handle_analiz(durum: str):
    # Adım adım ilerleme göster
    step_msg = cl.Message(content="⏳ Durum analiz ediliyor…", author="BeeChat")
    await step_msg.send()

    analizor = get_analiz()

    # Fact extraction
    step_msg.content = "🔍 Bilgiler çıkarılıyor…"
    await step_msg.update()
    faktler = analizor._fact_extraction(durum)

    # Retrieval
    step_msg.content = "📚 İlgili mevzuat aranıyor…"
    await step_msg.update()
    queries  = analizor._queries_from_facts(faktler)
    kaynaklar = analizor._multi_retrieve(queries)

    # Analiz (streaming)
    step_msg.content = "✍️ Analiz hazırlanıyor…"
    await step_msg.update()

    from rag.situation import _ANALYSIS_SYSTEM, _ANALYSIS_USER
    from rag import llm

    user_msg = _ANALYSIS_USER.format(
        ham_giris=durum,
        faktler=analizor._faktler_str(faktler),
        kaynaklar=analizor._build_source_text(kaynaklar),
    )

    await step_msg.remove()
    msg  = cl.Message(content="", author="BeeChat")
    await msg.send()
    full = []

    for text in llm.stream(_ANALYSIS_SYSTEM, user_msg,
                            analizor.cfg.claude_model, analizor.cfg.max_tokens):
        full.append(text)
        msg.content = "".join(full)
        await msg.update()

    msg.content  = "".join(full)
    msg.elements = _build_source_elements(kaynaklar)
    await msg.update()


# ── Yardımcılar ───────────────────────────────────────────────────────── #

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
