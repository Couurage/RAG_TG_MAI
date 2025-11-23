from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, Optional, List, Any

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton

from bot.config import settings
from bot.rag_client import RAGClient

router = Router()

# активный doc_id на пользователя
user_docs: Dict[int, int] = {}

# история документов пользователя:
# user_id -> [{"doc_id": int, "name": str}, ...]
user_doc_history: Dict[int, List[Dict[str, Any]]] = {}

rag_client = RAGClient(settings.rag_api_base, timeout=settings.request_timeout)


def _keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="/mydocs"), KeyboardButton(text="/reset")],
        ],
        resize_keyboard=True,
    )


async def _progress_notifier(message: Message, stop_event: asyncio.Event) -> None:
    """Периодически обновляет сообщение, пока идёт индексация."""
    start = asyncio.get_event_loop().time()
    tick = 0
    while not stop_event.is_set():
        await asyncio.sleep(8)
        if stop_event.is_set():
            break
        tick += 1
        elapsed = int(asyncio.get_event_loop().time() - start)
        try:
            await message.edit_text(f"⏳ Индексация идёт... {elapsed} c (шаг {tick})")
        except Exception:
            # если сообщение удалили или бот не может обновить — выходим
            return


def _author_id(message: Message) -> int:
    if message.from_user:
        return message.from_user.id
    return message.chat.id


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    text = (
        "👋 Привет! Я RAG-бот.\n\n"
        "• Отправь текстовое сообщение — я спрошу RAG и верну ответ.\n"
        "• Пришли файл (PDF/DOCX/MD) — я его проиндексирую и буду отвечать только по нему.\n"
        "• /mydocs — список твоих файлов.\n"
        "• /use <номер> — выбрать файл из списка.\n"
        "• /reset — выключает фильтр по документу."
    )
    await message.answer(text, reply_markup=_keyboard())


@router.message(Command("reset"))
async def cmd_reset(message: Message) -> None:
    user_docs.pop(_author_id(message), None)
    await message.answer(
        "🔄 Фильтр по документу отключён. Буду искать по всем материалам.",
        reply_markup=_keyboard(),
    )


@router.message(Command("use"))
async def cmd_use(message: Message) -> None:
    """Выбор активного документа по номеру из /mydocs."""
    author_id = _author_id(message)
    history = user_doc_history.get(author_id, [])

    # аккуратно вытащим аргументы из текстовой команды
    full_text = message.text or ""
    parts = full_text.split(maxsplit=1)
    args = parts[1].strip() if len(parts) > 1 else ""

    if not args:
        await message.answer(
            "Используй: /use <номер документа из /mydocs>\n"
            "Например: /use 1",
            reply_markup=_keyboard(),
        )
        return

    try:
        if not args.isdigit():
            await message.answer(
                "Номер должен быть целым числом. Пример: /use 1",
                reply_markup=_keyboard(),
            )
            return

        idx = int(args)
        if idx < 1 or idx > len(history):
            await message.answer(
                "Нет документа с таким номером. Посмотри список: /mydocs",
                reply_markup=_keyboard(),
            )
            return

        entry = history[idx - 1]
        user_docs[author_id] = entry["doc_id"]

        await message.answer(
            f"🎯 Активный документ: {entry['name']}\n"
            "Теперь все ответы будут искаться только в этом документе.\n"
            "Сбросить фильтр: /reset",
            reply_markup=_keyboard(),
        )
    except Exception as exc:
        logging.exception("Failed to handle /use command")
        await message.answer(
            f"⚠️ Не получилось переключить документ: {exc}",
            reply_markup=_keyboard(),
        )


@router.message(Command("mydocs"))
async def cmd_mydocs(message: Message) -> None:
    author_id = _author_id(message)
    history = user_doc_history.get(author_id, [])
    active_doc = user_docs.get(author_id)

    if not history:
        await message.answer(
            "У тебя ещё нет загруженных документов. Пришли файл, чтобы проиндексировать.",
            reply_markup=_keyboard(),
        )
        return

    lines: List[str] = []
    for i, entry in enumerate(history, 1):
        mark = "⭐️" if entry["doc_id"] == active_doc else "  "
        lines.append(f"{mark} {i}. {entry['name']}")

    await message.answer(
        "📑 Твои документы:\n"
        + "\n".join(lines)
        + "\n\nВыбрать документ: /use <номер>\nСбросить фильтр: /reset",
        reply_markup=_keyboard(),
    )


@router.message(F.document)
async def handle_document(message: Message) -> None:
    doc = message.document
    if not doc:
        return

    status = await message.answer("📥 Скачиваю и индексирую документ, подожди...")
    stop_event = asyncio.Event()
    progress_task = asyncio.create_task(_progress_notifier(status, stop_event))

    result = None
    tmp_path: Path | None = None

    try:
        suffix = Path(doc.file_name or "uploaded").suffix or ".tmp"
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = Path(tmp.name)

        telegram_file = await message.bot.get_file(doc.file_id)
        await message.bot.download_file(
            telegram_file.file_path,
            destination=str(tmp_path),
        )

        result = await rag_client.index_document(
            tmp_path,
            section=settings.default_section,
            owner_id=_author_id(message),
        )
    except Exception as exc:
        logging.exception("Failed to index document")
        progress_task.cancel()
        try:
            await status.edit_text(f"❌ Не удалось проиндексировать файл: {exc}")
        except Exception:
            await message.answer(f"❌ Не удалось проиндексировать файл: {exc}")
        return
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        stop_event.set()
        if not progress_task.done():
            progress_task.cancel()
        try:
            await progress_task
        except asyncio.CancelledError:
            pass

    # успешная индексация
    author_id = _author_id(message)
    file_name = doc.file_name or f"doc_{result['doc_id']}"

    user_docs[author_id] = result["doc_id"]

    history = user_doc_history.setdefault(author_id, [])
    if not any(entry["doc_id"] == result["doc_id"] for entry in history):
        history.append({"doc_id": result["doc_id"], "name": file_name})

    # короткий финал для статусного сообщения
    try:
        await status.edit_text("✅ Индексация завершена.")
    except Exception:
        logging.exception("Failed to edit status message after indexing")

    # отдельное итоговое сообщение
    await message.answer(
        "✅ Документ индексирован!\n\n"
        f"📄 Файл: {file_name}\n"
        f"📦 Чанков: {result['chunks_indexed']}\n"
        f"🆔 doc_id: {result['doc_id']}\n\n"
        "Теперь все вопросы будут искаться только в этом документе.\n"
        "/mydocs — список всех твоих документов\n"
        "/use <номер> — выбрать другой документ\n"
        "/reset — выключить фильтр",
        reply_markup=_keyboard(),
    )


@router.message(F.text)
async def handle_question(message: Message) -> None:
    question = message.text.strip()
    if not question:
        return

    author_id = _author_id(message)
    doc_filter: Optional[int] = user_docs.get(author_id)

    try:
        resp = await rag_client.query(
            question=question,
            top_k=settings.default_top_k,
            doc_id=doc_filter,
            owner_id=author_id,
        )
    except Exception as exc:
        logging.exception("Failed to query RAG")
        await message.answer(f"❌ Ошибка запроса к RAG API: {exc}")
        return

    answer = resp.get("answer") or "Контекст найден, но LLM не вернул ответ."

    # найдём имя файла по активному doc_id, если оно есть
    name: Optional[str] = None
    if doc_filter is not None:
        history = user_doc_history.get(author_id, [])
        for entry in history:
            if entry["doc_id"] == doc_filter:
                name = entry["name"]
                break

    if doc_filter is None:
        prefix = "🤖 Ответ по всей базе:\n\n"
    elif name:
        prefix = f"🤖 Ответ в границах файла «{name}»:\n\n"
    else:
        prefix = f"🤖 Ответ в границах doc_id={doc_filter}:\n\n"

    await message.answer(prefix + answer, reply_markup=_keyboard())


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    bot = Bot(token=settings.telegram_bot_token)
    dp = Dispatcher()
    dp.include_router(router)

    try:
        await dp.start_polling(bot)
    finally:
        await rag_client.close()


if __name__ == "__main__":
    asyncio.run(main())