from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, Optional

from aiogram import Bot, Dispatcher, F, Router
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton

from bot.config import settings
from bot.rag_client import RAGClient

router = Router()
user_docs: Dict[int, int] = {}
user_doc_history: Dict[int, set[int]] = {}
rag_client = RAGClient(settings.rag_api_base, timeout=settings.request_timeout)


def _keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="/mydocs"), KeyboardButton(text="/reset")],
        ],
        resize_keyboard=True,
    )


async def _progress_notifier(message: Message) -> None:
    """Периодически обновляет сообщение, пока идёт индексация."""
    start = asyncio.get_event_loop().time()
    tick = 0
    while True:
        await asyncio.sleep(8)
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
        "• Команда /reset выключает фильтр по документу."
    )
    await message.answer(text, reply_markup=_keyboard())


@router.message(Command("reset"))
async def cmd_reset(message: Message) -> None:
    user_docs.pop(_author_id(message), None)
    await message.answer("🔄 Фильтр по документу отключён. Буду искать по всем материалам.", reply_markup=_keyboard())


@router.message(Command("use"))
async def cmd_use(message: Message) -> None:
    args = message.get_args().strip()
    if not args:
        await message.answer("Укажи doc_id после команды: /use 123456. /mydocs — список своих документов.")
        return
    try:
        doc_id = int(args)
    except ValueError:
        await message.answer("doc_id должен быть числом. Пример: /use 123456")
        return
    user_docs[_author_id(message)] = doc_id
    await message.answer(f"🎯 Теперь ищу только внутри doc_id={doc_id}. Вернуть общий поиск — /reset.")


@router.message(Command("mydocs"))
async def cmd_mydocs(message: Message) -> None:
    doc_ids = sorted(user_doc_history.get(_author_id(message), []))
    if not doc_ids:
        await message.answer("У тебя ещё нет загруженных документов. Пришли файл, чтобы проиндексировать.")
        return
    doc_list = "\n".join(str(d) for d in doc_ids)
    await message.answer(f"📑 Твои doc_id:\n{doc_list}\n\nСменить активный: /use <doc_id>.\nСбросить: /reset.")


@router.message(F.document)
async def handle_document(message: Message) -> None:
    doc = message.document
    if not doc:
        return

    status = await message.answer("📥 Скачиваю и индексирую документ, подожди...")
    progress_task = asyncio.create_task(_progress_notifier(status))
    try:
        suffix = Path(doc.file_name or "uploaded").suffix or ".tmp"
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = Path(tmp.name)
        telegram_file = await message.bot.get_file(doc.file_id)
        await message.bot.download_file(telegram_file.file_path, destination=str(tmp_path))
        result = await rag_client.index_document(
            tmp_path,
            section=settings.default_section,
            owner_id=_author_id(message),
        )
    except Exception as exc:
        logging.exception("Failed to index document")
        progress_task.cancel()
        await status.edit_text(f"❌ Не удалось проиндексировать файл: {exc}")
        return
    finally:
        if 'tmp_path' in locals():
            tmp_path.unlink(missing_ok=True)
        progress_task.cancel()

    user_docs[_author_id(message)] = result["doc_id"]
    user_doc_history.setdefault(_author_id(message), set()).add(result["doc_id"])
    await status.edit_text(
        "✅ Документ индексирован.\n"
        f"Чанков: {result['chunks_indexed']}\n"
        f"doc_id: `{result['doc_id']}`\n\n"
        "Теперь все вопросы будут искаться только в этом документе.\n"
        "Сменить активный: /use <doc_id>\n"
        "Список документов: /mydocs\n"
        "Сбросить на поиск по всем: /reset",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=_keyboard(),
    )


@router.message(F.text)
async def handle_question(message: Message) -> None:
    question = message.text.strip()
    if not question:
        return

    doc_filter: Optional[int] = user_docs.get(_author_id(message))
    try:
        resp = await rag_client.query(
            question=question,
            top_k=settings.default_top_k,
            doc_id=doc_filter,
            owner_id=_author_id(message),
        )
    except Exception as exc:
        logging.exception("Failed to query RAG")
        await message.answer(f"❌ Ошибка запроса к RAG API: {exc}")
        return

    answer = resp.get("answer") or "Контекст найден, но LLM не вернул ответ."
    hits = resp.get("hits") or []

    if doc_filter is None:
        prefix = "🤖 Ответ по всей базе:\n\n"
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
