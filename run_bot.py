#!/usr/bin/env python3
"""
Единый Telegram бот с выбором модели: ChatGPT или GigaChat.

Пользователь переключается между моделями командой /mode.
Каждый пользователь хранит свой выбор — разные люди могут
одновременно общаться с разными моделями.

Запуск из корня проекта:
    python run_bot.py
"""

import sys
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ReplyKeyboardMarkup,
    KeyboardButton,
    BotCommand,
)
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes,
)

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# DatabaseLogger — автономный модуль без конфликтов, импорт безопасен
sys.path.insert(0, str(ROOT_DIR / 'assistant_api'))
from db_logger import DatabaseLogger
sys.path.pop(0)


# ─── Инициализация пайплайнов с изоляцией импортов ──────────────────────
#
# assistant_api/ и assistant_giga/ содержат модули с одинаковыми именами
# (rag_pipeline.py, vector_store.py, cache.py), но разными реализациями.
# Изолируем импорты через временное переключение sys.path и cwd.

CONFLICT_MODULES = ['rag_pipeline', 'vector_store', 'cache', 'gigachat_client']


def _init_in_directory(directory, factory):
    """
    Выполняет factory() с sys.path и cwd, указывающими на directory.
    После инициализации очищает sys.modules от конфликтующих имён.
    """
    saved_cwd = os.getcwd()
    saved = {n: sys.modules.pop(n) for n in CONFLICT_MODULES if n in sys.modules}

    sys.path.insert(0, directory)
    os.chdir(directory)

    try:
        return factory()
    finally:
        os.chdir(saved_cwd)
        if directory in sys.path:
            sys.path.remove(directory)
        for n in CONFLICT_MODULES:
            sys.modules.pop(n, None)
        sys.modules.update(saved)


SHARED_DATA_PATH = str(ROOT_DIR / 'data')


def init_api_pipeline():
    """Инициализация RAG pipeline на OpenAI API (ChatGPT)."""
    api_dir = str(ROOT_DIR / 'assistant_api')
    cache_path = str(ROOT_DIR / 'assistant_api' / 'api_rag_cache.db')

    def factory():
        from rag_pipeline import RAGPipeline
        return RAGPipeline(
            collection_name="api_rag_collection",
            cache_db_path=cache_path,
            data_path=SHARED_DATA_PATH,
            model="gpt-4o-mini"
        )

    return _init_in_directory(api_dir, factory)


def init_giga_pipeline():
    """Инициализация RAG pipeline на GigaChat."""
    giga_dir = str(ROOT_DIR / 'assistant_giga')
    cache_path = str(ROOT_DIR / 'assistant_giga' / 'gigachat_rag_cache.db')

    def factory():
        from rag_pipeline import RAGPipeline
        return RAGPipeline(
            collection_name="gigachat_rag_collection",
            cache_db_path=cache_path,
            data_path=SHARED_DATA_PATH,
            model="GigaChat"
        )

    return _init_in_directory(giga_dir, factory)


# ─── Единый Telegram бот ────────────────────────────────────────────────

MODE_LABELS = {
    'chatgpt': 'ChatGPT (OpenAI)',
    'gigachat': 'GigaChat (Сбер)',
}

BTN_MODE = "Сменить модель"
BTN_STATS = "Статистика"
BTN_LOGS = "Мои логи"
BTN_HELP = "Справка"
BTN_REINDEX = "Переиндексация"

MENU_BUTTONS = [BTN_MODE, BTN_STATS, BTN_LOGS, BTN_HELP, BTN_REINDEX]

BOT_COMMANDS = [
    BotCommand("start", "Запустить бота"),
    BotCommand("mode", "Выбрать модель (ChatGPT / GigaChat)"),
    BotCommand("stats", "Статистика системы"),
    BotCommand("logs", "Экспорт логов в CSV"),
    BotCommand("reindex", "Переиндексировать базу знаний"),
    BotCommand("help", "Справка по командам"),
]


class UnifiedTelegramBot:
    """Telegram бот с переключением между ChatGPT и GigaChat."""

    def __init__(self, token: str, pipelines: dict, logger: DatabaseLogger):
        self.pipelines = pipelines
        self.logger = logger
        self.user_modes = {}
        self.default_mode = list(pipelines.keys())[0]

        self.reply_keyboard = ReplyKeyboardMarkup(
            [
                [KeyboardButton(BTN_MODE), KeyboardButton(BTN_STATS)],
                [KeyboardButton(BTN_LOGS), KeyboardButton(BTN_REINDEX)],
                [KeyboardButton(BTN_HELP)],
            ],
            resize_keyboard=True,
        )

        self.app = Application.builder().token(token).build()
        self.app.add_handler(CommandHandler("start", self.cmd_start))
        self.app.add_handler(CommandHandler("help", self.cmd_help))
        self.app.add_handler(CommandHandler("mode", self.cmd_mode))
        self.app.add_handler(CommandHandler("stats", self.cmd_stats))
        self.app.add_handler(CommandHandler("logs", self.cmd_logs))
        self.app.add_handler(CommandHandler("reindex", self.cmd_reindex))
        self.app.add_handler(CallbackQueryHandler(self.on_mode_callback, pattern=r"^mode_"))
        self.app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.on_message)
        )

        self.app.post_init = self._post_init

    async def _post_init(self, application):
        """Регистрирует список команд в меню Telegram."""
        await application.bot.set_my_commands(BOT_COMMANDS)

    def _get_user_mode(self, user_id: str) -> str:
        return self.user_modes.get(user_id, self.default_mode)

    # ── Команды ──

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.effective_user.id)
        current = self._get_user_mode(user_id)
        modes_list = "\n".join(f"  • {MODE_LABELS[k]}" for k in self.pipelines)

        await update.message.reply_text(
            f"Добро пожаловать в RAG-ассистент!\n\n"
            f"Доступные модели:\n{modes_list}\n\n"
            f"Текущая модель: {MODE_LABELS[current]}\n\n"
            f"Используйте кнопки меню внизу или просто напишите вопрос!",
            reply_markup=self.reply_keyboard,
        )

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "Как пользоваться ботом:\n\n"
            "Напишите вопрос — бот ответит на основе базы знаний.\n\n"
            "Кнопки меню:\n"
            f"  {BTN_MODE} — переключить ChatGPT / GigaChat\n"
            f"  {BTN_STATS} — статистика системы\n"
            f"  {BTN_LOGS} — экспорт ваших логов в CSV\n"
            f"  {BTN_REINDEX} — обновить базу знаний\n"
            f"  {BTN_HELP} — эта справка\n\n"
            "Также доступны /-команды:\n"
            "/mode  /stats  /logs  /reindex  /help",
            reply_markup=self.reply_keyboard,
        )

    async def cmd_mode(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.effective_user.id)
        current = self._get_user_mode(user_id)

        buttons = []
        for key in self.pipelines:
            label = MODE_LABELS[key]
            if key == current:
                label = f">> {label} <<"
            buttons.append([InlineKeyboardButton(label, callback_data=f"mode_{key}")])

        await update.message.reply_text(
            f"Текущая модель: {MODE_LABELS[current]}\n\nВыберите модель:",
            reply_markup=InlineKeyboardMarkup(buttons)
        )

    async def on_mode_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()

        mode_key = query.data.removeprefix("mode_")
        if mode_key not in self.pipelines:
            await query.edit_message_text("Эта модель недоступна.")
            return

        user_id = str(query.from_user.id)
        self.user_modes[user_id] = mode_key
        await query.edit_message_text(
            f"Модель переключена на: {MODE_LABELS[mode_key]}"
        )

    async def cmd_reindex(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "Переиндексация базы знаний...\n"
            "Это может занять несколько минут."
        )

        results = []
        for key, pipeline in self.pipelines.items():
            try:
                count = pipeline.reindex()
                results.append(f"{MODE_LABELS[key]}: {count} чанков")
            except Exception as e:
                results.append(f"{MODE_LABELS[key]}: ошибка — {e}")

        report = "\n".join(results)
        await update.message.reply_text(
            f"Переиндексация завершена!\n\n{report}"
        )

    async def cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            user_id = str(update.effective_user.id)
            current = self._get_user_mode(user_id)
            pipeline = self.pipelines[current]
            stats = pipeline.get_stats()
            log_stats = self.logger.get_stats()

            await update.message.reply_text(
                f"СТАТИСТИКА СИСТЕМЫ\n\n"
                f"Текущая модель: {MODE_LABELS[current]}\n\n"
                f"База знаний:\n"
                f"  Документов (чанков): {stats['vector_store']['count']}\n"
                f"  Модель: {stats['model']}\n\n"
                f"Кеш:\n"
                f"  Записей: {stats['cache']['total_entries']}\n\n"
                f"Логи:\n"
                f"  Всего запросов: {log_stats['total_requests']}\n"
                f"  Из кеша: {log_stats['cached_requests']}\n"
                f"  Уникальных пользователей: {log_stats['unique_users']}\n"
                f"  Среднее время ответа: {log_stats['avg_response_time_ms']:.0f} мс"
            )
        except Exception as e:
            await update.message.reply_text(f"Ошибка: {e}")

    async def cmd_logs(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            user_id = str(update.effective_user.id)
            csv_content = self.logger.export_to_csv(user_id=user_id)

            if not csv_content:
                await update.message.reply_text("Логов пока нет.")
                return

            filename = f"logs_{user_id}_{int(time.time())}.csv"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(csv_content)

            with open(filename, 'rb') as f:
                await update.message.reply_document(
                    document=f,
                    filename=filename,
                    caption="Ваши логи взаимодействий"
                )
            os.remove(filename)
        except Exception as e:
            await update.message.reply_text(f"Ошибка: {e}")

    # ── Обработка сообщений ──

    async def on_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_message = update.message.text

        if user_message == BTN_MODE:
            return await self.cmd_mode(update, context)
        if user_message == BTN_STATS:
            return await self.cmd_stats(update, context)
        if user_message == BTN_LOGS:
            return await self.cmd_logs(update, context)
        if user_message == BTN_HELP:
            return await self.cmd_help(update, context)
        if user_message == BTN_REINDEX:
            return await self.cmd_reindex(update, context)

        user = update.effective_user
        user_id = str(user.id)
        username = user.username or user.first_name or "Unknown"

        mode = self._get_user_mode(user_id)
        pipeline = self.pipelines[mode]

        await update.message.chat.send_action(action="typing")
        start_time = time.time()

        try:
            result = pipeline.query(user_message)
            response_time_ms = int((time.time() - start_time) * 1000)

            self.logger.log_interaction(
                query=user_message,
                response=result['answer'],
                source="telegram",
                user_id=user_id,
                username=username,
                from_cache=result['from_cache'],
                response_time_ms=response_time_ms
            )

            answer = result['answer']
            mode_label = MODE_LABELS[mode]
            header = f"[{mode_label}]\n\n"

            max_len = 4000 - len(header)
            if len(answer) <= max_len:
                await update.message.reply_text(header + answer)
            else:
                parts = [answer[i:i + max_len] for i in range(0, len(answer), max_len)]
                for i, part in enumerate(parts):
                    prefix = header if i == 0 else ""
                    await update.message.reply_text(prefix + part)

            if result['from_cache']:
                await update.message.reply_text("(ответ из кеша)")

        except Exception as e:
            error_msg = f"Ошибка: {e}"
            await update.message.reply_text(error_msg)
            self.logger.log_interaction(
                query=user_message,
                response=error_msg,
                source="telegram",
                user_id=user_id,
                username=username,
                from_cache=False,
                response_time_ms=int((time.time() - start_time) * 1000)
            )

    def run(self):
        print("Telegram бот запущен! Нажмите Ctrl+C для остановки.")
        self.app.run_polling()


# ─── Точка входа ────────────────────────────────────────────────────────

def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token or token == "your_telegram_bot_token_here":
        print("Ошибка: TELEGRAM_BOT_TOKEN не задан в .env")
        print("Получите токен у @BotFather в Telegram и впишите в .env")
        sys.exit(1)

    print("=" * 60)
    print("ИНИЦИАЛИЗАЦИЯ ЕДИНОГО TELEGRAM БОТА")
    print("=" * 60)

    pipelines = {}

    # ChatGPT (OpenAI)
    if os.getenv("OPENAI_API_KEY"):
        try:
            print("\n[1/2] Инициализация ChatGPT (OpenAI)...")
            pipelines['chatgpt'] = init_api_pipeline()
            print("[+] ChatGPT готов")
        except Exception as e:
            print(f"[-] Ошибка ChatGPT: {e}")
    else:
        print("\n[-] OPENAI_API_KEY не задан — ChatGPT недоступен")

    # GigaChat (Сбер)
    if os.getenv("GIGACHAT_AUTH_KEY") and os.getenv("GIGACHAT_RQUID"):
        try:
            print("\n[2/2] Инициализация GigaChat...")
            pipelines['gigachat'] = init_giga_pipeline()
            print("[+] GigaChat готов")
        except Exception as e:
            print(f"[-] Ошибка GigaChat: {e}")
    else:
        print("\n[-] GIGACHAT_AUTH_KEY / GIGACHAT_RQUID не заданы — GigaChat недоступен")

    if not pipelines:
        print("\nНи одна модель не инициализирована. Проверьте ключи в .env")
        sys.exit(1)

    # Логгер
    print("\n[*] Инициализация логгера...")
    logger = DatabaseLogger(db_path=str(ROOT_DIR / "bot_logs.db"))
    print("[+] Логгер готов")

    available = ", ".join(MODE_LABELS[k] for k in pipelines)
    print(f"\nДоступные модели: {available}")
    print("=" * 60)

    bot = UnifiedTelegramBot(token=token, pipelines=pipelines, logger=logger)
    bot.run()


if __name__ == "__main__":
    main()
