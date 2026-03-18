"""
Telegram бот для RAG-ассистента (GigaChat mode).

Бот позволяет пользователям задавать вопросы ассистенту через Telegram
и получать ответы на основе векторного поиска и GigaChat.
"""

import os
import time
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes
)
from rag_pipeline import RAGPipeline
from db_logger import DatabaseLogger


class TelegramRAGBot:
    """
    Telegram бот для RAG-ассистента (GigaChat mode).
    
    Обрабатывает команды и сообщения от пользователей,
    логирует все взаимодействия в базу данных.
    """
    
    def __init__(
        self,
        token: str,
        pipeline: RAGPipeline,
        logger: DatabaseLogger
    ):
        """
        Инициализация Telegram бота.
        
        Args:
            token: Токен Telegram бота от @BotFather
            pipeline: Экземпляр RAG pipeline
            logger: Экземпляр логгера базы данных
        """
        self.pipeline = pipeline
        self.logger = logger
        
        self.application = Application.builder().token(token).build()
        
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("stats", self.stats_command))
        self.application.add_handler(CommandHandler("logs", self.logs_command))
        
        self.application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message)
        )
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /start"""
        welcome_message = (
            "Добро пожаловать в RAG-ассистент (GigaChat mode)!\n\n"
            "Я могу отвечать на ваши вопросы, используя базу знаний "
            "и GigaChat от Сбера для генерации ответов.\n\n"
            "Доступные команды:\n"
            "/help - показать справку\n"
            "/stats - статистика системы\n"
            "/logs - получить логи в CSV формате\n\n"
            "Просто напишите мне вопрос, и я постараюсь на него ответить!"
        )
        await update.message.reply_text(welcome_message)
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /help"""
        help_text = (
            "Справка по использованию бота:\n\n"
            "- Просто напишите вопрос - я отвечу на основе базы знаний\n"
            "- Использую RAG (Retrieval-Augmented Generation) для точных ответов\n"
            "- Ответы кешируются для быстрой работы\n\n"
            "Команды:\n"
            "/start - начать работу с ботом\n"
            "/help - показать эту справку\n"
            "/stats - статистика системы (документы, кеш, логи)\n"
            "/logs - получить логи взаимодействий в CSV формате"
        )
        await update.message.reply_text(help_text)
    
    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /stats"""
        try:
            stats = self.pipeline.get_stats()
            log_stats = self.logger.get_stats()
            
            stats_message = (
                "СТАТИСТИКА СИСТЕМЫ:\n\n"
                "База знаний:\n"
                f"  Документов в ChromaDB: {stats['vector_store']['count']}\n"
                f"  Модель LLM: {stats['model']}\n\n"
                "Кеш:\n"
                f"  Записей в кеше: {stats['cache']['total_entries']}\n"
                f"  Размер БД: {stats['cache']['db_size_mb']:.2f} MB\n\n"
                "Логи:\n"
                f"  Всего запросов: {log_stats['total_requests']}\n"
                f"  Из кеша: {log_stats['cached_requests']}\n"
                f"  Уникальных пользователей: {log_stats['unique_users']}\n"
                f"  Среднее время ответа: {log_stats['avg_response_time_ms']:.0f} мс"
            )
            
            await update.message.reply_text(stats_message)
            
        except Exception as e:
            await update.message.reply_text(f"Ошибка при получении статистики: {str(e)}")
    
    async def logs_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /logs - экспорт логов в CSV"""
        try:
            user_id = str(update.effective_user.id)
            
            csv_content = self.logger.export_to_csv(user_id=user_id)
            
            if not csv_content:
                await update.message.reply_text(
                    "Логов для вашего пользователя пока не найдено."
                )
                return
            
            filename = f"logs_{user_id}_{int(time.time())}.csv"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(csv_content)
            
            with open(filename, 'rb') as f:
                await update.message.reply_document(
                    document=f,
                    filename=filename,
                    caption="Ваши логи взаимодействий с ботом"
                )
            
            os.remove(filename)
            
        except Exception as e:
            await update.message.reply_text(f"Ошибка при экспорте логов: {str(e)}")
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик текстовых сообщений от пользователей"""
        user_message = update.message.text
        user = update.effective_user
        user_id = str(user.id)
        username = user.username or user.first_name or "Unknown"
        
        await update.message.chat.send_action(action="typing")
        
        start_time = time.time()
        
        try:
            result = self.pipeline.query(user_message)
            
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
            max_length = 4000
            if len(answer) <= max_length:
                await update.message.reply_text(answer)
            else:
                parts = [answer[i:i+max_length] for i in range(0, len(answer), max_length)]
                for part in parts:
                    await update.message.reply_text(part)
            
            if result['from_cache']:
                await update.message.reply_text("(ответ из кеша)")
        
        except Exception as e:
            error_message = f"Произошла ошибка при обработке запроса: {str(e)}"
            await update.message.reply_text(error_message)
            
            self.logger.log_interaction(
                query=user_message,
                response=error_message,
                source="telegram",
                user_id=user_id,
                username=username,
                from_cache=False,
                response_time_ms=int((time.time() - start_time) * 1000)
            )
    
    def run(self):
        """Запускает бота"""
        print("Запуск Telegram бота (GigaChat mode)...")
        print("Бот готов к работе! Нажмите Ctrl+C для остановки.")
        self.application.run_polling()
