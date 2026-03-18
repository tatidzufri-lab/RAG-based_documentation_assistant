"""
Консольное приложение для взаимодействия с RAG ассистентом (API mode).

Поддерживает два режима работы:
1. Консольный интерактивный режим
2. Telegram бот
"""

import sys
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from rag_pipeline import RAGPipeline
from db_logger import DatabaseLogger

# Загрузка переменных окружения из .env файла
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()


def print_banner():
    """Вывод приветственного баннера."""
    banner = """
╔══════════════════════════════════════════════════════════╗
║         RAG Ассистент (API Mode)                        ║
║  Retrieval-Augmented Generation через OpenAI API        ║
╚══════════════════════════════════════════════════════════╝
    """
    print(banner)
    print("Введите 'exit' или 'quit' для выхода")
    print("Введите 'stats' для просмотра статистики")
    print("Введите 'clear' для очистки кеша")
    print("Введите 'logs' для экспорта логов в CSV\n")


def print_response(result: dict):
    """
    Форматированный вывод ответа.
    
    Args:
        result: словарь с результатом запроса
    """
    print(f"\n{'─'*60}")
    print(f"Вопрос: {result['query']}")
    print(f"{'─'*60}")
    
    if result['from_cache']:
        print("Источник: КЕШ")
        if 'cached_at' in result:
            print(f"   Сохранено: {result['cached_at']}")
    else:
        print(f"Источник: OpenAI API ({result.get('model', 'LLM')})")
        print(f"   Использовано документов: {len(result.get('context_docs', []))}")
    
    print(f"\nОтвет:\n{result['answer']}")
    
    if not result['from_cache'] and result.get('context_docs'):
        print(f"\nИспользованный контекст:")
        for i, doc in enumerate(result['context_docs'][:2], 1):
            preview = doc['text'][:150] + "..." if len(doc['text']) > 150 else doc['text']
            print(f"   {i}. {preview}")
    
    print(f"{'─'*60}\n")


def print_stats(pipeline: RAGPipeline, logger: DatabaseLogger):
    """
    Вывод статистики системы.
    
    Args:
        pipeline: экземпляр RAG pipeline
        logger: экземпляр логгера
    """
    stats = pipeline.get_stats()
    log_stats = logger.get_stats()
    
    print(f"\n{'═'*60}")
    print("СТАТИСТИКА СИСТЕМЫ")
    print(f"{'═'*60}")
    
    print("\nВекторное хранилище:")
    print(f"   Коллекция: {stats['vector_store']['name']}")
    print(f"   Документов: {stats['vector_store']['count']}")
    print(f"   Директория: {stats['vector_store']['persist_directory']}")
    
    print("\nКеш:")
    print(f"   Записей: {stats['cache']['total_entries']}")
    print(f"   Размер БД: {stats['cache']['db_size_mb']:.2f} MB")
    if stats['cache']['oldest_entry']:
        print(f"   Первая запись: {stats['cache']['oldest_entry']}")
    if stats['cache']['newest_entry']:
        print(f"   Последняя запись: {stats['cache']['newest_entry']}")
    
    print(f"\nМодель: {stats['model']}")
    print(f"Режим: {stats['mode']}")
    
    print(f"\nЛоги:")
    print(f"   Всего запросов: {log_stats['total_requests']}")
    print(f"   Из кеша: {log_stats['cached_requests']}")
    print(f"   Уникальных пользователей: {log_stats['unique_users']}")
    print(f"   Среднее время ответа: {log_stats['avg_response_time_ms']:.0f} мс")
    print(f"   По источникам: {log_stats['by_source']}")
    print(f"{'═'*60}\n")


def interactive_mode(pipeline: RAGPipeline, logger: DatabaseLogger):
    """Интерактивный консольный режим."""
    print_banner()
    
    while True:
        try:
            user_input = input("Ваш вопрос: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\nДо свидания!")
                break
            
            if user_input.lower() == 'stats':
                print_stats(pipeline, logger)
                continue
            
            if user_input.lower() == 'clear':
                confirm = input("Вы уверены, что хотите очистить кеш? (yes/no): ")
                if confirm.lower() in ['yes', 'y', 'да']:
                    pipeline.cache.clear()
                    print("Кеш очищен")
                continue
            
            if user_input.lower() == 'logs':
                filename = f"logs_console_{int(time.time())}.csv"
                result = logger.export_to_csv(output_path=filename, source="console")
                if result:
                    print(f"Логи экспортированы в файл: {filename}")
                else:
                    print("Логов пока нет")
                continue
            
            if not user_input:
                print("Пожалуйста, введите вопрос\n")
                continue
            
            start_time = time.time()
            
            result = pipeline.query(user_input)
            
            response_time_ms = int((time.time() - start_time) * 1000)
            
            logger.log_interaction(
                query=user_input,
                response=result['answer'],
                source="console",
                from_cache=result['from_cache'],
                response_time_ms=response_time_ms
            )
            
            print_response(result)
            
        except KeyboardInterrupt:
            print("\n\nПрервано пользователем. До свидания!")
            break
        except Exception as e:
            print(f"\nОшибка: {e}\n")


def main():
    """Главная функция приложения."""
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Ошибка: переменная окружения OPENAI_API_KEY не установлена")
        print("\nУстановите её следующим образом:")
        print("  Windows (PowerShell): $env:OPENAI_API_KEY='your-key'")
        print("  Windows (CMD): set OPENAI_API_KEY=your-key")
        print("  Linux/Mac: export OPENAI_API_KEY='your-key'")
        sys.exit(1)
    
    try:
        print("Инициализация системы...\n")
        shared_data = str(Path(__file__).parent.parent / 'data')
        pipeline = RAGPipeline(
            collection_name="api_rag_collection",
            cache_db_path="api_rag_cache.db",
            data_path=shared_data,
            model="gpt-4o-mini"
        )
        
        logger = DatabaseLogger(db_path="api_logs.db")
        print("Логгер базы данных инициализирован")
        
        print("\nСистема готова к работе!\n")
        
    except Exception as e:
        print(f"Ошибка инициализации: {e}")
        sys.exit(1)
    
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
    
    print(f"{'='*60}")
    print("ВЫБОР РЕЖИМА РАБОТЫ")
    print(f"{'='*60}")
    print("\n1. Интерактивный режим - задавайте вопросы в консоли")
    if telegram_token:
        print("2. Telegram бот - запуск бота для Telegram")
    else:
        print("2. Telegram бот (недоступен - задайте TELEGRAM_BOT_TOKEN в .env)")
    print()
    
    mode = input("Выберите режим (1 или 2, по умолчанию 1): ").strip()
    
    if mode == '2' and telegram_token:
        from telegram_bot import TelegramRAGBot
        
        print(f"\n{'='*60}")
        print("ЗАПУСК TELEGRAM БОТА (API mode)")
        print(f"{'='*60}")
        bot = TelegramRAGBot(
            token=telegram_token,
            pipeline=pipeline,
            logger=logger
        )
        bot.run()
    else:
        interactive_mode(pipeline, logger)


if __name__ == "__main__":
    main()
