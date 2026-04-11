import logging
import os
from typing import Any

from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType

logger = logging.getLogger("hbllm_assistant.telegram")


class TelegramInterfaceNode:
    """
    A foundational driver that intercepts external chat messages (e.g. from Telegram)
    and passes them seamlessly into the HBLLM MessageBus.
    """

    def __init__(self, node_id: str = "assistant_telegram"):
        self.node_id = node_id
        self.bot_app: Any = None
        self.bus: InProcessBus | None = None

    async def run(self, bus: InProcessBus) -> None:
        """
        Registers this interface onto the core MessageBus.
        """
        self.bus = bus
        logger.info(f"[{self.node_id}] Initializing Telegram comms driver...")

        # Subscribe to any outputs directed back to the user
        await bus.subscribe("sensory.output", self._handle_reply)

        # Init Telegram App
        token = os.environ.get("TELEGRAM_BOT_TOKEN")
        if not token:
            logger.warning(
                f"[{self.node_id}] TELEGRAM_BOT_TOKEN not found. Node will run in stub mode."
            )
            return

        try:
            from telegram import Update  # type: ignore
            from telegram.ext import (  # type: ignore
                Application,
                ContextTypes,
                MessageHandler,
                filters,
            )

            self.bot_app = Application.builder().token(token).build()

            async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:  # type: ignore
                if update.message and update.message.text:
                    await self.publish_incoming(
                        update.message.text, chat_id=update.effective_chat.id
                    )

            self.bot_app.add_handler(
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)  # type: ignore
            )

            await self.bot_app.initialize()  # type: ignore
            await self.bot_app.start()  # type: ignore
            await self.bot_app.updater.start_polling()  # type: ignore
            logger.info(f"[{self.node_id}] Telegram bot polling started.")

        except ImportError:
            logger.error("python-telegram-bot not installed. Please pip install it.")

    async def publish_incoming(self, text: str, chat_id: int) -> None:
        """
        Simulates an incoming text from Telegram to the core HBLLM brain.
        """
        if not self.bus:
            return

        msg = Message(
            topic="router.query",
            type=MessageType.EVENT,
            source_node_id=self.node_id,
            payload={"text": text, "chat_id": chat_id},
        )
        await self.bus.publish("router.query", msg)

    async def _handle_reply(self, message: Message) -> None:
        """
        Placeholder: push HBLLM's internal string replies back out to the Telegram API.
        """
        if message.payload.get("target") == "telegram":
            text = message.payload.get("text")
            chat_id = message.payload.get("chat_id")

            if self.bot_app and self.bot_app.bot and chat_id and text:
                try:
                    await self.bot_app.bot.send_message(chat_id=chat_id, text=text)
                except Exception as e:
                    logger.error(f"[{self.node_id}] Failed to send telegram message: {e}")


__plugin__ = {
    "name": "telegram_node",
    "version": "0.1.0",
    "description": "Connects HBLLM back to your mobile device via Telegram BOT API.",
}


async def register(bus: Any, registry: Any = None) -> Any:
    node = TelegramInterfaceNode(node_id="user_telegram")
    await node.run(bus)
    return node
