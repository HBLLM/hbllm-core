"""
Redis-backed MessageBus for distributed operation across multiple processes/machines.

Optionally signs messages with HMAC-SHA256 when an auth_secret is provided,
preventing unauthorized nodes from injecting messages into the cluster.
"""

import asyncio
import hashlib
import hmac
import logging
from collections import defaultdict
from typing import Any

import redis.asyncio as redis
from hbllm.network.bus import Subscription, MessageHandler, MessageBus
from hbllm.network.messages import Message

logger = logging.getLogger(__name__)

class RedisBus(MessageBus):
    """
    Distributed MessageBus using Redis Pub/Sub.
    
    Nodes can run in separate processes or servers. Uses `psubscribe("*")` 
    to observe all traffic, mimicking the InProcessBus behavior where any node 
    can correlate request-responses dynamically without explicit response-topic subscriptions.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379", auth_secret: str = ""):
        self.redis_url = redis_url
        self.auth_secret = auth_secret
        self.client: redis.Redis | None = None
        self.pubsub: redis.client.PubSub | None = None
        
        self._subscriptions: dict[str, list[Subscription]] = defaultdict(list)
        self._pending_requests: dict[str, asyncio.Future[Message]] = {}
        
        self._running = False
        self._dispatch_task: asyncio.Task[None] | None = None
        self._sub_counter = 0

    def _sign(self, payload_json: str) -> str:
        """Compute HMAC-SHA256 signature for a message payload."""
        return hmac.new(
            self.auth_secret.encode(), payload_json.encode(), hashlib.sha256
        ).hexdigest()[:32]

    def _verify(self, payload_json: str, signature: str) -> bool:
        """Verify a message signature."""
        expected = self._sign(payload_json)
        return hmac.compare_digest(expected, signature)

    async def start(self) -> None:
        """Connect to Redis and start the dispatch loop."""
        self.client = redis.from_url(self.redis_url, decode_responses=True)
        self.pubsub = self.client.pubsub()
        # Subscribe to all topics to catch responses and broadcast identically to InProcessBus
        await self.pubsub.psubscribe("*")
        
        self._running = True
        self._dispatch_task = asyncio.create_task(self._dispatch_loop())
        logger.info(f"RedisBus started connected to {self.redis_url}")

    async def stop(self) -> None:
        """Stop the bus and clean up."""
        self._running = False

        # Cancel all pending requests
        for future in self._pending_requests.values():
            if not future.done():
                future.cancel()
        self._pending_requests.clear()

        # Stop dispatch loop
        if self._dispatch_task and not self._dispatch_task.done():
            self._dispatch_task.cancel()
            try:
                await self._dispatch_task
            except asyncio.CancelledError:
                pass

        if self.pubsub:
            await self.pubsub.close()
        if self.client:
            await self.client.aclose()

        logger.info("RedisBus stopped")

    async def publish(self, topic: str, message: Message) -> None:
        """Publish a message to Redis, optionally signing with HMAC."""
        if not self._running or not self.client:
            logger.warning("Bus is not running, message dropped: %s", message.id)
            return

        payload_json = message.model_dump_json()

        # Sign if auth is configured
        if self.auth_secret:
            sig = self._sign(payload_json)
            wire = f"{sig}|{payload_json}"
        else:
            wire = payload_json

        await self.client.publish(topic, wire)
        
        # When a node fulfills a request locally, it calls bus.publish(response)
        # We intercept it right here if it happens to be our own local pending request,
        # otherwise we'd receive it from Redis a few milliseconds later anyway.
        # This mirrors InProcessBus exactly.
        if message.correlation_id and message.correlation_id in self._pending_requests:
            future = self._pending_requests.pop(message.correlation_id)
            if not future.done():
                future.set_result(message)

    async def request(self, topic: str, message: Message, timeout: float = 30.0) -> Message:
        """Send a request and wait for a correlated response."""
        # Create a future for the response
        future: asyncio.Future[Message] = asyncio.get_event_loop().create_future()
        self._pending_requests[message.id] = future

        # Publish the request
        await self.publish(topic, message)

        try:
            # Wait for correlated response
            response = await asyncio.wait_for(future, timeout=timeout)
            return response
        except asyncio.TimeoutError:
            self._pending_requests.pop(message.id, None)
            raise TimeoutError(
                f"Request {message.id} to topic '{topic}' timed out after {timeout}s"
            )

    async def subscribe(self, topic: str, handler: MessageHandler) -> Subscription:
        """Subscribe a handler to a topic locally."""
        self._sub_counter += 1
        sub = Subscription(topic=topic, handler=handler, sub_id=f"redis_sub-{self._sub_counter}")
        self._subscriptions[topic].append(sub)
        logger.debug("Subscribed locally to '%s' (id=%s)", topic, sub.id)
        return sub

    async def unsubscribe(self, subscription: Subscription) -> None:
        """Remove a local subscription."""
        subscription.cancel()
        subs = self._subscriptions.get(subscription.topic, [])
        self._subscriptions[subscription.topic] = [s for s in subs if s.id != subscription.id]
        logger.debug("Unsubscribed local '%s' from '%s'", subscription.id, subscription.topic)

    async def _dispatch_loop(self) -> None:
        """Main loop: dequeue messages from Redis and dispatch locally."""
        if not self.pubsub:
            return
            
        while self._running:
            try:
                # get_message is non-blocking technically, but with timeout it yields
                msg_dict = await self.pubsub.get_message(ignore_subscribe_messages=True, timeout=0.1)
                if msg_dict is None:
                    continue
                    
                if msg_dict["type"] in ("message", "pmessage"):
                    topic = msg_dict["channel"]
                    raw_data = msg_dict["data"]

                    # Verify HMAC signature if auth is configured
                    if self.auth_secret and "|" in raw_data:
                        sig, payload_json = raw_data.split("|", 1)
                        if not self._verify(payload_json, sig):
                            logger.warning("Dropped message with invalid signature on topic '%s'", topic)
                            continue
                    else:
                        payload_json = raw_data
                    
                    try:
                        message = Message.model_validate_json(payload_json)
                    except Exception as e:
                        logger.error(f"Failed to parse Redis message: {e}")
                        continue

                    # Check if this is a response to a pending request initiated by THIS node
                    if message.correlation_id and message.correlation_id in self._pending_requests:
                        future = self._pending_requests.pop(message.correlation_id)
                        if not future.done():
                            future.set_result(message)
                        continue

                    # Dispatch to local subscribers
                    await self._dispatch_to_subscribers(topic, message)

            except (asyncio.TimeoutError, asyncio.CancelledError):
                continue
            except redis.ConnectionError:
                logger.error("Redis connection error in dispatch loop. Retrying...")
                await asyncio.sleep(2)
            except Exception as e:
                logger.error(f"Error in Redis dispatch loop: {e}")

    async def _dispatch_to_subscribers(self, topic: str, message: Message) -> None:
        """Dispatch a message to all active local subscribers for a topic."""
        matching_topics = self._get_matching_topics(topic)

        for match_topic in matching_topics:
            for sub in self._subscriptions.get(match_topic, []):
                if not sub.active:
                    continue
                    
                async def _run_handler(s=sub, t=topic, m=message):
                    try:
                        response = await s.handler(m)
                        if response is not None:
                            if response.correlation_id is None:
                                response.correlation_id = m.id
                            # Publish the response back to Redis
                            await self.publish(response.topic, response)
                    except Exception:
                        logger.exception("Error in handler for topic '%s', message %s", t, m.id)
                        
                asyncio.create_task(_run_handler())

    def _get_matching_topics(self, topic: str) -> list[str]:
        """Get all registered topics that match (exact + wildcard)."""
        matches = []
        for registered_topic in self._subscriptions:
            if registered_topic == topic:
                matches.append(registered_topic)
            elif registered_topic.endswith(".*"):
                prefix = registered_topic[:-2]
                if topic.startswith(prefix):
                    matches.append(registered_topic)
            elif registered_topic == "*":
                matches.append(registered_topic)
        return matches
