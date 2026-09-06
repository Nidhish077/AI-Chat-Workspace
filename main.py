import asyncio
import json
import os
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from google import genai
from sqlalchemy import DateTime, Integer, String, Text, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

load_dotenv()

GEMINI_MODEL = "gemini-3.6-flash"
AI_PREFIX = "@ai"


def async_database_url(raw: str) -> str:
    url = raw.strip()
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://") :]
    if url.startswith("postgresql://") and "+asyncpg" not in url:
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


DATABASE_URL = async_database_url(os.getenv("DATABASE_URL", ""))
if not DATABASE_URL:
    raise RuntimeError("Set DATABASE_URL in .env to connect to PostgreSQL.")

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    connect_args={"statement_cache_size": 0},
)
SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    channel_id: Mapped[str] = mapped_column(String(120), index=True)
    username: Mapped[str] = mapped_column(String(80))
    text: Mapped[str] = mapped_column(Text)
    kind: Mapped[str] = mapped_column(String(20), default="chat")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    await engine.dispose()


app = FastAPI(title="AI Messaging Workspace", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_genai_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "your_key_here":
        raise RuntimeError("Set GEMINI_API_KEY in .env to use @ai replies.")
    return genai.Client(api_key=api_key)


async def save_message(channel_id: str, username: str, text: str, kind: str) -> None:
    async with SessionLocal() as session:
        session.add(
            Message(
                channel_id=channel_id,
                username=username,
                text=text,
                kind=kind,
            )
        )
        await session.commit()


class ConnectionManager:
    def __init__(self) -> None:
        self.channels: dict[str, set[WebSocket]] = defaultdict(set)

    async def connect(self, websocket: WebSocket, channel_id: str) -> None:
        await websocket.accept()
        self.channels[channel_id].add(websocket)
        await self.broadcast(
            channel_id,
            {
                "type": "presence",
                "channel": channel_id,
                "count": len(self.channels[channel_id]),
            },
        )

    def disconnect(self, websocket: WebSocket, channel_id: str) -> None:
        sockets = self.channels.get(channel_id)
        if not sockets:
            return
        sockets.discard(websocket)
        if not sockets:
            self.channels.pop(channel_id, None)

    async def broadcast(self, channel_id: str, payload: dict[str, Any]) -> None:
        sockets = list(self.channels.get(channel_id, set()))
        if not sockets:
            return
        message = json.dumps(payload)
        stale: list[WebSocket] = []
        for websocket in sockets:
            try:
                await websocket.send_text(message)
            except Exception:
                stale.append(websocket)
        for websocket in stale:
            self.disconnect(websocket, channel_id)


manager = ConnectionManager()


def parse_incoming(raw: str) -> tuple[str, str]:
    try:
        data = json.loads(raw)
        user = str(data.get("user") or data.get("username") or "Guest").strip() or "Guest"
        text = str(data.get("text") or data.get("message") or "").strip()
        return user, text
    except json.JSONDecodeError:
        return "Guest", raw.strip()


async def stream_ai_reply(channel_id: str, prompt: str) -> None:
    await manager.broadcast(
        channel_id,
        {"type": "ai_start", "user": "Gemini", "channel": channel_id},
    )
    full_text = ""
    try:
        client = get_genai_client()
        stream = client.aio.models.generate_content_stream(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        if asyncio.iscoroutine(stream) or not hasattr(stream, "__aiter__"):
            stream = await stream
        async for chunk in stream:
            piece = getattr(chunk, "text", None) or ""
            if not piece:
                continue
            full_text += piece
            await manager.broadcast(
                channel_id,
                {
                    "type": "ai_chunk",
                    "user": "Gemini",
                    "text": piece,
                    "channel": channel_id,
                },
            )
        reply = full_text or "(empty reply)"
        await save_message(channel_id, "Gemini", reply, "ai")
        await manager.broadcast(
            channel_id,
            {
                "type": "ai_end",
                "user": "Gemini",
                "text": reply,
                "channel": channel_id,
            },
        )
    except Exception as exc:
        await manager.broadcast(
            channel_id,
            {
                "type": "error",
                "user": "system",
                "text": f"Gemini error: {exc}",
                "channel": channel_id,
            },
        )
        await manager.broadcast(
            channel_id,
            {"type": "ai_end", "user": "Gemini", "text": "", "channel": channel_id},
        )


@app.get("/")
async def index() -> FileResponse:
    return FileResponse("templates/index.html")


@app.get("/history/{channel_id}")
async def channel_history(channel_id: str) -> list[dict[str, Any]]:
    channel_id = channel_id.strip() or "general"
    async with SessionLocal() as session:
        result = await session.execute(
            select(Message)
            .where(Message.channel_id == channel_id)
            .order_by(Message.created_at.asc(), Message.id.asc())
        )
        rows = result.scalars().all()
    return [
        {
            "id": row.id,
            "channel": row.channel_id,
            "user": row.username,
            "text": row.text,
            "kind": row.kind,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }
        for row in rows
    ]


@app.websocket("/ws/{channel_id}")
async def websocket_endpoint(websocket: WebSocket, channel_id: str) -> None:
    channel_id = channel_id.strip() or "general"
    await manager.connect(websocket, channel_id)
    try:
        while True:
            raw = await websocket.receive_text()
            user, text = parse_incoming(raw)
            if not text:
                continue

            await save_message(channel_id, user, text, "chat")
            await manager.broadcast(
                channel_id,
                {
                    "type": "chat",
                    "user": user,
                    "text": text,
                    "channel": channel_id,
                },
            )

            if text.lower().startswith(AI_PREFIX):
                prompt = text[len(AI_PREFIX) :].lstrip(" :,")
                if not prompt:
                    prompt = "Say hello and ask how you can help."
                asyncio.create_task(stream_ai_reply(channel_id, prompt))
    except WebSocketDisconnect:
        manager.disconnect(websocket, channel_id)
        await manager.broadcast(
            channel_id,
            {
                "type": "presence",
                "channel": channel_id,
                "count": len(manager.channels.get(channel_id, set())),
            },
        )
