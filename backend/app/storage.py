"""Dev: SQLite. Public deployment: Supabase PostgREST with end-user JWT + RLS."""
from datetime import datetime, timezone
import json
import sqlite3
from pathlib import Path
from uuid import uuid4
import httpx


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self._conn() as conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS chats (
                    id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, title TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY, chat_id TEXT NOT NULL,
                    owner_id TEXT NOT NULL, role TEXT NOT NULL,
                    content TEXT NOT NULL, created_at TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY(chat_id) REFERENCES chats(id)
                );
                CREATE INDEX IF NOT EXISTS ix_chats_owner ON chats(owner_id);
                CREATE INDEX IF NOT EXISTS ix_messages_chat ON messages(chat_id, owner_id);
            ''')

    def _conn(self):
        conn = sqlite3.connect(self.path, timeout=5)
        conn.row_factory = sqlite3.Row
        return conn

    async def list_chats(self, owner: str):
        with self._conn() as conn:
            rows = conn.execute('SELECT id,title,created_at FROM chats WHERE owner_id=? ORDER BY created_at DESC LIMIT 100', (owner,)).fetchall()
            return [dict(r) for r in rows]

    async def get_chat(self, owner: str, chat_id: str):
        with self._conn() as conn:
            row = conn.execute('SELECT id,title,created_at FROM chats WHERE id=? AND owner_id=?', (chat_id, owner)).fetchone()
            return dict(row) if row else None

    async def create_chat(self, owner: str, title: str):
        item = {'id': str(uuid4()), 'owner_id': owner, 'title': title[:90], 'created_at': now()}
        with self._conn() as conn:
            conn.execute('INSERT INTO chats VALUES (:id,:owner_id,:title,:created_at)', item)
        return item

    async def add_message(self, owner: str, chat_id: str, role: str, content: str, metadata: dict | None = None):
        # Do not trust chat IDs supplied by clients.
        if await self.get_chat(owner, chat_id) is None:
            raise ValueError('Chat does not exist for this user')
        item = {'id': str(uuid4()), 'chat_id': chat_id, 'owner_id': owner, 'role': role,
                'content': content, 'created_at': now(), 'metadata': json.dumps(metadata or {})}
        with self._conn() as conn:
            conn.execute('INSERT INTO messages VALUES (:id,:chat_id,:owner_id,:role,:content,:created_at,:metadata)', item)
        return {**item, 'metadata': metadata or {}}

    async def list_messages(self, owner: str, chat_id: str):
        with self._conn() as conn:
            rows = conn.execute('SELECT * FROM messages WHERE chat_id=? AND owner_id=? ORDER BY created_at,id LIMIT 200', (chat_id, owner)).fetchall()
            return [{**dict(r), 'metadata': json.loads(r['metadata'])} for r in rows]


class SupabaseStore:
    def __init__(self, url: str, anon_key: str, jwt: str):
        self.url = url.rstrip('/')
        self.jwt = jwt
        self.headers = {'apikey': anon_key, 'Authorization': f'Bearer {jwt}',
                        'Content-Type': 'application/json'}

    async def _request(self, method: str, table: str, **kwargs):
        async with httpx.AsyncClient(timeout=15) as client:
            result = await client.request(method, f'{self.url}/rest/v1/{table}',
                                          headers=self.headers | {'Prefer': 'return=representation'}, **kwargs)
            result.raise_for_status()
            return result.json()

    async def list_chats(self, owner: str):
        return await self._request('GET', 'chats', params={'select': 'id,title,created_at',
            'owner_id': f'eq.{owner}', 'order': 'created_at.desc', 'limit': '100'})

    async def get_chat(self, owner: str, chat_id: str):
        rows = await self._request('GET', 'chats', params={'select': 'id,title,created_at',
            'id': f'eq.{chat_id}', 'owner_id': f'eq.{owner}', 'limit': '1'})
        return rows[0] if rows else None

    async def create_chat(self, owner: str, title: str):
        rows = await self._request('POST', 'chats', json={'owner_id': owner, 'title': title[:90]})
        return rows[0]

    async def add_message(self, owner: str, chat_id: str, role: str, content: str, metadata: dict | None = None):
        if await self.get_chat(owner, chat_id) is None:
            raise ValueError('Chat does not exist for this user')
        rows = await self._request('POST', 'messages', json={'owner_id': owner,
            'chat_id': chat_id, 'role': role, 'content': content, 'metadata': metadata or {}})
        return rows[0]

    async def list_messages(self, owner: str, chat_id: str):
        return await self._request('GET', 'messages', params={'select': 'id,chat_id,role,content,created_at,metadata',
            'chat_id': f'eq.{chat_id}', 'owner_id': f'eq.{owner}',
            'order': 'created_at.asc', 'limit': '200'})
