-- Run in Supabase SQL Editor before STORAGE_MODE=supabase.
-- Client uses anonymous Supabase Auth; server forwards the user's JWT.
CREATE TABLE IF NOT EXISTS public.chats (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_chats_owner_created ON public.chats(owner_id, created_at DESC);

CREATE TABLE IF NOT EXISTS public.messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chat_id UUID NOT NULL REFERENCES public.chats(id) ON DELETE CASCADE,
    owner_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_messages_chat_created ON public.messages(chat_id, created_at);

ALTER TABLE public.chats ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.messages ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS chats_owner_read ON public.chats;
CREATE POLICY chats_owner_read ON public.chats FOR SELECT TO authenticated
USING (owner_id = (SELECT auth.uid()));
DROP POLICY IF EXISTS chats_owner_create ON public.chats;
CREATE POLICY chats_owner_create ON public.chats FOR INSERT TO authenticated
WITH CHECK (owner_id = (SELECT auth.uid()));

DROP POLICY IF EXISTS messages_owner_read ON public.messages;
CREATE POLICY messages_owner_read ON public.messages FOR SELECT TO authenticated
USING (owner_id = (SELECT auth.uid()) AND EXISTS (
  SELECT 1 FROM public.chats c WHERE c.id = chat_id AND c.owner_id = (SELECT auth.uid())
));
DROP POLICY IF EXISTS messages_owner_create ON public.messages;
CREATE POLICY messages_owner_create ON public.messages FOR INSERT TO authenticated
WITH CHECK (owner_id = (SELECT auth.uid()) AND EXISTS (
  SELECT 1 FROM public.chats c WHERE c.id = chat_id AND c.owner_id = (SELECT auth.uid())
));

-- No anonymous/public role policies: bearer token required.
-- Keep the service_role key off the browser and out of version control.
