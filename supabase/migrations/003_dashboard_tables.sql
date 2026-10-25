-- ============================================================================
-- SwasthyaAI Dashboard Tables Migration
-- Run this in your Supabase SQL Editor to create all tables needed for
-- full frontend-backend integration.
-- ============================================================================

-- 1. Report Approvals — persists approval state (replaces localStorage)
CREATE TABLE IF NOT EXISTS report_approvals (
    id          UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    call_id     TEXT NOT NULL,
    user_id     TEXT NOT NULL,
    approved    BOOLEAN NOT NULL DEFAULT TRUE,
    notes       TEXT,
    approved_at TIMESTAMPTZ DEFAULT now(),
    created_at  TIMESTAMPTZ DEFAULT now(),
    UNIQUE(call_id, user_id)
);

-- 2. Agent Configs — calling agent configuration persistence
CREATE TABLE IF NOT EXISTS agent_configs (
    id              UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id         TEXT NOT NULL,
    name            TEXT NOT NULL DEFAULT 'Default',
    config          JSONB NOT NULL DEFAULT '{}',
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_agent_configs_user ON agent_configs(user_id);

-- 3. User Preferences — language, voice, theme settings
CREATE TABLE IF NOT EXISTS user_preferences (
    id          UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id     TEXT NOT NULL UNIQUE,
    preferences JSONB NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT now(),
    updated_at  TIMESTAMPTZ DEFAULT now()
);

-- 4. Phone Registry — manual phone number entries
CREATE TABLE IF NOT EXISTS phone_registry (
    id              UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id         TEXT NOT NULL,
    phone_number    TEXT NOT NULL,
    provider        TEXT NOT NULL DEFAULT 'manual',
    label           TEXT,
    is_default      BOOLEAN NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ DEFAULT now(),
    UNIQUE(user_id, phone_number, provider)
);
CREATE INDEX IF NOT EXISTS idx_phone_registry_user ON phone_registry(user_id);

-- 5. Add approval-related columns to voice_calls if they don't exist
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'voice_calls' AND column_name = 'approved') THEN
        ALTER TABLE voice_calls ADD COLUMN approved BOOLEAN DEFAULT FALSE;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'voice_calls' AND column_name = 'approved_by') THEN
        ALTER TABLE voice_calls ADD COLUMN approved_by TEXT;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'voice_calls' AND column_name = 'approved_at') THEN
        ALTER TABLE voice_calls ADD COLUMN approved_at TIMESTAMPTZ;
    END IF;
END $$;
