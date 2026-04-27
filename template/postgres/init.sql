-- Creates the langflowdb database for Langflow's own storage.
-- This runs once when the postgres volume is first initialized.
CREATE DATABASE langflowdb;
GRANT ALL PRIVILEGES ON DATABASE langflowdb TO agent;
