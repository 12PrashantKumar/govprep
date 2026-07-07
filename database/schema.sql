CREATE TABLE chunks(
    id BIGSERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    subject TEXT,
    source TEXT,
    metadata JSONB,
    embedding VECTOR(768)
);

ALTER TABLE chunks ADD COLUMN ts tsvector
GENERATED ALWAYS AS (to_tsvector('english',content)) STORED;