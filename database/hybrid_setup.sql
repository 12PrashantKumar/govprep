-- 1. Add the text search column
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS search_vector tsvector;

-- 2. Populate it by parsing the English text from your chunks
UPDATE chunks SET search_vector = to_tsvector('english', content);

-- 3. Build a GIN index to make keyword searching lightning fast
CREATE INDEX IF NOT EXISTS ts_idx ON chunks USING GIN (search_vector);