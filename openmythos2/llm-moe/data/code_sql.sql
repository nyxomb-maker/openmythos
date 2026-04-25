-- SQL Training Data
-- Database design, queries, optimization, and patterns.
-- Covering DDL, DML, views, indexes, CTEs, window functions.

-- ═══════════════════════════════════════════════════════════════════════
-- Schema Design
-- ═══════════════════════════════════════════════════════════════════════

CREATE TABLE users (
    id            BIGSERIAL PRIMARY KEY,
    username      VARCHAR(50) UNIQUE NOT NULL,
    email         VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    display_name  VARCHAR(100),
    bio           TEXT,
    avatar_url    VARCHAR(500),
    role          VARCHAR(20) DEFAULT 'user' CHECK (role IN ('user', 'admin', 'moderator')),
    is_verified   BOOLEAN DEFAULT FALSE,
    created_at    TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at    TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deleted_at    TIMESTAMP WITH TIME ZONE
);

CREATE TABLE posts (
    id          BIGSERIAL PRIMARY KEY,
    author_id   BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title       VARCHAR(500) NOT NULL,
    slug        VARCHAR(500) UNIQUE NOT NULL,
    content     TEXT NOT NULL,
    excerpt     VARCHAR(1000),
    status      VARCHAR(20) DEFAULT 'draft' CHECK (status IN ('draft', 'published', 'archived')),
    view_count  INTEGER DEFAULT 0,
    published_at TIMESTAMP WITH TIME ZONE,
    created_at  TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at  TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE tags (
    id   SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    slug VARCHAR(100) UNIQUE NOT NULL
);

CREATE TABLE post_tags (
    post_id BIGINT NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    tag_id  INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
    PRIMARY KEY (post_id, tag_id)
);

CREATE TABLE comments (
    id        BIGSERIAL PRIMARY KEY,
    post_id   BIGINT NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    author_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    parent_id BIGINT REFERENCES comments(id) ON DELETE CASCADE,
    content   TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE user_follows (
    follower_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    followed_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at  TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (follower_id, followed_id),
    CHECK (follower_id <> followed_id)
);

-- ═══════════════════════════════════════════════════════════════════════
-- Indexes
-- ═══════════════════════════════════════════════════════════════════════

CREATE INDEX idx_posts_author_id ON posts(author_id);
CREATE INDEX idx_posts_status_published ON posts(status, published_at DESC)
    WHERE status = 'published';
CREATE INDEX idx_posts_slug ON posts(slug);
CREATE INDEX idx_comments_post_id ON comments(post_id);
CREATE INDEX idx_comments_parent_id ON comments(parent_id);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_not_deleted ON users(id) WHERE deleted_at IS NULL;

-- Full text search index
CREATE INDEX idx_posts_search ON posts
    USING GIN (to_tsvector('english', title || ' ' || content));

-- ═══════════════════════════════════════════════════════════════════════
-- Views
-- ═══════════════════════════════════════════════════════════════════════

CREATE VIEW published_posts AS
SELECT
    p.id,
    p.title,
    p.slug,
    p.excerpt,
    p.content,
    p.view_count,
    p.published_at,
    u.username AS author_username,
    u.display_name AS author_display_name,
    u.avatar_url AS author_avatar,
    ARRAY_AGG(DISTINCT t.name) FILTER (WHERE t.name IS NOT NULL) AS tags,
    COUNT(DISTINCT c.id) AS comment_count
FROM posts p
JOIN users u ON p.author_id = u.id
LEFT JOIN post_tags pt ON p.id = pt.post_id
LEFT JOIN tags t ON pt.tag_id = t.id
LEFT JOIN comments c ON p.id = c.post_id
WHERE p.status = 'published'
  AND u.deleted_at IS NULL
GROUP BY p.id, u.username, u.display_name, u.avatar_url;

-- ═══════════════════════════════════════════════════════════════════════
-- Queries — CRUD
-- ═══════════════════════════════════════════════════════════════════════

-- Create user
INSERT INTO users (username, email, password_hash, display_name)
VALUES ('johndoe', 'john@example.com', '$2b$12$hash...', 'John Doe')
RETURNING id, username, email, created_at;

-- Get user profile with stats
SELECT
    u.id,
    u.username,
    u.display_name,
    u.bio,
    u.avatar_url,
    u.created_at,
    COUNT(DISTINCT p.id) AS post_count,
    COUNT(DISTINCT f1.followed_id) AS following_count,
    COUNT(DISTINCT f2.follower_id) AS follower_count
FROM users u
LEFT JOIN posts p ON u.id = p.author_id AND p.status = 'published'
LEFT JOIN user_follows f1 ON u.id = f1.follower_id
LEFT JOIN user_follows f2 ON u.id = f2.followed_id
WHERE u.id = 1 AND u.deleted_at IS NULL
GROUP BY u.id;

-- ═══════════════════════════════════════════════════════════════════════
-- Common Table Expressions (CTEs)
-- ═══════════════════════════════════════════════════════════════════════

-- User feed: posts from followed users, scored by recency and popularity
WITH user_feed AS (
    SELECT
        p.*,
        uf.follower_id AS viewer_id,
        -- Recency score (exponential decay)
        EXP(-EXTRACT(EPOCH FROM (NOW() - p.published_at)) / 86400.0) AS recency_score,
        -- Engagement score
        LOG(1 + p.view_count) + LOG(1 + (
            SELECT COUNT(*) FROM comments c WHERE c.post_id = p.id
        )) AS engagement_score
    FROM posts p
    JOIN user_follows uf ON p.author_id = uf.followed_id
    WHERE p.status = 'published'
),
ranked_feed AS (
    SELECT *,
        recency_score * 0.6 + engagement_score * 0.4 AS final_score,
        ROW_NUMBER() OVER (
            PARTITION BY viewer_id
            ORDER BY recency_score * 0.6 + engagement_score * 0.4 DESC
        ) AS rank
    FROM user_feed
)
SELECT * FROM ranked_feed
WHERE viewer_id = 1 AND rank <= 50
ORDER BY final_score DESC;

-- Recursive CTE: nested comment tree
WITH RECURSIVE comment_tree AS (
    -- Base: root comments
    SELECT
        c.id, c.content, c.author_id, c.parent_id,
        c.created_at, 0 AS depth,
        ARRAY[c.id] AS path
    FROM comments c
    WHERE c.post_id = 1 AND c.parent_id IS NULL

    UNION ALL

    -- Recursive: child comments
    SELECT
        c.id, c.content, c.author_id, c.parent_id,
        c.created_at, ct.depth + 1,
        ct.path || c.id
    FROM comments c
    JOIN comment_tree ct ON c.parent_id = ct.id
    WHERE ct.depth < 10  -- Max nesting level
)
SELECT
    ct.id,
    ct.content,
    u.username,
    ct.depth,
    REPEAT('  ', ct.depth) || ct.content AS indented_content,
    ct.created_at
FROM comment_tree ct
JOIN users u ON ct.author_id = u.id
ORDER BY ct.path;

-- ═══════════════════════════════════════════════════════════════════════
-- Window Functions
-- ═══════════════════════════════════════════════════════════════════════

-- Monthly post statistics with running totals
SELECT
    DATE_TRUNC('month', published_at) AS month,
    COUNT(*) AS posts_this_month,
    SUM(view_count) AS views_this_month,
    SUM(COUNT(*)) OVER (ORDER BY DATE_TRUNC('month', published_at)) AS cumulative_posts,
    AVG(view_count) OVER (
        ORDER BY DATE_TRUNC('month', published_at)
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS rolling_avg_views_3m,
    RANK() OVER (ORDER BY COUNT(*) DESC) AS rank_by_volume
FROM posts
WHERE status = 'published'
GROUP BY DATE_TRUNC('month', published_at)
ORDER BY month;

-- Top authors by engagement per month
SELECT
    u.username,
    DATE_TRUNC('month', p.published_at) AS month,
    COUNT(p.id) AS post_count,
    SUM(p.view_count) AS total_views,
    DENSE_RANK() OVER (
        PARTITION BY DATE_TRUNC('month', p.published_at)
        ORDER BY SUM(p.view_count) DESC
    ) AS monthly_rank
FROM posts p
JOIN users u ON p.author_id = u.id
WHERE p.status = 'published'
GROUP BY u.username, DATE_TRUNC('month', p.published_at);

-- ═══════════════════════════════════════════════════════════════════════
-- Full Text Search
-- ═══════════════════════════════════════════════════════════════════════

-- Search posts with ranking
SELECT
    p.id,
    p.title,
    p.excerpt,
    ts_rank(
        to_tsvector('english', p.title || ' ' || p.content),
        plainto_tsquery('english', 'transformer attention mechanism')
    ) AS relevance
FROM posts p
WHERE to_tsvector('english', p.title || ' ' || p.content)
      @@ plainto_tsquery('english', 'transformer attention mechanism')
  AND p.status = 'published'
ORDER BY relevance DESC
LIMIT 20;

-- ═══════════════════════════════════════════════════════════════════════
-- Stored Procedures / Functions
-- ═══════════════════════════════════════════════════════════════════════

CREATE OR REPLACE FUNCTION publish_post(p_post_id BIGINT, p_user_id BIGINT)
RETURNS BOOLEAN
LANGUAGE plpgsql
AS $$
DECLARE
    v_author_id BIGINT;
BEGIN
    SELECT author_id INTO v_author_id FROM posts WHERE id = p_post_id;

    IF v_author_id IS NULL THEN
        RAISE EXCEPTION 'Post % not found', p_post_id;
    END IF;

    IF v_author_id <> p_user_id THEN
        RAISE EXCEPTION 'User % is not the author of post %', p_user_id, p_post_id;
    END IF;

    UPDATE posts
    SET status = 'published',
        published_at = NOW(),
        updated_at = NOW()
    WHERE id = p_post_id AND status = 'draft';

    RETURN FOUND;
END;
$$;

-- Trigger: auto-update updated_at
CREATE OR REPLACE FUNCTION update_timestamp()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

CREATE TRIGGER tr_users_updated BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_timestamp();

CREATE TRIGGER tr_posts_updated BEFORE UPDATE ON posts
    FOR EACH ROW EXECUTE FUNCTION update_timestamp();
