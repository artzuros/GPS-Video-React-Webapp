CREATE TABLE videos (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    duration FLOAT
);

CREATE TABLE gps_points (
    id SERIAL PRIMARY KEY,
    video_id INTEGER REFERENCES videos(id) ON DELETE CASCADE,
    lat DOUBLE PRECISION,
    lon DOUBLE PRECISION,
    highlight BOOLEAN,
    timestamp FLOAT
);

CREATE TABLE inference_ing (
    id SERIAL PRIMARY KEY,
    video_id INTEGER REFERENCES videos(id) ON DELETE CASCADE
    inference_results_path TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    heatmap_path TEXT
);

