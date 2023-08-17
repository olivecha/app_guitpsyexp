DROP TABLE IF EXISTS soundstats;

CREATE TABLE soundstats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    env_param_A FLOAT NOT NULL,
    env_param_B FLOAT NOT NULL,
    fav_sound TEXT NOT NULL,
    ability INT NOT NULL
);
