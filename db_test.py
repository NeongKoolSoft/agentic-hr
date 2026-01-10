import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "erp_sample.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS sales (
    id INTEGER PRIMARY KEY,
    product_name TEXT,
    amount INTEGER,
    sale_date TEXT
)
""")

sample_data = [
    (1, '넝쿨OS Pro', 550000, '2025-12-25'),
    (2, 'ERP 컨설팅', 1200000, '2025-12-26'),
    (3, '넝쿨OS Basic', 330000, '2025-12-27')
]

cursor.executemany("INSERT OR IGNORE INTO sales VALUES (?, ?, ?, ?)", sample_data)
conn.commit()
conn.close()

print("ERP 샘플 DB가 생성되었습니다:", DB_PATH)
