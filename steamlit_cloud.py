# --start-- This fixes a streamlit cloud sqlite issue
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# --end-- hopefully

from main import main

if __name__ == "__main__":
    main()