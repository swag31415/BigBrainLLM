# --start-- This fixes a streamlit cloud sqlite issue
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# --end-- hopefully

# runs RAG_Retrieval.py because of how python works
import src.app as app