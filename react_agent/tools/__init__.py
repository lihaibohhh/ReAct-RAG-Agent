from react_agent.tools.search import search
from react_agent.tools.excel import make_excel_table
from react_agent.tools.rag import query_internal_knowledge

TOOLS = [search, make_excel_table, query_internal_knowledge]